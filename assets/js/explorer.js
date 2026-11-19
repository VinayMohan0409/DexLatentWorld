/* 3D explorer: retargeted demonstrations (ground truth) as particles with flow trails.
   Data: assets/data/explorer/manifest.json + int16 tracks (see DexLatentWorld-video/scenes/explorer_export.py).
   Needs three.js and OrbitControls (loaded before this file). */
(function () {
  var root = document.getElementById('explorer');
  if (!root) return;
  var stage = root.querySelector('.ex-stage'), note = root.querySelector('.ex-note');
  var BASE = 'assets/data/explorer/';
  var OBJECT = 0xf09600, TABLE = 0xb4bccd, DISTRACTOR = 0x9a9ca4, TRAIL_FRAMES = 12, TRAIL_POINTS = 260;

  function fail(msg) { root.classList.add('is-static'); note.textContent = msg; }
  if (!window.THREE || !THREE.OrbitControls) { fail('The 3D view could not be loaded.'); return; }
  var probe = document.createElement('canvas');
  if (!(probe.getContext('webgl') || probe.getContext('experimental-webgl'))) { fail('This browser cannot show the 3D view.'); return; }

  var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  stage.appendChild(renderer.domElement);
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(34, 16 / 9, 0.01, 20); camera.up.set(0, 0, 1);
  var controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08; controls.minDistance = 0.25; controls.maxDistance = 2.5; controls.maxPolarAngle = Math.PI * 0.49;

  // round, softly shaded particle sprite
  var sprite = (function () {
    var c = document.createElement('canvas'); c.width = c.height = 64; var g = c.getContext('2d');
    var grd = g.createRadialGradient(24, 22, 2, 32, 32, 30); grd.addColorStop(0, '#ffffff'); grd.addColorStop(0.55, '#e4e4e4'); grd.addColorStop(1, '#a8a8a8');
    g.fillStyle = grd; g.beginPath(); g.arc(32, 32, 30, 0, Math.PI * 2); g.fill();
    var t = new THREE.CanvasTexture(c); return t;
  })();
  function cloud(n, color, size) {
    var geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    var mat = new THREE.PointsMaterial({ color: color, size: size, map: sprite, alphaTest: 0.5, sizeAttenuation: true });
    var p = new THREE.Points(geo, mat); p.frustumCulled = false; scene.add(p); return p;
  }
  function rainbow(t) {   // blue -> green -> yellow -> red, as [r, g, b] in 0..1
    var c = new THREE.Color(); c.setHSL(0.68 * (1 - t), 0.9, 0.5); return [c.r, c.g, c.b];
  }

  var M = null, clip = 0, hand = 0, cache = {}, cur = null, playing = true, time = 0, last = 0, flow = true, visible = false;
  var handPts, objPts, tablePts, disPts, trails, trailIdx, trailCol;

  function load(url) {
    if (!cache[url]) cache[url] = fetch(BASE + url).then(function (r) { if (!r.ok) throw new Error(url); return r.arrayBuffer(); }).then(function (b) { return new Int16Array(b); });
    return cache[url];
  }
  function show(c, h, resetView) {
    var info = M.clips[c], key = M.hands[h].key.toLowerCase(), d = info.dir + '/';
    root.classList.add('is-loading');
    return Promise.all([load(d + 'hand_' + key + '.bin'), load(d + 'object.bin'), load(d + 'table.bin'), load(d + 'distractors.bin')]).then(function (a) {
      clip = c; hand = h; cur = { hand: a[0], object: a[1], scale: info.scale / 32767 };
      var s = cur.scale, i, col = M.hands[h].color;
      handPts.material.color.setRGB(col[0] / 255, col[1] / 255, col[2] / 255);
      function fill(points, src) { var p = new Float32Array(src.length); for (i = 0; i < src.length; i++) p[i] = src[i] * s; points.geometry.setAttribute('position', new THREE.BufferAttribute(p, 3)); points.geometry.attributes.position.needsUpdate = true; }
      fill(tablePts, a[2]); fill(disPts, a[3]); disPts.visible = a[3].length > 0;
      // trail colours: one rainbow colour per sampled point, by where it starts (left to right in the opening view)
      var nh = M.hand_points, no = M.object_points, n = nh + no, keyv = [], order;
      for (i = 0; i < TRAIL_POINTS; i++) { var id = trailIdx[i], src = id < nh ? a[0] : a[1], o = (id < nh ? id : id - nh) * 3; keyv.push([src[o + 1] * info.view[0] - src[o] * info.view[1], i]); }
      keyv.sort(function (p, q) { return p[0] - q[0]; });
      for (i = 0; i < TRAIL_POINTS; i++) trailCol[keyv[i][1]] = rainbow(i / (TRAIL_POINTS - 1));
      if (resetView) {   // frame the whole motion: centre and radius of the hand and object over the clip, seen from the recording camera's side
        var lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9], ax;   // (not `c`: that is this call's clip index)
        [a[0], a[1]].forEach(function (src) { for (i = 0; i + 2 < src.length; i += 21) for (ax = 0; ax < 3; ax++) { var val = src[i + ax] * s; if (val < lo[ax]) lo[ax] = val; if (val > hi[ax]) hi[ax] = val; } });
        var ctr = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2], rad = 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
        var vh = Math.hypot(info.view[0], info.view[1]) || 1, v = [info.view[0] / vh, info.view[1] / vh], dist = Math.max(0.45, rad / Math.tan(camera.fov * Math.PI / 360) * 1.0), el = 0.62;       // el: elevation, radians
        camera.position.set(ctr[0] + v[0] * dist * Math.cos(el), ctr[1] + v[1] * dist * Math.cos(el), ctr[2] + dist * Math.sin(el)); controls.target.set(ctr[0], ctr[1], ctr[2]); controls.update();
      }
      root.classList.remove('is-loading'); buttons(); draw();
    }).catch(function () { fail('The demonstration data could not be loaded.'); });
  }
  function draw() {
    if (!cur) return;
    var F = M.frames, nh = M.hand_points, no = M.object_points, s = cur.scale;
    var t = Math.min(F - 1.0001, Math.max(0, time)), f0 = Math.floor(t), a = t - f0, i;
    function lerp(points, src, n) {
      var p = points.geometry.attributes.position.array, o0 = f0 * n * 3, o1 = (f0 + 1) * n * 3;
      for (i = 0; i < n * 3; i++) p[i] = (src[o0 + i] + (src[o1 + i] - src[o0 + i]) * a) * s;
      points.geometry.attributes.position.needsUpdate = true;
    }
    lerp(handPts, cur.hand, nh); lerp(objPts, cur.object, no);
    trails.visible = flow;
    if (flow) {
      var pos = trails.geometry.attributes.position.array, col = trails.geometry.attributes.color.array, k = 0, seg, j;
      function at(src, n, id, tt, out, o) { var g0 = Math.floor(tt), b = tt - g0, q0 = (g0 * n + id) * 3, q1 = ((g0 + 1) * n + id) * 3; for (var c = 0; c < 3; c++) out[o + c] = (src[q0 + c] + (src[q1 + c] - src[q0 + c]) * b) * s; }
      for (j = 0; j < TRAIL_POINTS; j++) {
        var id = trailIdx[j], isHand = id < nh, src = isHand ? cur.hand : cur.object, n = isHand ? nh : no, pid = isHand ? id : id - nh, cc = trailCol[j];
        for (seg = 0; seg < TRAIL_FRAMES; seg++) {
          var t1 = Math.max(0, t - seg), t0 = Math.max(0, t - seg - 1), w1 = 1 - seg / TRAIL_FRAMES, w0 = 1 - (seg + 1) / TRAIL_FRAMES;   // newest segment first; fade to white with age
          at(src, n, pid, Math.min(F - 1.0001, t1), pos, k); at(src, n, pid, Math.min(F - 1.0001, t0), pos, k + 3);
          for (var c = 0; c < 3; c++) { col[k + c] = 1 - (1 - cc[c]) * w1; col[k + 3 + c] = 1 - (1 - cc[c]) * w0; }
          k += 6;
        }
      }
      trails.geometry.attributes.position.needsUpdate = true; trails.geometry.attributes.color.needsUpdate = true;
    }
    slider.value = Math.round(t / (F - 1) * 1000);
    clock.textContent = (t * M.step_seconds).toFixed(1) + ' s';
  }
  function size() { var w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  function tick(now) {
    requestAnimationFrame(tick);
    if (!visible || !cur) { last = now; return; }
    var dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (playing) { time += dt / M.step_seconds; if (time >= M.frames - 1) time = 0; draw(); }
    controls.update(); renderer.render(scene, camera);
  }

  // controls
  var clipBar = root.querySelector('.ex-clips'), handBar = root.querySelector('.ex-hands'), slider = root.querySelector('.ex-time'), clock = root.querySelector('.ex-clock');
  var playBtn = root.querySelector('.ex-play'), flowBtn = root.querySelector('.ex-flow');
  function buttons() {
    clipBar.querySelectorAll('button').forEach(function (b, i) { b.classList.toggle('on', i === clip); });
    handBar.querySelectorAll('button').forEach(function (b, i) { b.classList.toggle('on', i === hand); });
    playBtn.textContent = playing ? 'Pause' : 'Play'; flowBtn.classList.toggle('on', flow);
  }
  playBtn.addEventListener('click', function () { playing = !playing; buttons(); });
  flowBtn.addEventListener('click', function () { flow = !flow; buttons(); draw(); });
  slider.addEventListener('input', function () { playing = false; time = slider.value / 1000 * (M.frames - 1); buttons(); draw(); });

  fetch(BASE + 'manifest.json').then(function (r) { return r.json(); }).then(function (m) {
    M = m; M.clips = M.clips.filter(Boolean);
    handPts = cloud(M.hand_points, 0xffffff, 0.013); objPts = cloud(M.object_points, OBJECT, 0.013); tablePts = cloud(1, TABLE, 0.008); disPts = cloud(1, DISTRACTOR, 0.01);
    var n = M.hand_points + M.object_points, i; trailIdx = []; trailCol = [];
    for (i = 0; i < TRAIL_POINTS; i++) trailIdx.push(Math.floor((i + 0.5) / TRAIL_POINTS * n));
    var g = new THREE.BufferGeometry(), cnt = TRAIL_POINTS * TRAIL_FRAMES * 2;
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cnt * 3), 3)); g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cnt * 3), 3));
    trails = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true })); trails.frustumCulled = false; scene.add(trails);
    M.clips.forEach(function (c, i) { var b = document.createElement('button'); b.type = 'button'; b.textContent = c.name; b.addEventListener('click', function () { time = 0; show(i, hand, true); }); clipBar.appendChild(b); });
    M.hands.forEach(function (h, i) { var b = document.createElement('button'); b.type = 'button'; b.innerHTML = '<i style="background:rgb(' + h.color.join(',') + ')"></i>' + h.name; b.addEventListener('click', function () { show(clip, i, false); }); handBar.appendChild(b); });
    size(); window.addEventListener('resize', size);
    // load and animate only once the viewer is on screen
    new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (visible && !cur) show(0, 0, true); }, { rootMargin: '200px' }).observe(root);
    requestAnimationFrame(tick);
    root.__explorer = { show: show, set: function (t, p) { time = t; playing = p; buttons(); draw(); renderer.render(scene, camera); }, render: function () { controls.update(); renderer.render(scene, camera); } };
  }).catch(function () { fail('The demonstration data could not be loaded.'); });
})();

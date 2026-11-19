/* Latent-space map: a 2D embedding of latent actions (one point per transition and embodiment). Pointing at a region
   highlights it; clicking shows a ground-truth transition from that region on three embodiments, in 3D.
   Data: assets/data/latent/map.json + int16 tracks (DexLatentWorld-video/scenes/latent_map_export.py). Needs three.js + OrbitControls. */
(function () {
  var root = document.getElementById('latentmap');
  if (!root) return;
  var BASE = 'assets/data/latent/', cv = root.querySelector('.lm-map canvas'), ctx = cv.getContext('2d'), stage = root.querySelector('.lm-stage');
  var legend = root.querySelector('.lm-legend'), title = root.querySelector('.lm-title'), picks = root.querySelector('.lm-picks'), labels = root.querySelector('.lm-labels');
  var M = null, hover = -1, sel = 0, mot = 0, cur = null, t = 0, last = 0, visible = false, S = 1, PAD = 26;
  var OBJECT = 0xf09600, SCENE = 0xb4bccd, TRAIL = 8, TP = 120;

  // ---------- map ----------
  function px(v) { return PAD + v / 1000 * (S - 2 * PAD); }
  function drawMap() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2); S = cv.clientWidth; cv.width = cv.height = Math.round(S * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, S, S);
    var p = M.points, n = p.x.length, i, focus = hover >= 0 ? hover : sel;
    for (var pass = 0; pass < 2; pass++) {                    // dimmed points first, the focused region on top
      for (i = 0; i < n; i++) {
        var on = p.r[i] === focus; if ((pass === 1) !== on) continue;
        var c = M.embodiments[p.e[i]].color; ctx.fillStyle = 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + (on ? 0.85 : 0.16) + ')';
        ctx.beginPath(); ctx.arc(px(p.x[i]), S - px(p.y[i]), on ? 2.1 : 1.6, 0, 6.2832); ctx.fill();
      }
    }
    M.regions.forEach(function (r, k) {
      var x = px(r.centre[0]), y = S - px(r.centre[1]), on = k === sel, hv = k === hover;
      ctx.beginPath(); ctx.arc(x, y, 13, 0, 6.2832); ctx.fillStyle = on ? '#1d3f94' : '#ffffff'; ctx.fill();
      ctx.lineWidth = hv || on ? 2 : 1; ctx.strokeStyle = hv || on ? '#1d3f94' : '#8a909a'; ctx.stroke();
      ctx.fillStyle = on ? '#ffffff' : '#15171b'; ctx.font = '500 12px Jost, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(r.n, x, y + 0.5);
    });
    if (cur) { var m = M.regions[sel].motions[mot]; ctx.beginPath(); ctx.arc(px(m.centre[0]), S - px(m.centre[1]), 5, 0, 6.2832); ctx.fillStyle = '#15171b'; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke(); }
  }
  function regionAt(ev) {                                     // nearest region centre to the pointer
    var b = cv.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top, best = -1, bd = 1e9;
    M.regions.forEach(function (r, k) { var d = Math.hypot(px(r.centre[0]) - x, S - px(r.centre[1]) - y); if (d < bd) { bd = d; best = k; } });
    return best;
  }
  cv.addEventListener('mousemove', function (ev) { var k = regionAt(ev); if (k !== hover) { hover = k; drawMap(); } });
  cv.addEventListener('mouseleave', function () { hover = -1; drawMap(); });
  cv.addEventListener('click', function (ev) { choose(regionAt(ev), 0); });

  // ---------- 3D: one canvas, three views of the same transition ----------
  var ok = window.THREE && THREE.OrbitControls && (function () { var c = document.createElement('canvas'); return !!(c.getContext('webgl') || c.getContext('experimental-webgl')); })();
  var renderer, camera, controls, views = [], sprite;
  function initGL() {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.autoClear = false; stage.appendChild(renderer.domElement);
    camera = new THREE.PerspectiveCamera(34, 1, 0.01, 20); camera.up.set(0, 0, 1);
    controls = new THREE.OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.dampingFactor = 0.08; controls.enablePan = false; controls.minDistance = 0.2; controls.maxDistance = 4; controls.maxPolarAngle = Math.PI * 0.49;
    var c = document.createElement('canvas'); c.width = c.height = 64; var g = c.getContext('2d'), grd = g.createRadialGradient(24, 22, 2, 32, 32, 30);
    grd.addColorStop(0, '#fff'); grd.addColorStop(0.55, '#e4e4e4'); grd.addColorStop(1, '#a8a8a8'); g.fillStyle = grd; g.beginPath(); g.arc(32, 32, 30, 0, 6.2832); g.fill(); sprite = new THREE.CanvasTexture(c);
    M.shown.forEach(function (h) {
      var sc = new THREE.Scene();
      function cloud(n, color, size) { var geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3)); var p = new THREE.Points(geo, new THREE.PointsMaterial({ color: color, size: size, map: sprite, alphaTest: 0.5 })); p.frustumCulled = false; sc.add(p); return p; }
      var v = { scene: sc, hand: cloud(M.hand_points, new THREE.Color(h.color[0] / 255, h.color[1] / 255, h.color[2] / 255), 0.015), object: cloud(M.object_points, OBJECT, 0.015), rest: cloud(1, SCENE, 0.009) };
      var geo = new THREE.BufferGeometry(), cnt = TP * TRAIL * 2; geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cnt * 3), 3)); geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cnt * 3), 3));
      v.trails = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true })); v.trails.frustumCulled = false; sc.add(v.trails); views.push(v);
      var s = document.createElement('span'); s.innerHTML = '<i style="background:rgb(' + h.color.join(',') + ')"></i>' + h.name + (h.held_out ? ' <em>unseen in training</em>' : ''); labels.appendChild(s);
    });
    size(); window.addEventListener('resize', function () { size(); drawMap(); });
  }
  function size() { if (!renderer) return; renderer.setSize(stage.clientWidth, stage.clientHeight, false); }
  var cache = {};
  function load(u) { if (!cache[u]) cache[u] = fetch(BASE + u).then(function (r) { if (!r.ok) throw new Error(u); return r.arrayBuffer(); }).then(function (b) { return new Int16Array(b); }); return cache[u]; }
  function choose(k, m) {
    sel = k; mot = m; var R = M.regions[k], info = R.motions[m];
    title.innerHTML = 'Region ' + R.n + ' <span>' + R.windows + ' transitions &times; ' + M.embodiments.length + ' embodiments</span>';
    picks.innerHTML = ''; R.motions.forEach(function (mm, i) { var b = document.createElement('button'); b.type = 'button'; b.textContent = 'Example ' + (i + 1); if (i === m) b.className = 'on'; b.addEventListener('click', function () { choose(k, i); }); picks.appendChild(b); });
    drawMap(); if (!ok) return;
    root.classList.add('is-loading');
    Promise.all(M.shown.map(function (h) { return load(info.id + '_' + h.key + '.bin'); }).concat([load(info.id + '_scene.bin')])).then(function (a) {
      if (sel !== k || mot !== m) return;
      var s = info.scale / 32767, n = M.hand_points + M.object_points, i, lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9], ax, rest = a[a.length - 1];
      cur = { tracks: a.slice(0, M.shown.length), s: s }; t = 0;
      views.forEach(function (v) { var p = new Float32Array(rest.length); for (i = 0; i < rest.length; i++) p[i] = rest[i] * s; v.rest.geometry.setAttribute('position', new THREE.BufferAttribute(p, 3)); v.rest.visible = rest.length > 0; });
      cur.tracks.forEach(function (src) { for (i = 0; i + 2 < src.length; i += 15) for (ax = 0; ax < 3; ax++) { var val = src[i + ax] * s; if (val < lo[ax]) lo[ax] = val; if (val > hi[ax]) hi[ax] = val; } });
      var ctr = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2], rad = 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), asp = Math.min(1, stage.clientWidth / views.length / stage.clientHeight), dist = Math.max(0.4, rad / (Math.tan(17 * Math.PI / 180) * asp) * 0.72), el = 0.6, vh = Math.hypot(info.view[0], info.view[1]) || 1, vw = [info.view[0] / vh, info.view[1] / vh];   // each view is taller than wide: fit the motion to its width
      camera.position.set(ctr[0] + vw[0] * dist * Math.cos(el), ctr[1] + vw[1] * dist * Math.cos(el), ctr[2] + dist * Math.sin(el)); controls.target.set(ctr[0], ctr[1], ctr[2]); controls.update();
      root.classList.remove('is-loading'); drawMap();
    }).catch(function () { root.classList.add('is-static'); });
  }
  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || !cur || !renderer) { last = now; return; }
    var dt = Math.min(0.1, (now - last) / 1000); last = now; var F = M.frames, hold = 0.6 / M.step_seconds;
    t += dt / M.step_seconds * 0.6; if (t > F - 1 + hold) t = 0;            // 0.6x speed, short hold on the last frame
    var tt = Math.min(F - 1.0001, t), f0 = Math.floor(tt), a = tt - f0, nh = M.hand_points, no = M.object_points, n = nh + no, s = cur.s, i;
    views.forEach(function (v, vi) {
      var src = cur.tracks[vi], o0 = f0 * n * 3, o1 = (f0 + 1) * n * 3, hp = v.hand.geometry.attributes.position.array, op = v.object.geometry.attributes.position.array;
      for (i = 0; i < nh * 3; i++) hp[i] = (src[o0 + i] + (src[o1 + i] - src[o0 + i]) * a) * s;
      for (i = 0; i < no * 3; i++) op[i] = (src[o0 + nh * 3 + i] + (src[o1 + nh * 3 + i] - src[o0 + nh * 3 + i]) * a) * s;
      v.hand.geometry.attributes.position.needsUpdate = v.object.geometry.attributes.position.needsUpdate = true;
      var pos = v.trails.geometry.attributes.position.array, col = v.trails.geometry.attributes.color.array, k = 0, j, seg, c, col3 = new THREE.Color();
      for (j = 0; j < TP; j++) {
        var id = Math.floor((j + 0.5) / TP * n); col3.setHSL(0.68 * (1 - j / (TP - 1)), 0.9, 0.5);
        for (seg = 0; seg < TRAIL; seg++) {
          var t1 = Math.max(0, tt - seg), t0 = Math.max(0, tt - seg - 1), w1 = 1 - seg / TRAIL, w0 = 1 - (seg + 1) / TRAIL, ends = [t1, t0];
          for (var e2 = 0; e2 < 2; e2++) { var g0 = Math.floor(ends[e2]), b = ends[e2] - g0, q0 = (g0 * n + id) * 3, q1 = (Math.min(F - 1, g0 + 1) * n + id) * 3; for (c = 0; c < 3; c++) pos[k + e2 * 3 + c] = (src[q0 + c] + (src[q1 + c] - src[q0 + c]) * b) * s; }
          col[k] = 1 - (1 - col3.r) * w1; col[k + 1] = 1 - (1 - col3.g) * w1; col[k + 2] = 1 - (1 - col3.b) * w1; col[k + 3] = 1 - (1 - col3.r) * w0; col[k + 4] = 1 - (1 - col3.g) * w0; col[k + 5] = 1 - (1 - col3.b) * w0; k += 6;
        }
      }
      v.trails.geometry.attributes.position.needsUpdate = v.trails.geometry.attributes.color.needsUpdate = true;
    });
    controls.update(); var W = stage.clientWidth, H = stage.clientHeight, w = W / views.length; renderer.setScissorTest(true); renderer.clear();
    camera.aspect = w / H; camera.updateProjectionMatrix();
    views.forEach(function (v, vi) { renderer.setViewport(vi * w, 0, w, H); renderer.setScissor(vi * w, 0, w, H); renderer.render(v.scene, camera); });
  }

  fetch(BASE + 'map.json').then(function (r) { return r.json(); }).then(function (m) {
    M = m;
    M.embodiments.forEach(function (e) { var s = document.createElement('span'); s.innerHTML = '<i style="background:rgb(' + e.color.join(',') + ')"></i>' + e.name + (e.held_out ? '*' : ''); legend.appendChild(s); });
    if (ok) initGL(); else root.classList.add('is-static');
    new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (visible && !cur) choose(sel, 0); }, { rootMargin: '200px' }).observe(root);
    drawMap(); requestAnimationFrame(frame);
    root.__latent = { choose: choose, state: function () { return [sel, mot, !!cur]; }, views: views };
  }).catch(function () { root.classList.add('is-static'); });
})();

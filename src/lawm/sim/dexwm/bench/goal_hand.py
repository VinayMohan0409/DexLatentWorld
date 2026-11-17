"""The goal hand from the goal frame alone (DexWM's planner never sees goal joint angles)."""
# SAM 3 ("black-coloured robot hand") masks the robot in the goal RGB-D frame. It covers the whole arm, so the whole robot mesh (right arm 7 + Allegro 16 joints;
# base and left arm fixed at their start values from proprioception) is fitted to the back-projected mask points. The hand is then read off the fitted joint angles
# as the HandKinematics barycentric samples, i.e. the same rows as the FK hand, so the hand costs take it unchanged.
from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from lawm.sim import clouds

PROMPTS = ("black-coloured robot hand", "robot", "robotic hand")   # generic, no task or object words
N_INSTANCES = 5                  # SAM instances kept per prompt (SAM often returns the arm and the hand as separate instances, the arm ranked first)
VOXEL = 0.01
ROBOT_Q = np.arange(49)          # base 0-2, right arm 3-9, right hand 10-25, left arm + hand 26-48 (qpos == dof addresses in this model)
FIT = np.arange(3, 26)           # fitted: right arm + right hand
FIXED_MARGIN = 0.02              # observed points within this of the fixed robot parts (base, torso, left arm) are dropped


def robot_mask(view, instances, touch_px=2, link=0.05):
    """The robot mask of one view from SAM instances (list of (mask, score, prompt)), no score threshold."""
    # 1. arm: the union of each prompt's top instance, cleaned (clean_mask: background bleed and far blobs removed); the robot is always among them;
    # 2. attached pieces: every other instance that touches it in the image (within touch_px) and continues it in depth (some of its points within `link` of it)
    #    is added, so a hand SAM returned as a separate, lower-ranked instance is not lost; no agreement between prompts is required;
    # 3. the union is cleaned again. Returns (mask, list of (instance index, mask) of the added pieces) - the fit later checks the added pieces against the model.
    from scipy.ndimage import binary_dilation
    shape = view[1].shape
    top = {}
    for i, (m, sc, pr) in enumerate(instances):
        if pr not in top or sc > instances[top[pr]][1]:
            top[pr] = i
    core = np.zeros(shape, bool)
    for i in top.values():
        core |= instances[i][0]
    core = clean_mask(view, core)
    if core.sum() < 30:                                   # no prompt's top instance holds the robot: fall back to all instances
        allm = np.zeros(shape, bool)
        for m, _, _ in instances:
            allm |= m
        return clean_mask(view, allm), []
    near = binary_dilation(core, iterations=touch_px)
    tree = cKDTree(mask_points(view, core))
    added = []
    for i, (m, sc, pr) in enumerate(instances):
        new = m & ~core
        if i in top.values() or new.sum() < 10 or not (m & near).any():
            continue
        p = mask_points(view, clean_mask(view, m))
        if len(p) and tree.query(p)[0].min() < link:
            added.append((i, new))
    mask = core.copy()
    for _, m in added:
        mask |= m
    return clean_mask(view, mask), added


def clean_mask(view, mask, win=5, behind=0.03, link=0.10):
    """The robot mask with what is not robot removed, without thinning the robot (the fingers are only a few pixels wide; erosion or a depth-edge cut removes about half of them)."""
    # 1. bleed: a mask pixel more than `behind` deeper than the median depth of the masked pixels in its win x win window is background seen past the mask edge;
    # 2. blobs: of the 3D connected pieces (2 cm), the largest is the robot; pieces within `link` of the kept set are kept too (a hand separated from the
    #    arm by a few missing pixels), anything farther is dropped (e.g. a dark object SAM took for the hand).
    from numpy.lib.stride_tricks import sliding_window_view
    _, depth, K, pose = view
    m = mask & (depth > 0.1) & (depth < 3.0)
    v, u = np.nonzero(m)
    if len(v) == 0:
        return m
    h = win // 2
    d = np.pad(np.where(m, depth, np.nan), h, constant_values=np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(sliding_window_view(d, (win, win))[v, u].reshape(len(v), -1), axis=1)   # only at mask pixels
    m[v[depth[v, u] > med + behind], u[depth[v, u] > med + behind]] = False
    v, u = np.nonzero(m)
    if len(v) < 30:
        return m
    pts = backproject(view, v, u)
    from lawm.sim.dexwm.planspace.object_track import clusters
    lab = clusters(pts, 0.02)
    sizes = np.bincount(lab)
    keep = lab == np.argmax(sizes)
    while True:
        tree = cKDTree(pts[keep])
        added = False
        for c in np.unique(lab[~keep]):
            sel = lab == c
            if tree.query(pts[sel])[0].min() < link:
                keep |= sel
                added = True
        if not added:
            break
    out = np.zeros_like(m)
    out[v[keep], u[keep]] = True
    return out


def backproject(view, v, u):
    """World points of the pixels (v, u) of a view (rgb, depth, K, pose)."""
    _, depth, K, pose = view
    z = depth[v, u]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1) @ pose[:3, :3].T + pose[:3, 3]


def mask_points(view, mask):
    """World points of the mask pixels with valid depth (clean the mask first with clean_mask)."""
    m = mask & (view[1] > 0.1) & (view[1] < 3.0)
    v, u = np.nonzero(m)
    return backproject(view, v, u)


class RobotFit:
    """The right arm + hand of the compiled robot model (robot.mjb + meta.pkl from save_model.py), fitted to observed robot points.
    Only the right arm and hand are sampled and fitted; the base, torso and left arm never move and are only used to drop observed points that lie on them."""

    def __init__(self, mjb, meta, n_points=6000):
        """Load the compiled robot model and set up the surface samplers of its moving (right arm + hand) and fixed parts."""
        meta = pickle.load(open(meta, "rb"))
        self.kin = meta["kin"]
        self.m = mujoco.MjModel.from_binary_path(str(mjb))
        self.d = mujoco.MjData(self.m)
        self.lim = self.m.jnt_range[FIT].copy()
        moving_body = np.zeros(self.m.nbody, bool)   # bodies moved by the fitted joints (right arm + hand)
        for b in range(self.m.nbody):
            a = b
            while a > 0:
                if np.any(self.m.jnt_bodyid[FIT] == a):
                    moving_body[b] = True
                    break
                a = self.m.body_parentid[a]
        geoms = np.asarray(meta["robot_geoms"])
        moving_geoms = geoms[moving_body[self.m.geom_bodyid[geoms]]]
        self.sampler_geoms = moving_geoms
        self.sampler = clouds.SurfaceSampler(self.m, moving_geoms, n_points, np.random.default_rng(0))
        self.body_of = self.m.geom_bodyid[self.sampler.owner]
        self.root_body = self.kin.bodies[self.kin.root]
        self.arm_base = int(self.m.jnt_bodyid[FIT[0]])
        fixed = clouds.SurfaceSampler(self.m, geoms[~moving_body[self.m.geom_bodyid[geoms]]], 100000, np.random.default_rng(1))
        self.fixed_sampler = fixed

    def _fk(self, q):
        """Surface points of the moving robot parts at robot joint angles q (forward kinematics)."""
        self.d.qpos[ROBOT_Q] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.sampler(self.d).astype(float)

    def _root_T(self):
        """4x4 pose of the hand root body at the current kinematics."""
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[self.root_body].reshape(3, 3)
        T[:3, 3] = self.d.xpos[self.root_body]
        return T

    def _point_jac(self, P, bodies):
        """Translational Jacobians (n, 3, len(FIT)) of world points P rigidly attached to `bodies`: one mj_jac per body at its origin, then J(p) = Jp(body) + Jr(body) x (p - body origin) for every point (equal to mj_jac at each point)."""
        m, d = self.m, self.d
        out = np.empty((len(P), 3, len(FIT)))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for b in np.unique(bodies):
            sel = bodies == b
            mujoco.mj_jac(m, d, jp, jr, d.xpos[b], b)
            r = P[sel] - d.xpos[b]
            out[sel] = jp[:, FIT][None] + np.cross(jr[:, FIT].T[None], r[:, None, :]).transpose(0, 2, 1)
        return out

    def _fit_once(self, q0, Y, iters=60, delta0=0.10, delta1=0.01, n_sub=1500, sampler=None):
        """Damped Gauss-Newton on the right arm + hand joints, each observed point to its nearest model point, Huber loss with its scale shrinking delta0 -> delta1. Returns (q, median residual m)."""
        m, d = self.m, self.d
        sampler = sampler or self.sampler
        body_of = self.m.geom_bodyid[sampler.owner]
        q = q0.copy()
        for it in range(iters):
            self._fk(q); P = sampler(d).astype(float)
            dist, k = cKDTree(P).query(Y)
            delta = delta0 * (delta1 / delta0) ** (it / (iters - 1))
            w = np.where(dist <= delta, 1.0, delta / np.maximum(dist, 1e-9))
            sub = np.random.default_rng(it).choice(len(Y), min(len(Y), n_sub), replace=False)
            i = k[sub]
            J = self._point_jac(P[i], body_of[i]).reshape(-1, len(FIT))
            r = (Y[sub] - P[i]).reshape(-1)
            W = np.repeat(w[sub], 3)
            H = J.T @ (J * W[:, None])
            g = J.T @ (W * r)
            dq = np.linalg.solve(H + 1e-2 * np.diag(np.diag(H)) + 1e-6 * np.eye(len(FIT)), g)
            q[FIT] = np.clip(q[FIT] + np.clip(dq, -0.15, 0.15), self.lim[:, 0], self.lim[:, 1])
        self._fk(q); P = sampler(d).astype(float)
        return q, float(np.median(cKDTree(P).query(Y)[0]))

    def _ik_far_to(self, q0, base, target, far_frac, iters=60):
        """Arm joints (7) that put the centroid of the model's farthest `far_frac` points from the arm base onto `target` (the same statistic of the observed points), damped least squares from q0; hand joints unchanged."""
        # Matching like with like: both sides are "the far end of the robot".
        q = q0.copy()
        jacp = np.zeros((3, self.m.nv))
        arm = FIT[:7]
        for _ in range(iters):
            P = self._fk(q)
            r = np.linalg.norm(P - base, axis=1)
            far = np.flatnonzero(r >= np.quantile(r, 1 - far_frac))
            e = target - P[far].mean(0)
            if np.linalg.norm(e) < 1e-3:
                break
            sub = far[:: max(1, len(far) // 60)]
            J = np.zeros((3, 7))
            for i in sub:
                mujoco.mj_jac(self.m, self.d, jacp, None, P[i], self.body_of[i])
                J += jacp[:, arm]
            J /= len(sub)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), e)
            q[arm] = np.clip(q[arm] + np.clip(dq, -0.2, 0.2), self.lim[:7, 0], self.lim[:7, 1])
        return q

    def seeds(self, q0, Y, far_frac=0.15, twists=(0.0, np.pi / 2, np.pi, -np.pi / 2)):
        """Starting poses from the data: the hand is the end of the visible robot farthest from the arm base."""
        # IK moves the model's far end (its farthest
        # `far_frac` points from the base) onto the data's far end, then the last wrist joint is turned by each of `twists`.
        self._fk(q0)
        base = self.d.xpos[self.arm_base].copy()
        r = np.linalg.norm(Y - base, axis=1)
        target = Y[r >= np.quantile(r, 1 - far_frac)].mean(0)
        q = self._ik_far_to(q0, base, target, far_frac)
        out = []
        for tw in twists:
            qq = q.copy()
            j = FIT[6]
            qq[j] = np.clip(q[j] + tw, self.lim[6, 0], self.lim[6, 1])
            out.append(qq)
        return out, target

    def model_to_view(self, q, T, view, mask, occl=0.015, trunc=0.05):
        """Mean cost (m) of the dense model at joint angles q against the view: each model point not hidden behind measured geometry must sit on a mask pixel at the measured depth (|dz|, capped), else it costs `trunc`."""
        # Points out of view or hidden cost nothing.
        _, depth, K, pose = view
        self._fk(q)
        P = self.dense_sampler(self.d).astype(float) @ T[:3, :3].T + T[:3, 3]
        pc = (P - pose[:3, 3]) @ pose[:3, :3]
        z = pc[:, 2]
        H, W = depth.shape
        u = np.round(K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(int)
        ins = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        D, zi, on = depth[v[ins], u[ins]], z[ins], mask[v[ins], u[ins]]
        vis = D > zi - occl
        c = np.where(on, np.minimum(np.abs(D - zi), trunc), trunc)[vis]
        return float(c.mean()) if len(c) else trunc

    def view_cost_points(self, P, T, view, mask, occl=0.015, trunc=0.05):
        """model_to_view for given model points P (model frame), e.g. the coarse sample."""
        _, depth, K, pose = view
        pc = (P @ T[:3, :3].T + T[:3, 3] - pose[:3, 3]) @ pose[:3, :3]
        z = pc[:, 2]
        H, W = depth.shape
        u = np.round(K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(int)
        ins = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        D, zi, on = depth[v[ins], u[ins]], z[ins], mask[v[ins], u[ins]]
        c = np.where(on, np.minimum(np.abs(D - zi), trunc), trunc)[D > zi - occl]
        return float(c.mean()) if len(c) else trunc

    def arm_search(self, q0, Y, T, view, mask, restarts=8, pop=96, iters=12, n_elite=10, keep=4, seed=0):
        """Global starts for the arm: multi-start CEM over the 7 arm joints (fingers at the start angles) on the fit's own score (data -> model median distance + model -> view), restart 0 from the start pose, the others from random arm configurations; returns up to `keep` distinct results."""
        # The far-end seeds all start from the start configuration and miss goal arms in another configuration (elbow / wrist re-arranged).
        rng = np.random.default_rng(seed)
        arm, lo, hi = FIT[:7], self.lim[:7, 0], self.lim[:7, 1]

        def score(qa):
            """Fit score of arm joints qa (other joints from q0): median data -> model distance plus the model -> view cost."""
            q = q0.copy()
            q[arm] = qa
            P = self._fk(q)
            return float(np.median(cKDTree(P).query(Y)[0])) + self.view_cost_points(P, T, view, mask)
        found = []
        for k in range(restarts):
            mu, sd = (q0[arm].copy() if k == 0 else rng.uniform(lo, hi)), (hi - lo) / 4
            for _ in range(iters):
                X = np.clip(rng.normal(mu, sd, (pop, 7)), lo, hi)
                X[0] = mu
                elite = X[np.argsort([score(x) for x in X])[:n_elite]]
                mu, sd = elite.mean(0), np.maximum(elite.std(0), 0.01)
            found.append((score(mu), mu))
        out = []
        for _, qa in sorted(found, key=lambda f: f[0]):
            if all(np.abs(qa - o[arm]).max() > 0.3 for o in out):
                q = q0.copy()
                q[arm] = qa
                out.append(q)
            if len(out) == keep:
                break
        return out

    def fit(self, points, start_q, start_root, refine=True, view=None, mask=None, far_fracs=(0.05, 0.15, 0.3), n_fine=2):
        """Goal hand from observed robot points (world frame)."""
        # start_q: the robot's qpos at the start (ROBOT_Q entries; proprioception); start_root: 4x4 world pose of the hand root at the start (FK), which places the
        # model in the world. Starts: the start pose + far-end IK seeds + arm_search starts; all get a coarse fit, the best n_fine a fine fit; lowest residual + model_to_view wins.
        # Returns dict(hand=(n, 3) barycentric hand rows, hand_q, hand_root 4x4, robot_q, resid_mm, n_points, seed) or None if too few points.
        q0 = np.asarray(start_q, float)
        self._fk(q0)
        T = np.asarray(start_root) @ np.linalg.inv(self._root_T())          # model -> world
        Ti = np.linalg.inv(T)
        Y = np.asarray(points, float) @ Ti[:3, :3].T + Ti[:3, 3]
        Y = Y[cKDTree(self.fixed_sampler(self.d)).query(Y)[0] > FIXED_MARGIN]  # observed points on the base / torso / left arm say nothing about the right arm
        if len(Y) < 30:
            return None
        Yv = Y[np.sort(np.unique(np.floor(Y / VOXEL).astype(np.int64), axis=0, return_index=True)[1])]
        if not hasattr(self, "dense_sampler"):
            self.dense_sampler = clouds.SurfaceSampler(self.m, self.sampler_geoms, 30000, np.random.default_rng(2))
        # starts: the start pose, and IK of the model's far end onto the data's far end for several far-end sizes (a held object or a gap can shift the
        # data's far end, so no single size is trusted), each with 4 wrist twists. All get the coarse fit; the best n_fine go on to the fine fit.
        seeds = [q0] + [s for ff in far_fracs for s in self.seeds(q0, Yv, far_frac=ff)[0]]
        if view is not None:
            seeds += self.arm_search(q0, Yv, T, view, mask)
        view_cost = (lambda q: self.model_to_view(q, T, view, mask)) if view is not None else (lambda q: 0.0)
        coarse = []
        for i, qs in enumerate(seeds):
            qc, rc = self._fit_once(qs, Yv)                     # coarse: 6000-point model, 1 cm voxels
            coarse.append((rc + view_cost(qc), i, qc, rc))
        coarse.sort(key=lambda c: c[0])
        best = None
        for _, i, qc, rc in coarse[: n_fine if refine else 1]:
            q, res = (self._fit_once(qc, Y, iters=30, delta0=0.01, delta1=0.003, n_sub=4000, sampler=self.dense_sampler)  # fine: 30k-point model, all points
                      if refine else (qc, rc))
            # data -> model (fine residual) plus model -> data: the fitted robot must also land on the robot pixels at the measured depth
            score = res + view_cost(q)
            if best is None or score < best[5]:
                best = (q, res, i, qc, rc, score)
        q, res, i, qc, coarse_res, _ = best
        self._fk(qc); Tc = T @ self._root_T(); coarse_hand = self.kin.points(Tc[:3, :3], Tc[:3, 3], qc[self.kin.qpos_adr])
        self._fk(q)
        Tw = T @ self._root_T()
        hand_q = q[self.kin.qpos_adr]
        return dict(hand=self.kin.points(Tw[:3, :3], Tw[:3, 3], hand_q), hand_q=hand_q, hand_root=Tw, robot_q=q, resid_mm=res * 1000, coarse_resid_mm=coarse_res * 1000, coarse_hand=coarse_hand, n_points=len(Y), seed=i, model_to_world=T)


def model_distance(rf, r, points):
    """Distance (m) of world points to the fitted right arm + hand surface (dense model) of fit result r."""
    T = r["model_to_world"]
    rf._fk(r["robot_q"])
    P = rf.dense_sampler(rf.d).astype(float) @ T[:3, :3].T + T[:3, 3]
    return cKDTree(P).query(points)[0]


def goal_hand(rf, view, instances, start_q, start_root, piece_tol=0.02, hand_tol=0.01, min_hand_points=50):
    """The goal hand from the goal RGB-D view and SAM instances."""
    # Information used: the goal frame (RGB-D, camera intrinsics and pose), SAM 3 with generic
    # prompts, the robot's own model, and the robot's start joint angles / start hand pose (proprioception). Nothing about the goal state itself.
    # 1. robot mask: arm + attached pieces (robot_mask);
    # 2. fit the right arm + hand (RobotFit.fit, with the mask for the two-way seed choice);
    # 3. model check: an attached piece whose points the fitted robot does not explain (median distance > piece_tol), e.g. a held object, is dropped and the
    #    fit is redone without it;
    # 4. flag: hand_points = observed points within hand_tol of the fitted hand; below min_hand_points the hand is not seen and only the arm constrains it.
    # Returns the fit dict plus mask, n_pieces_kept, n_pieces_dropped, hand_points, hand_seen; or None.
    mask, added = robot_mask(view, instances)
    pts = mask_points(view, mask)
    if len(pts) < 30:
        return None
    r = rf.fit(pts, start_q, start_root, view=view, mask=mask)
    if r is None:
        return None
    dropped = [i for i, m in added if len(p := mask_points(view, clean_mask(view, m))) and np.median(model_distance(rf, r, p)) > piece_tol]
    if dropped:
        mask, _ = robot_mask(view, [inst for i, inst in enumerate(instances) if i not in dropped])  # the arm core is unchanged (top instances are never 'added')
        pts2 = mask_points(view, mask)
        r2 = rf.fit(pts2, start_q, start_root, view=view, mask=mask) if len(pts2) >= 30 else None
        if r2 is not None:
            pts, r = pts2, r2
        else:
            mask, _ = robot_mask(view, instances)
    hand = r["hand"]
    hp = int((cKDTree(hand).query(pts)[0] < hand_tol).sum())
    r.update(mask=mask, n_pieces_kept=len(added) - len(dropped), n_pieces_dropped=len(dropped), hand_points=hp, hand_seen=hp >= min_hand_points)
    if not hasattr(rf, "dense_sampler"):
        rf.dense_sampler = clouds.SurfaceSampler(rf.m, rf.sampler_geoms, 30000, np.random.default_rng(2))
    r["checks"], r["flags"] = goal_checks(rf, r, view, mask, pts, r.get("candidates"))   # truth-free; the planner refuses a goal with flags
    return r


# ---- truth-free checks of a fitted goal hand (the planner refuses a goal with any flag; thresholds: see CHECK_LIMITS) ----
# Calibrated on 5 demos x reach/grasp/place: 25 good fits (torch + GN, err <= 28.6 mm) vs the 5 known-bad GN fits (34-163 mm).
# Good-fit ranges in brackets.
CHECK_LIMITS = dict(
    score_mm=25.0,             # max: fit score (median data->model + model_to_view) [good 11.1-19.6; bad 34.6-45.8, and 18.6 for place 1364]
    unexplained=0.04,          # max: fraction of observed robot points > 2 cm from the fitted robot (the median score is blind to the hand) [good <= 0.026; bad 0.046-0.267]
    hand_offmask=0.22,         # max: fraction of the fitted hand's visible points off the robot mask (hand drawn where no robot is seen) [good 0.069-0.200; bad 0.246-0.748]
    hand_buried=0.10,          # max: fraction of the fitted hand's in-image points behind NON-robot geometry (hand inside counter / object) [good <= 0.011; no bad example]
    hand_in_view=0.5,          # min: fraction of the fitted hand's points inside the image [good >= 0.807]
    hand_points=50,            # min: observed points within 1 cm of the fitted hand (the hand is actually seen)
    ambiguity_mm=20.0,         # max: farthest near-best candidate hand (score within max(1.5 mm, 10%) of the best), mean hand-row distance [torch fits only; not yet calibrated]
)


def hand_point_mask(rf, sampler):
    """Bool per sampler point: on the hand (the subtree of the hand root body)."""
    hand = np.zeros(rf.m.nbody, bool)
    for b in range(rf.m.nbody):
        a = b
        while a > 0:
            if a == rf.root_body:
                hand[b] = True
                break
            a = rf.m.body_parentid[a]
    return hand[rf.m.geom_bodyid[sampler.owner]]


def goal_checks(rf, r, view, mask, pts, candidates=None, occl=0.015):
    """Truth-free checks of fit result r (needs robot_q, model_to_world, hand) against the goal view, the robot mask and the observed mask points `pts` (world)."""
    # candidates: optional list of (score, robot_q) of the fit's final candidates, for the ambiguity check. Returns (checks dict, list of flag names).
    _, depth, K, pose = view
    T = r["model_to_world"]
    if not hasattr(rf, "dense_hand"):
        rf.dense_hand = hand_point_mask(rf, rf.dense_sampler)
    rf._fk(r["robot_q"])
    P = rf.dense_sampler(rf.d).astype(float) @ T[:3, :3].T + T[:3, 3]
    # observed points not on the fixed robot parts (as in the fit) that the fitted robot does not explain
    rf._fk(r["robot_q"])            # base / torso / left arm are at their start values in robot_q
    fixed = rf.fixed_sampler(rf.d).astype(float) @ T[:3, :3].T + T[:3, 3]
    Y = pts[cKDTree(fixed).query(pts)[0] > FIXED_MARGIN]
    unexplained = float((cKDTree(P).query(Y)[0] > 0.02).mean()) if len(Y) else 1.0
    # the fitted hand in the image
    H = P[rf.dense_hand]
    pc = (H - pose[:3, 3]) @ pose[:3, :3]
    z = pc[:, 2]
    u = np.round(K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(int)
    ins = (z > 0.05) & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    D, on, zi = depth[v[ins], u[ins]], mask[v[ins], u[ins]], z[ins]
    vis = D > zi - occl
    hidden_by_other = (~vis) & (~on)            # behind measured geometry that is not robot (self-occlusion by the robot's own front surface is on-mask)
    checks = dict(
        unexplained=round(unexplained, 3),
        hand_offmask=round(float((~on[vis]).mean()) if vis.any() else 1.0, 3),
        hand_buried=round(float(hidden_by_other.mean()) if len(zi) else 0.0, 3),
        hand_in_view=round(float(ins.mean()), 3),
        hand_points=int((cKDTree(r["hand"]).query(pts)[0] < 0.01).sum()),
    )
    if "score_mm" in r:
        checks["score_mm"] = round(float(r["score_mm"]), 2)
    if candidates:
        best = min(s for s, _ in candidates)
        tol = max(0.0015, 0.1 * best)
        far = 0.0
        for s, q in candidates:
            if s <= best + tol:
                rf._fk(q)
                Tw = T @ rf._root_T()
                h = rf.kin.points(Tw[:3, :3], Tw[:3, 3], q[rf.kin.qpos_adr])
                far = max(far, float(np.linalg.norm(h - r["hand"], axis=1).mean() * 1000))
        checks["ambiguity_mm"] = round(far, 1)
    L = CHECK_LIMITS
    flags = [k for k in ("score_mm", "unexplained", "hand_offmask", "hand_buried", "ambiguity_mm") if k in checks and checks[k] > L[k]]
    flags += [k for k in ("hand_in_view", "hand_points") if checks[k] < L[k]]
    return checks, flags

"""The goal-hand fit as a batched, differentiable multi-start optimisation (torch): the silhouette + visible-depth fit."""
# Why: the cost of RobotFit (median data->model + model_to_view) is lower at the true goal pose than at the returned
# pose on every bad fit, i.e. the failures are search failures. RobotFit's Gauss-Newton optimises data->model only (model_to_view only ranks finished fits),
# and its arm CEM misses the true arm basin. Here both terms are optimised, from ~512 starts in parallel:
#   data->model   Huber distance of each observed robot point to its nearest model point (scale annealed);
#   model->view   each model point projected into the goal camera: hidden behind measured geometry -> 0; on the mask -> |measured depth - model depth| (capped);
#                 off the mask -> its distance to the mask (distance transform, pixels -> metres), capped. The off-mask term is the silhouette gradient.
# Joint limits by parameterisation (q = lo + (hi - lo) * sigmoid(theta)). Starts: the start pose, RobotFit's far-end IK seeds, and random arm configurations
# (each first moved so the model's far end meets the data's far end) x finger postures {start, mid, closed}. The final pick uses RobotFit's own numpy score, so
# selection is the same as before. Inputs are unchanged: the goal RGB-D frame, the SAM robot mask, the robot model, the start proprioception.
from __future__ import annotations

import time

import mujoco
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from lawm.sim import clouds
from lawm.sim.dexwm.bench.goal_hand import FIT, FIXED_MARGIN, VOXEL, RobotFit


def _quat_mat(q):
    """3x3 rotation matrix of a MuJoCo quaternion."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _axis_rot(axis, th):
    """Rotations (B, 3, 3) about a fixed unit axis (3,) by angles th (B,) (Rodrigues)."""
    a = axis / axis.norm()
    Kx = torch.zeros(3, 3, dtype=th.dtype, device=th.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -a[2], a[1], a[2], -a[0], -a[1], a[0]
    s, c = torch.sin(th)[:, None, None], torch.cos(th)[:, None, None]
    return torch.eye(3, dtype=th.dtype, device=th.device) + s * Kx + (1 - c) * (Kx @ Kx)


class TorchFK:
    """Batched forward kinematics of the right arm + hand (the FIT joints) in the model frame. Bodies not moved by FIT (base, torso) sit at their pose for q0
    (the start proprioception), exactly as in RobotFit. Surface points of a SurfaceSampler are carried in their body frames."""

    def __init__(self, rf, q0, device, dtype=torch.float32):
        """Build the chain of the bodies moved by the FIT joints (fixed parents at their pose for q0) and the joint limits lo / hi."""
        m = rf.m
        rf._fk(np.asarray(q0, float))
        d = rf.d
        fit_cols = {int(a): i for i, a in enumerate(FIT)}
        moving = np.zeros(m.nbody, bool)
        for b in range(1, m.nbody):
            a = b
            while a > 0:
                if any(int(m.jnt_qposadr[j]) in fit_cols for j in range(m.body_jntadr[a], m.body_jntadr[a] + m.body_jntnum[a])):
                    moving[b] = True
                    break
                a = m.body_parentid[a]
        self.bodies = np.flatnonzero(moving)                      # MuJoCo orders parents before children
        self.index = {int(b): i for i, b in enumerate(self.bodies)}
        self.dev, self.dt = device, dtype
        T = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        self.links = []
        for b in self.bodies:
            p = int(m.body_parentid[b])
            par = self.index.get(p)
            fixed = None if par is not None else (T(d.xmat[p].reshape(3, 3)), T(d.xpos[p]))
            nj = int(m.body_jntnum[b])
            assert nj <= 1, f"body {b} has {nj} joints"
            joint = None
            if nj == 1:
                j = int(m.body_jntadr[b])
                assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, f"joint {j} is not a hinge"
                joint = (fit_cols[int(m.jnt_qposadr[j])], T(m.jnt_axis[j]), T(m.jnt_pos[j]))
            self.links.append((par, fixed, T(_quat_mat(m.body_quat[b])), T(m.body_pos[b]), joint))
        lim = m.jnt_range[[int(np.flatnonzero(m.jnt_qposadr == a)[0]) for a in FIT]]
        assert np.all(lim[:, 1] > lim[:, 0]), "unlimited fitted joint"
        self.lo, self.hi = T(lim[:, 0]), T(lim[:, 1])

    def bind(self, rf, sampler):
        """(body index (N,), body-frame points (N, 3)) of a SurfaceSampler's points (stored in their geom frames)."""
        m = rf.m
        g = sampler.owner
        Rg = np.stack([_quat_mat(q) for q in m.geom_quat[g]])
        pb = np.einsum("nij,nj->ni", Rg, sampler.local) + m.geom_pos[g]
        bi = np.array([self.index[int(b)] for b in m.geom_bodyid[g]])
        return torch.as_tensor(bi, device=self.dev), torch.as_tensor(pb, dtype=self.dt, device=self.dev)

    def poses(self, q):
        """q (B, len(FIT)) -> body rotations (B, nb, 3, 3), positions (B, nb, 3)."""
        B = q.shape[0]
        Rs, ts = [], []
        for par, fixed, Rl, pos, joint in self.links:
            if par is None:
                Rp, tp = fixed[0].expand(B, 3, 3), fixed[1].expand(B, 3)
            else:
                Rp, tp = Rs[par], ts[par]
            R = Rp @ Rl
            t = tp + Rp @ pos
            if joint is not None:
                col, axis, jp = joint
                Rj = _axis_rot(axis, q[:, col])
                t = t + torch.einsum("bij,bj->bi", R, jp - torch.einsum("bij,j->bi", Rj, jp))   # rotation about the joint anchor jp (body frame)
                R = R @ Rj
            Rs.append(R)
            ts.append(t)
        return torch.stack(Rs, 1), torch.stack(ts, 1)

    def points(self, q, bound):
        """Model points (B, N, 3) of bound (body index, body-frame points) at joint angles q."""
        bi, pb = bound
        R, t = self.poses(q)
        return torch.einsum("bnij,nj->bni", R[:, bi], pb) + t[:, bi]


class Objective:
    """The fit cost of a batch of model point sets (B, N, 3) in the model frame against the observed points and the goal view."""

    def __init__(self, Y, view, mask, T, device, n_obs, rng, occl=0.015, trunc=0.05, off_cap=0.10, dtype=torch.float32):
        """Store the subsampled observed points, the goal view in the model frame (camera, depth, mask, distance transform) and the cost parameters."""
        _, depth, K, pose = view
        Ti = np.linalg.inv(T)
        C = Ti @ pose                                                                  # camera -> model frame
        F = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        sub = rng.choice(len(Y), min(len(Y), n_obs), replace=False)
        self.Y = F(Y[sub])
        self.cR, self.ct = F(C[:3, :3]), F(C[:3, 3])
        self.fx, self.fy, self.cx, self.cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        self.H, self.W = depth.shape
        self.depth = F(depth)
        self.mask = torch.as_tensor(mask, device=device)
        self.dtpx = F(distance_transform_edt(~mask))[None, None]                      # pixels to the nearest mask pixel
        self.occl, self.trunc, self.off_cap = occl, trunc, off_cap

    def __call__(self, P, delta, chunk=128):
        """Per-start cost (B,): mean Huber data->model + mean model->view over the visible model points."""
        with torch.no_grad():                                                          # nearest model point of each observed point
            idx = torch.cat([torch.cdist(self.Y[None].expand(len(p), -1, -1), p).argmin(-1) for p in P.detach().split(chunk)])
        Pn = torch.gather(P, 1, idx[..., None].expand(-1, -1, 3))
        dist = (self.Y[None] - Pn).norm(dim=-1)
        d2m = torch.where(dist < delta, dist ** 2 / (2 * delta), dist - delta / 2).mean(-1)
        pc = (P - self.ct) @ self.cR
        z = pc[..., 2]
        zs = z.clamp(min=1e-3)
        u = self.fx * pc[..., 0] / zs + self.cx
        v = self.fy * pc[..., 1] / zs + self.cy
        ui, vi = u.detach().round().long(), v.detach().round().long()
        ins = (z > 0.05) & (ui >= 0) & (ui < self.W) & (vi >= 0) & (vi < self.H)
        ui, vi = ui.clamp(0, self.W - 1), vi.clamp(0, self.H - 1)
        D, on = self.depth[vi, ui], self.mask[vi, ui]
        vis = ins & (D > z - self.occl)
        grid = torch.stack([2 * u / (self.W - 1) - 1, 2 * v / (self.H - 1) - 1], -1).reshape(1, -1, 1, 2)
        dt = torch.nn.functional.grid_sample(self.dtpx, grid, align_corners=True, padding_mode="border").reshape(P.shape[:2])
        c_on = (D - z).abs().clamp(max=self.trunc)
        c_off = (dt * zs / self.fx).clamp(max=self.off_cap)
        c = torch.where(on, c_on, c_off) * vis
        m2v = c.sum(-1) / vis.sum(-1).clamp(min=1)
        return d2m + m2v


class TorchFit(RobotFit):
    """RobotFit with fit() replaced by the batched silhouette + depth optimisation. Same inputs, same output dict (goal_hand() and the planner are unchanged)."""

    def __init__(self, mjb, meta, device=None, n_starts=512, n_coarse=1500, n_obs_coarse=800, n_obs_fine=3000, steps=(40, 150, 150), keep=32, chunk=128, seed=0):
        """RobotFit plus the dense and coarse surface samplers, the torch device and the optimisation settings."""
        super().__init__(mjb, meta)
        self.dense_sampler = clouds.SurfaceSampler(self.m, self.sampler_geoms, 30000, np.random.default_rng(2))
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_starts, self.n_coarse, self.n_obs_coarse, self.n_obs_fine = n_starts, n_coarse, n_obs_coarse, n_obs_fine
        self.steps, self.keep, self.chunk, self.seed = steps, keep, chunk, seed
        sub = np.random.default_rng(3).choice(len(self.sampler.local), n_coarse, replace=False)
        self.coarse = _Subset(self.sampler, sub)

    def _theta(self, fk, q):
        """Unconstrained parameters theta of joint angles q (inverse of the sigmoid parameterisation)."""
        x = ((q - fk.lo) / (fk.hi - fk.lo)).clamp(1e-3, 1 - 1e-3)
        return torch.log(x / (1 - x))

    def _q(self, fk, th):
        """Joint angles from theta: lo + (hi - lo) * sigmoid(theta)."""
        return fk.lo + (fk.hi - fk.lo) * torch.sigmoid(th)

    def _adam(self, fk, bound, q, loss_fn, steps, lr, arm_only=False):
        """Adam on theta for `steps` steps over chunks of starts (arm joints only if arm_only); returns (final q, final loss per start)."""
        th = self._theta(fk, q).clone().requires_grad_(True)
        opt = torch.optim.Adam([th], lr=lr)
        for it in range(steps):
            opt.zero_grad()
            for c in range(0, len(th), self.chunk):
                sl = slice(c, c + self.chunk)
                P = fk.points(self._q(fk, th[sl]), bound)
                loss_fn(P, it).sum().backward()
            if arm_only:
                th.grad[:, 7:] = 0
            opt.step()
        with torch.no_grad():
            q = self._q(fk, th)
            loss = torch.cat([loss_fn(fk.points(q[c:c + self.chunk], bound), steps - 1) for c in range(0, len(q), self.chunk)])
        return q.detach(), loss

    def fit(self, points, start_q, start_root, refine=True, view=None, mask=None, far_fracs=(0.05, 0.15, 0.3)):
        """Same contract as RobotFit.fit (view and mask are required here)."""
        assert view is not None and mask is not None, "TorchFit needs the goal view and the robot mask"
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        q0 = np.asarray(start_q, float)
        self._fk(q0)
        T = np.asarray(start_root) @ np.linalg.inv(self._root_T())                     # model -> world (as RobotFit.fit)
        Ti = np.linalg.inv(T)
        Y = np.asarray(points, float) @ Ti[:3, :3].T + Ti[:3, 3]
        Y = Y[cKDTree(self.fixed_sampler(self.d)).query(Y)[0] > FIXED_MARGIN]
        if len(Y) < 30:
            return None
        Yv = Y[np.sort(np.unique(np.floor(Y / VOXEL).astype(np.int64), axis=0, return_index=True)[1])]
        fk = TorchFK(self, q0, self.dev)
        coarse, fine = fk.bind(self, self.coarse), fk.bind(self, self.sampler)
        Fq = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)
        lo, hi = fk.lo.cpu().numpy(), fk.hi.cpu().numpy()
        nf = len(FIT)

        # starts: the start pose + far-end IK seeds (RobotFit.seeds) ...
        tm = {}
        tick = lambda k, t=[time.time()]: (tm.__setitem__(k, round(time.time() - t[0], 1)), t.__setitem__(0, time.time()))
        tick("prep")
        det = [q0[FIT]] + [s[FIT] for ff in far_fracs for s in self.seeds(q0, Yv, far_frac=ff)[0]]
        # ... + random arms x finger postures, each first moved so the model's far end meets the data's far end
        n_rand = max(0, self.n_starts - len(det))
        fingers = np.stack([q0[FIT][7:], (lo[7:] + hi[7:]) / 2, lo[7:] + 0.75 * (hi[7:] - lo[7:])])
        R = np.concatenate([rng.uniform(lo[:7], hi[:7], (n_rand, 7)), fingers[np.arange(n_rand) % 3]], 1)
        self._fk(q0)
        base = Fq(self.d.xpos[self.arm_base])
        groups = np.arange(n_rand) % len(far_fracs)
        for gi, ff in enumerate(far_fracs):
            sel = np.flatnonzero(groups == gi)
            if not len(sel):
                continue
            r = np.linalg.norm(Yv - self.d.xpos[self.arm_base], axis=1)
            target = Fq(Yv[r >= np.quantile(r, 1 - ff)].mean(0))
            k = max(1, int(round(ff * self.n_coarse)))

            def far_loss(P, it, target=target, k=k):
                """Squared distance from the target to the centroid of the k model points farthest from the base."""
                rr = (P - base).norm(dim=-1)
                far = torch.topk(rr, k, dim=1).indices
                return ((torch.gather(P, 1, far[..., None].expand(-1, -1, 3)).mean(1) - target) ** 2).sum(-1)
            q_, _ = self._adam(fk, coarse, Fq(R[sel]), far_loss, self.steps[0], 0.1, arm_only=True)
            R[sel] = q_.cpu().numpy()
        tick("seeds+far_end")
        Q = Fq(np.concatenate([np.stack(det), R]))
        src = ["start"] + [f"seed{i}" for i in range(len(det) - 1)] + [f"rand{i}" for i in range(n_rand)]

        # stage 1: all starts, coarse model; stage 2: the best `keep`, full sampler, more observed points
        obj1 = Objective(Yv, view, mask, T, self.dev, self.n_obs_coarse, rng)
        s1 = self.steps[1]
        Q1, L1 = self._adam(fk, coarse, Q, lambda P, it: obj1(P, 0.05 * (0.2) ** (it / max(1, s1 - 1)), self.chunk), s1, 0.05)
        tick("stage1")
        top = torch.argsort(L1)[: self.keep]
        obj2 = Objective(Yv, view, mask, T, self.dev, self.n_obs_fine, rng)
        s2 = self.steps[2]
        Q2, L2 = self._adam(fk, fine, Q1[top], lambda P, it: obj2(P, 0.01 * (0.3) ** (it / max(1, s2 - 1)), self.chunk), s2, 0.02) if refine else (Q1[top], L1[top])

        # final pick: RobotFit's own numpy score (median data->model on the dense model + model_to_view), as before
        tick("stage2")
        best, cands = None, []
        for j, qq in enumerate(Q2.cpu().numpy().astype(float)):
            q = q0.copy()
            q[FIT] = qq
            self._fk(q)
            Pd = self.dense_sampler(self.d).astype(float)
            res = float(np.median(cKDTree(Pd).query(Y)[0]))
            score = res + self.model_to_view(q, T, view, mask)
            cands.append((score, q.copy()))
            if best is None or score < best[0]:
                best = (score, q, res, j)
        score, q, res, j = best
        tick("final_pick")
        self._fk(q)
        Tw = T @ self._root_T()
        hand_q = q[self.kin.qpos_adr]
        hand = self.kin.points(Tw[:3, :3], Tw[:3, 3], hand_q)
        return dict(hand=hand, hand_q=hand_q, hand_root=Tw, robot_q=q, resid_mm=res * 1000, coarse_resid_mm=res * 1000, coarse_hand=hand, n_points=len(Y),
                    seed=int(top[j]), seed_kind=src[int(top[j])], score_mm=score * 1000, model_to_world=T, fit_seconds=time.time() - t0, timing=tm,
                    candidates=cands)                                          # final candidates (score m, robot q), for the ambiguity check


class _Subset:
    """A row subset of a SurfaceSampler (same interface: owner, local)."""

    def __init__(self, s, rows):
        """Keep the owner and local entries of the given rows of sampler s."""
        self.owner, self.local = s.owner[rows], s.local[rows]

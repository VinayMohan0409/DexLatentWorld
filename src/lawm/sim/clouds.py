"""Surface point clouds of MuJoCo bodies, placed with MuJoCo's own world poses (data.geom_xpos, data.geom_xmat)."""
from __future__ import annotations

import mujoco
import numpy as np

SPHERE, CYLINDER, BOX, MESH = (int(getattr(mujoco.mjtGeom, f"mjGEOM_{k}")) for k in ("SPHERE", "CYLINDER", "BOX", "MESH"))
CYLINDER_SEGMENTS = 24


def outward(verts, faces):
    """Faces re-wound so every normal points away from the origin (valid for convex shapes centred on it)."""
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = (normal * tri.mean(axis=1)).sum(axis=1) < 0
    out = faces.copy()
    out[flip] = faces[flip][:, [0, 2, 1]]
    return out


def box_triangles(half):
    """Vertices and 12 outward-wound triangles of a box with half-extents half, centred on the origin."""
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64) * np.asarray(half)
    faces = []
    for axis in range(3):
        for side in (-1, 1):
            on_face = [i for i, c in enumerate(corners) if np.sign(c[axis]) == side]
            a, b, c, d = sorted(on_face, key=lambda i: np.arctan2(*np.delete(corners[i], axis)))
            faces += [[a, b, c], [a, c, d]]
    faces = np.array(faces, dtype=np.int64)
    return corners, outward(corners, faces)


def cylinder_triangles(radius, half_length, segments=CYLINDER_SEGMENTS):
    """Vertices and outward-wound triangles of a z-axis cylinder (side wall and both caps), centred on the origin."""
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)
    top, bottom = np.c_[ring, np.full(segments, half_length)], np.c_[ring, np.full(segments, -half_length)]
    verts = np.concatenate([top, bottom, [[0, 0, half_length]], [[0, 0, -half_length]]])
    top_centre, bottom_centre = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [[i, j, segments + i], [j, segments + j, segments + i], [top_centre, i, j], [bottom_centre, segments + j, segments + i]]
    return verts, outward(verts, np.array(faces, dtype=np.int64))


def sphere_triangles(radius, rings=8, segments=12):
    """Vertices and outward-wound triangles of a latitude-longitude sphere centred on the origin."""
    polar = np.linspace(0, np.pi, rings + 1)[1:-1]
    azimuth = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([np.outer(np.sin(polar), np.cos(azimuth)), np.outer(np.sin(polar), np.sin(azimuth)),
                     np.repeat(np.cos(polar)[:, None], segments, axis=1)], axis=-1).reshape(-1, 3)
    verts = radius * np.concatenate([ring, [[0, 0, 1]], [[0, 0, -1]]])
    north, south = len(ring), len(ring) + 1
    faces = []
    for r in range(rings - 2):
        for i in range(segments):
            a, b, c, d = r * segments + i, r * segments + (i + 1) % segments, (r + 1) * segments + i, (r + 1) * segments + (i + 1) % segments
            faces += [[a, b, c], [b, d, c]]
    for i in range(segments):
        j = (i + 1) % segments
        faces += [[north, i, j], [south, (rings - 2) * segments + j, (rings - 2) * segments + i]]
    return verts, outward(verts, np.array(faces, dtype=np.int64))


def geom_triangles(model, gid):
    """Local-frame vertices and triangles of geom gid: a box or cylinder tessellation, or the model's own mesh."""
    kind = int(model.geom_type[gid])
    if kind == SPHERE:
        return sphere_triangles(model.geom_size[gid][0])
    if kind == BOX:
        return box_triangles(model.geom_size[gid])
    if kind == CYLINDER:
        return cylinder_triangles(model.geom_size[gid][0], model.geom_size[gid][1])
    if kind == MESH:
        mesh = int(model.geom_dataid[gid])
        v0, n_v = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        f0, n_f = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        return model.mesh_vert[v0:v0 + n_v].astype(np.float64), model.mesh_face[f0:f0 + n_f].astype(np.int64)
    raise ValueError(f"geom {gid} has unsupported type {kind} (need sphere, box, cylinder or mesh)")


def sample_triangles(verts, faces, n, rng):
    """n points uniform over the area of a triangle mesh (a face drawn by area, then the sqrt trick of Osada et al.) and the face of each."""
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    face = rng.choice(len(faces), size=n, p=area / area.sum())
    root, u2 = np.sqrt(rng.uniform(size=n)), rng.uniform(size=n)
    weights = np.stack([1 - root, root * (1 - u2), root * u2], axis=1)
    return np.einsum("nk,nkd->nd", weights, tri[face]), face


def body_geom_ids(model, body_id, kind):
    """Geom ids of a body that are collision geoms (contype or conaffinity set) or visual geoms (neither)."""
    assert kind in ("collision", "visual"), kind
    first = int(model.body_geomadr[body_id])
    ids = range(first, first + int(model.body_geomnum[body_id]))
    return [g for g in ids if (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0) == (kind == "collision")]


class SurfaceSampler:
    """Points on a fixed set of geoms drawn once (geom, triangle, barycentric weights), so row i is the same physical point in every frame; calling it places them with the current geom poses."""

    def __init__(self, model, gids, n, rng):
        """Draw n points over the geoms, split by surface area, and store each in its own geom's frame."""
        verts, faces, owner, offset = [], [], [], 0
        for g in gids:
            v, f = geom_triangles(model, g)
            verts.append(v)
            faces.append(f + offset)
            owner.append(np.full(len(f), g))
            offset += len(v)
        self.local, face = sample_triangles(np.concatenate(verts), np.concatenate(faces), n, rng)
        self.owner = np.concatenate(owner)[face].astype(np.int32)

    def __call__(self, data):
        """The points in the world frame for the current state, float32 (n, 3)."""
        rot = data.geom_xmat[self.owner].reshape(-1, 3, 3)
        return (np.einsum("nij,nj->ni", rot, self.local) + data.geom_xpos[self.owner]).astype(np.float32)


def geoms_cloud(model, data, gids, n, rng):
    """n points over the given geoms, split by surface area, placed with MuJoCo's world poses; returns (points (n, 3) float32, geom id of each point)."""
    sampler = SurfaceSampler(model, gids, n, rng)
    return sampler(data), sampler.owner


def body_cloud(model, data, body, n, rng, kind="visual"):
    """n points over all kind geoms of a body, split by surface area; returns (points (n, 3) float32, geom id of each point)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    assert body_id >= 0, f"no body called {body!r}"
    gids = body_geom_ids(model, body_id, kind)
    assert gids, f"body {body!r} has no {kind} geoms"
    return geoms_cloud(model, data, gids, n, rng)


def box_top_cloud(model, data, geom, n, rng):
    """n points uniform on the +z face of a box geom (a table top), in the world frame, float32."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    assert gid >= 0 and int(model.geom_type[gid]) == BOX, f"{geom!r} is not a box geom"
    hx, hy, hz = model.geom_size[gid]
    local = np.stack([rng.uniform(-hx, hx, n), rng.uniform(-hy, hy, n), np.full(n, hz)], axis=1)
    return (local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]).astype(np.float32)

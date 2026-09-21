"""Area samples and exact distances to piecewise triangular surfaces."""

from __future__ import annotations

import numpy as np
from weak_layout_pipeline.pipeline.mesh import Mesh


def triangles(mesh: Mesh) -> np.ndarray:
    faces = mesh.faces
    if faces.shape[1] == 4:
        faces = np.vstack((faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]))
    elif faces.shape[1] != 3:
        raise ValueError("only triangle and quad surfaces are supported")
    return mesh.vertices[faces]


def area_samples(mesh: Mesh, count: int = 2048, seed: int = 1729):
    tri = triangles(mesh)
    area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    if count < 1 or not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("invalid surface or sample count")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(tri), count, p=area / area.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    return np.einsum("ni,nij->nj", bary, tri[ids]), ids, bary


def surface_distances(points: np.ndarray, mesh: Mesh) -> np.ndarray:
    """Exhaustive point/triangle distance in bounded point batches.

    Includes triangle interiors and segment boundaries; no nearest-centroid
    approximation. Quad surfaces use the fixed 0--2 diagonal convention.
    """
    tri = triangles(mesh)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    normal = np.cross(ab, ac)
    nn = dot(normal, normal)
    den = d00 * d11 - d01 * d01
    good = den > np.finfo(float).eps * np.maximum(d00 * d11, np.finfo(float).tiny)
    result = []
    # Keep memory bounded also for large meshes.
    batch = max(1, min(48, 500000 // max(len(tri), 1)))
    for start in range(0, len(points), batch):
        p = points[start : start + batch]
        w = p[:, None, :] - a
        d20 = np.einsum("pfi,fi->pf", w, ab)
        d21 = np.einsum("pfi,fi->pf", w, ac)
        u = (d11 * d20 - d01 * d21) / np.where(good, den, 1)
        v = (d00 * d21 - d01 * d20) / np.where(good, den, 1)
        sq = np.einsum("pfi,fi->pf", w, normal) ** 2 / np.maximum(
            nn, np.finfo(float).tiny
        )
        sq = np.where((u >= 0) & (v >= 0) & (u + v <= 1) & good, sq, np.inf)
        for x, y in ((a, b), (b, c), (c, a)):
            edge = y - x
            delta = p[:, None, :] - x
            t = np.clip(
                np.einsum("pfi,fi->pf", delta, edge)
                / np.maximum(dot(edge, edge), np.finfo(float).tiny),
                0,
                1,
            )
            diff = delta - t[:, :, None] * edge
            sq = np.minimum(sq, np.einsum("pfi,pfi->pf", diff, diff))
        result.extend(np.sqrt(np.maximum(sq.min(axis=1), 0)))
    return np.asarray(result)

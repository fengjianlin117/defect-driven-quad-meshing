"""Small deterministic OBJ and surface-geometry helpers.

The implementation intentionally avoids importing either frozen baseline package.
This keeps the baseline hash invariant meaningful and makes face-order handling
explicit at every pipeline boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


def load_obj(path: str | Path, *, require_triangles: bool = True) -> Mesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            parts = raw.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                ids = [int(token.split("/")[0]) for token in parts[1:]]
                ids = [
                    index - 1 if index > 0 else len(vertices) + index for index in ids
                ]
                if require_triangles and len(ids) != 3:
                    raise ValueError(
                        f"{path}:{number}: expected triangle, got {len(ids)} vertices"
                    )
                faces.append(ids)
    if not vertices or not faces:
        raise ValueError(f"OBJ has no usable vertices/faces: {path}")
    width = len(faces[0])
    if any(len(face) != width for face in faces):
        raise ValueError(f"mixed face sizes are unsupported: {path}")
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if not np.isfinite(v).all():
        raise ValueError(f"OBJ has non-finite vertex coordinates: {path}")
    if width not in (3, 4):
        raise ValueError(f"OBJ requires triangular or quadrilateral faces: {path}")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"OBJ face index out of bounds: {path}")
    return Mesh(v, f)


def write_obj(path: str | Path, mesh: Mesh) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for point in mesh.vertices:
            stream.write(f"v {point[0]:.15g} {point[1]:.15g} {point[2]:.15g}\n")
        for face in mesh.faces:
            stream.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")


def triangle_geometry(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mesh.faces.shape[1] != 3:
        raise ValueError("triangle_geometry requires triangular faces")
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(cross, axis=1)
    if np.any(norms <= 1e-14):
        bad = np.flatnonzero(norms <= 1e-14)[:8].tolist()
        raise ValueError(f"degenerate input triangles: {bad}")
    normals = cross / norms[:, None]
    return normals, 0.5 * norms, tri.mean(axis=1)


def bbox_diagonal(mesh: Mesh) -> float:
    value = float(np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("mesh bounding box has zero or invalid diagonal")
    return value


def canonical_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def edge_topology(
    mesh: Mesh,
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int]]], list[tuple[int, int, tuple[int, int]]]
]:
    """Return edge incidents and face adjacency `(f, g, shared_edge)`."""
    incidents: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_id, face in enumerate(mesh.faces):
        for local in range(len(face)):
            edge = canonical_edge(int(face[local]), int(face[(local + 1) % len(face)]))
            incidents.setdefault(edge, []).append((face_id, local))
    adjacency = []
    for edge, rows in incidents.items():
        if len(rows) == 2:
            adjacency.append((rows[0][0], rows[1][0], edge))
    adjacency.sort()
    return incidents, adjacency


def face_neighbors(mesh: Mesh) -> list[list[int]]:
    neighbors = [[] for _ in range(mesh.face_count)]
    _, adjacency = edge_topology(mesh)
    for left, right, _ in adjacency:
        neighbors[left].append(right)
        neighbors[right].append(left)
    for row in neighbors:
        row.sort()
    return neighbors


def closest_points_on_segments(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vectors = ends - starts
    denom = np.einsum("ij,ij->i", vectors, vectors)
    denom = np.maximum(denom, 1e-30)
    t = np.einsum("ij,ij->i", points - starts, vectors) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = starts + t[:, None] * vectors
    return closest, np.linalg.norm(points - closest, axis=1)


def closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Ericson's closest-point construction for one point and triangle."""
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = point - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = point - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + vb * denom * ab + vc * denom * ac


class SurfaceProjector:
    def __init__(self, triangle_mesh: Mesh, candidates: int = 12):
        self.mesh = triangle_mesh
        self.triangles = triangle_mesh.vertices[triangle_mesh.faces]
        self.centroids = self.triangles.mean(axis=1)
        self.tree = cKDTree(self.centroids)
        self.candidates = min(candidates, triangle_mesh.face_count)
        # A centroid-radius bound makes the candidate query exact: a triangle
        # farther than best_distance + its own radius cannot contain a closer
        # point.  Using the global maximum radius only widens the candidate set.
        self.maximum_vertex_radius = float(
            np.linalg.norm(
                self.triangles - self.centroids[:, None, :],
                axis=2,
            ).max(initial=0.0)
        )

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        output = np.empty_like(points)
        distances = np.empty(len(points), dtype=float)
        for row, point in enumerate(points):
            _, seed = self.tree.query(point, k=1)
            seed_id = int(seed)
            seed_distance = float(np.linalg.norm(point - self.centroids[seed_id]))
            radius = seed_distance + self.maximum_vertex_radius + 1e-12
            candidate_ids = self.tree.query_ball_point(point, radius)
            if not candidate_ids:
                candidate_ids = [seed_id]
            candidate_ids = sorted({seed_id, *(int(value) for value in candidate_ids)})
            candidates = [
                closest_point_on_triangle(point, self.triangles[index])
                for index in candidate_ids
            ]
            values = np.asarray(candidates)
            errors = np.linalg.norm(values - point, axis=1)
            best = int(np.argmin(errors))
            output[row] = values[best]
            distances[row] = errors[best]
        return output, distances


def unique_edges(mesh: Mesh) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for face in mesh.faces:
        for i in range(len(face)):
            edges.add(canonical_edge(int(face[i]), int(face[(i + 1) % len(face)])))
    return np.asarray(sorted(edges), dtype=np.int64)


def polyline_segments(
    vertices: np.ndarray, indices: Iterable[int], closed: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    ids = list(map(int, indices))
    if len(ids) < 2:
        return np.empty((0, 3)), np.empty((0, 3))
    pairs = list(zip(ids[:-1], ids[1:]))
    if closed and ids[0] != ids[-1]:
        pairs.append((ids[-1], ids[0]))
    return (
        np.asarray([vertices[a] for a, _ in pairs], dtype=float),
        np.asarray([vertices[b] for _, b in pairs], dtype=float),
    )

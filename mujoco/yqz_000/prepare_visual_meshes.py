#!/usr/bin/env python3
"""Create MuJoCo-safe visual STL files without touching the source meshes.

MuJoCo accepts at most 200,000 triangles in one STL.  The source YQZ base and
thigh meshes exceed that limit, so this script uses vertex clustering to make
a display-only copy.  Collision geometry continues to come from the URDF.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "resources/robots/YQZ-000/meshes"
DEFAULT_OUTPUT = PROJECT_ROOT / "resources/robots/YQZ-000/meshes_mujoco"


def clustered(mesh, cell_size):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    origin = vertices.min(axis=0)
    keys = np.floor((vertices - origin) / cell_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    reduced_vertices = np.column_stack([
        np.bincount(inverse, weights=vertices[:, axis]) / count
        for axis in range(3)
    ])
    reduced_faces = inverse[faces]
    keep = (
        (reduced_faces[:, 0] != reduced_faces[:, 1])
        & (reduced_faces[:, 1] != reduced_faces[:, 2])
        & (reduced_faces[:, 0] != reduced_faces[:, 2])
    )
    reduced_faces = reduced_faces[keep]
    # Adjacent source triangles can collapse to the same triangle.
    _, unique = np.unique(np.sort(reduced_faces, axis=1), axis=0,
                          return_index=True)
    reduced_faces = reduced_faces[np.sort(unique)]
    result = trimesh.Trimesh(reduced_vertices, reduced_faces, process=False)
    result.remove_unreferenced_vertices()
    return result


def simplify(mesh, target_faces):
    if len(mesh.faces) <= target_faces:
        return mesh
    extent = float(np.max(mesh.extents))
    low, high = extent * 1e-7, extent * 0.2
    best = None
    for _ in range(18):
        middle = (low + high) * 0.5
        candidate = clustered(mesh, middle)
        if len(candidate.faces) <= target_faces:
            best, high = candidate, middle
        else:
            low = middle
    if best is None:
        raise RuntimeError("failed to reduce mesh below target face count")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-faces", type=int, default=120_000)
    args = parser.parse_args()
    if not 1 <= args.target_faces < 200_000:
        parser.error("--target-faces must be between 1 and 199999")
    args.output.mkdir(parents=True, exist_ok=True)
    for source in sorted(args.source.glob("*.STL")):
        mesh = trimesh.load_mesh(source, process=True)
        before = len(mesh.faces)
        visual = simplify(mesh, args.target_faces)
        target = args.output / source.name
        visual.export(target, file_type="stl")
        print(f"[MESH] {source.name}: {before} -> {len(visual.faces)} faces")


if __name__ == "__main__":
    main()

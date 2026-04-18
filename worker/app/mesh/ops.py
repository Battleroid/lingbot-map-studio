from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.mesh.io import latest_revision_path, next_revision, revision_path

log = logging.getLogger(__name__)


def _load_mesh_set(path: Path):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    return ms


def _save_mesh_set(ms, out: Path) -> None:
    # GLB ensures we keep per-vertex color and the scene structure the frontend already loads.
    ms.save_current_mesh(str(out))


def _trimesh_to_pml_mesh(tmesh):
    import pymeshlab

    verts = np.asarray(tmesh.vertices, dtype=np.float64)
    faces = np.asarray(tmesh.faces, dtype=np.int32)
    v_colors = None
    try:
        if getattr(tmesh.visual, "vertex_colors", None) is not None:
            v_colors = np.asarray(tmesh.visual.vertex_colors, dtype=np.float64) / 255.0
    except Exception:
        pass
    if v_colors is not None:
        return pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces, v_color_matrix=v_colors)
    return pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces)


def _load_mesh_from_glb(path: Path):
    """pymeshlab can load .glb in newer builds, but trimesh is more reliable.

    We concatenate geometries into a single mesh; the viewer doesn't depend on
    scene structure after editing.
    """
    import pymeshlab
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    if isinstance(scene, trimesh.Trimesh):
        tmesh = scene
    else:
        geoms = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError(f"No triangle mesh inside {path}")
        tmesh = trimesh.util.concatenate(geoms)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(_trimesh_to_pml_mesh(tmesh), "mesh")
    return ms


def _export_glb(ms, out: Path) -> None:
    """Round-trip through trimesh for GLB (pymeshlab's native writer is PLY/OBJ).

    We export to a temp PLY with colors, reload in trimesh, then write GLB.
    """
    import trimesh

    tmp_ply = out.with_suffix(".tmp.ply")
    ms.save_current_mesh(
        str(tmp_ply),
        save_vertex_color=True,
        save_face_color=False,
        binary=True,
    )
    loaded = trimesh.load(str(tmp_ply), force="mesh")
    tmp_ply.unlink(missing_ok=True)
    scene = trimesh.Scene(loaded)
    scene.export(str(out))


def _run_op(op: str, params: dict[str, Any], ms, face_indices: Optional[list[int]]) -> None:
    import pymeshlab

    if op == "cull":
        if not face_indices:
            raise ValueError("cull requires face_indices")
        mesh = ms.current_mesh()
        n_faces = mesh.face_number()
        keep = np.ones(n_faces, dtype=bool)
        idx = np.asarray(face_indices, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_faces)]
        keep[idx] = False
        verts = mesh.vertex_matrix()
        faces = mesh.face_matrix()[keep]
        new_mesh = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces)
        ms.clear()
        ms.add_mesh(new_mesh, "culled")
    elif op == "fill_holes":
        max_hole_size = int(params.get("max_hole_size", 30))
        ms.apply_filter("meshing_close_holes", maxholesize=max_hole_size)
    elif op == "decimate":
        target_faces = params.get("target_faces")
        ratio = params.get("ratio")
        kw: dict[str, Any] = {"preserveboundary": True, "preservenormal": True}
        if target_faces is not None:
            kw["targetfacenum"] = int(target_faces)
        elif ratio is not None:
            kw["targetperc"] = float(ratio)
        else:
            kw["targetperc"] = 0.5
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", **kw)
    elif op == "smooth":
        iters = int(params.get("iters", 3))
        ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=iters)
    elif op == "remove_small":
        min_diag_perc = float(params.get("min_diag_perc", 5.0))
        ms.apply_filter(
            "compute_selection_by_small_disconnected_components_per_face",
            nbfaceratio=min_diag_perc / 100.0,
        )
        ms.apply_filter("meshing_remove_selected_vertices_and_faces")
    else:
        raise ValueError(f"Unknown op: {op}")


def apply_op(
    artifacts_dir: Path,
    op: str,
    params: dict[str, Any],
    face_indices: Optional[list[int]] = None,
    source_revision: Optional[int] = None,
) -> tuple[Path, int]:
    """Run a mesh op against the latest (or specified) revision, write a new
    revision file, and return (path, revision).
    """
    if source_revision is None:
        source = latest_revision_path(artifacts_dir)
    else:
        source = revision_path(artifacts_dir, source_revision)
    if source is None or not source.exists():
        raise FileNotFoundError("No mesh revision to edit")

    ms = _load_mesh_from_glb(source)
    _run_op(op, params or {}, ms, face_indices)

    rev = next_revision(artifacts_dir)
    out = revision_path(artifacts_dir, rev)
    _export_glb(ms, out)
    return out, rev


def mesh_summary(path: Path) -> dict[str, Any]:
    import trimesh

    m = trimesh.load(str(path), force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        return {"vertices": 0, "faces": 0, "watertight": False}
    return {
        "vertices": int(len(m.vertices)),
        "faces": int(len(m.faces)),
        "watertight": bool(m.is_watertight),
    }

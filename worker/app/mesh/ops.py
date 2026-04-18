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


def _load_point_cloud_ply(path: Path):
    """Load a PLY point cloud into a pymeshlab MeshSet (no face data)."""
    import pymeshlab

    if not path.exists():
        raise FileNotFoundError(f"point cloud missing: {path}")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    return ms


def _surface_reconstruct(ms, params: dict[str, Any]) -> None:
    """Screened-Poisson surface reconstruction on the current point cloud.

    Expects `ms` to have the point cloud loaded as its current mesh. Computes
    per-point normals, runs Poisson, then leaves the triangle-mesh result as
    the current mesh (the exporter saves current_mesh only, so the source
    point cloud simply stays in the set unused and gets GC'd).
    """
    depth = int(params.get("depth", 8))
    samples_per_node = float(params.get("samples_per_node", 1.5))
    point_weight = float(params.get("point_weight", 4.0))
    normal_k = int(params.get("normal_k", 10))

    # Normals first — Poisson requires them and raw point clouds usually
    # don't carry normals.
    try:
        ms.compute_normal_for_point_clouds(
            k=normal_k, smoothiter=0, flipflag=False, viewpos=[0.0, 0.0, 0.0]
        )
    except Exception:
        ms.apply_filter(
            "compute_normal_for_point_clouds",
            k=normal_k,
            smoothiter=0,
            flipflag=False,
            viewpos=[0.0, 0.0, 0.0],
        )

    try:
        ms.generate_surface_reconstruction_screened_poisson(
            visiblelayer=False,
            depth=depth,
            fulldepth=max(1, depth - 3),
            cgdepth=0,
            scale=1.1,
            samplespernode=samples_per_node,
            pointweight=point_weight,
            iters=8,
            confidence=False,
            preclean=False,
        )
    except Exception:
        ms.apply_filter(
            "generate_surface_reconstruction_screened_poisson",
            visiblelayer=False,
            depth=depth,
            fulldepth=max(1, depth - 3),
            cgdepth=0,
            scale=1.1,
            samplespernode=samples_per_node,
            pointweight=point_weight,
            iters=8,
            confidence=False,
            preclean=False,
        )

    # The reconstruction filter appended a new mesh to the set. Find the
    # one that actually has faces (there may be several meshes now — the
    # original cloud with 0 faces plus the Poisson output) and make that
    # the current mesh so the exporter picks it up.
    best_id = -1
    best_faces = 0
    # mesh_number() gives the count of LIVE meshes; iterate by mesh_id
    # which can be sparse after deletions (safe for a fresh MeshSet too).
    try:
        ids = list(ms.mesh_id_list())
    except Exception:
        ids = list(range(ms.mesh_number()))
    for mid in ids:
        try:
            ms.set_current_mesh(mid)
            f = ms.current_mesh().face_number()
            if f > best_faces:
                best_faces = f
                best_id = mid
        except Exception:
            continue
    if best_id < 0:
        raise RuntimeError(
            "Poisson reconstruction produced no triangle mesh — "
            "try a lower depth, or more points may be needed"
        )
    ms.set_current_mesh(best_id)


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
    elif op == "surface_recon":
        _surface_reconstruct(ms, params)
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

    `surface_recon` is special-cased: it reconstructs from the raw point
    cloud (reconstruction.ply) instead of a prior revision, since the GLB
    scenes from predictions_to_glb only contain camera frustum triangles,
    not the scene points.
    """
    if op == "surface_recon":
        ply_path = artifacts_dir / "reconstruction.ply"
        if not ply_path.exists():
            raise FileNotFoundError(
                "surface_recon needs reconstruction.ply — run an export first"
            )
        ms = _load_point_cloud_ply(ply_path)
        _run_op(op, params or {}, ms, face_indices)
    else:
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

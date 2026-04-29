"""Pin the world-points-synthesis fallback used by export.py.

Some lingbot-map model variants emit `depth` but skip `world_points`
in their inference output. The upstream `predictions_to_glb` then logs
`world_points not found, falling back to depth-based points` and
KeyError's on `world_points_from_depth` (its fallback branch wants
that derived field, not raw depth).

`_ensure_world_points_for_export` synthesises the missing field from
depth + intrinsic + extrinsic. Pin the four branches:
  * world_points already present → no-op.
  * world_points_from_depth already present → no-op.
  * neither key + depth + intrinsic + extrinsic available → synthesised.
  * neither key + depth missing → no-op (gracefully skips).
"""

from __future__ import annotations

import numpy as np


def test_noop_when_world_points_present():
    from app.pipeline.export import _ensure_world_points_for_export

    pred = {"world_points": np.zeros((2, 4, 4, 3), dtype=np.float32)}
    _ensure_world_points_for_export(pred)
    assert "world_points_from_depth" not in pred


def test_noop_when_depth_already_synthesised():
    from app.pipeline.export import _ensure_world_points_for_export

    pred = {"world_points_from_depth": np.zeros((1, 1, 1, 3), dtype=np.float32)}
    sentinel = pred["world_points_from_depth"]
    _ensure_world_points_for_export(pred)
    # Untouched — same array object.
    assert pred["world_points_from_depth"] is sentinel


def test_synthesises_from_depth_when_missing():
    from app.pipeline.export import _ensure_world_points_for_export

    S, H, W = 2, 4, 4
    pred = {
        "depth": np.full((S, H, W), 2.0, dtype=np.float32),
        "intrinsic": np.tile(
            np.array(
                [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            (S, 1, 1),
        ),
        # Identity c2w — world coords match camera coords.
        "extrinsic": np.tile(
            np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            (S, 1, 1),
        ),
    }
    _ensure_world_points_for_export(pred)
    out = pred["world_points_from_depth"]
    assert out.shape == (S, H, W, 3)
    # z (forward) of every pixel = depth = 2.0.
    np.testing.assert_allclose(out[..., 2], 2.0)
    # x at the center pixel (cx=2, cy=2): (2 - 2) * 2 / 10 = 0.
    assert abs(out[0, 2, 2, 0]) < 1e-5
    assert abs(out[0, 2, 2, 1]) < 1e-5


def test_skips_when_depth_missing():
    from app.pipeline.export import _ensure_world_points_for_export

    pred = {
        "intrinsic": np.zeros((1, 3, 3), dtype=np.float32),
        "extrinsic": np.zeros((1, 3, 4), dtype=np.float32),
    }
    _ensure_world_points_for_export(pred)
    assert "world_points_from_depth" not in pred


def test_skips_on_shape_mismatch():
    from app.pipeline.export import _ensure_world_points_for_export

    # Depth says S=2 frames, intrinsics says S=3 — mismatch should skip
    # rather than crash the export pipeline.
    pred = {
        "depth": np.zeros((2, 4, 4), dtype=np.float32),
        "intrinsic": np.zeros((3, 3, 3), dtype=np.float32),
        "extrinsic": np.zeros((2, 3, 4), dtype=np.float32),
    }
    _ensure_world_points_for_export(pred)
    assert "world_points_from_depth" not in pred

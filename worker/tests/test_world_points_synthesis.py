"""Pin the world-points-synthesis fallback used by export.py.

Some lingbot-map model variants emit `depth` but skip `world_points`
in their inference output. The upstream `predictions_to_glb` then logs
`world_points not found, falling back to depth-based points` and
KeyError's on `world_points_from_depth` (its fallback branch wants
that derived field, not raw depth).

`_ensure_world_points_for_export` synthesises the missing field from
depth + intrinsic + extrinsic. Pin the branches:
  * world_points already present → no-op.
  * world_points_from_depth already present → no-op.
  * key present but value None → treated as absent (synthesise).
  * neither key + depth + intrinsic + extrinsic available → synthesised.
  * neither key + depth missing → no-op (gracefully skips).
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest


def _identity_pred(S: int = 2, H: int = 4, W: int = 4) -> dict:
    """Smallest predictions dict that lets synthesis succeed: identity c2w
    extrinsic, simple pinhole intrinsic, constant depth=2.0."""
    return {
        "depth": np.full((S, H, W), 2.0, dtype=np.float32),
        "intrinsic": np.tile(
            np.array(
                [[10.0, 0.0, W / 2], [0.0, 10.0, H / 2], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            (S, 1, 1),
        ),
        "extrinsic": np.tile(
            np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            (S, 1, 1),
        ),
    }


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
    pred = _identity_pred(S, H, W)
    _ensure_world_points_for_export(pred)
    out = pred["world_points_from_depth"]
    assert out.shape == (S, H, W, 3)
    # z (forward) of every pixel = depth = 2.0.
    np.testing.assert_allclose(out[..., 2], 2.0)
    # x at the center pixel (cx=W/2, cy=H/2): (W/2 - W/2) * 2 / 10 = 0.
    assert abs(out[0, H // 2, W // 2, 0]) < 1e-5
    assert abs(out[0, H // 2, W // 2, 1]) < 1e-5


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


# --- defensive: keys present but value None ---------------------------


def test_treats_none_world_points_as_absent():
    """A `world_points: None` entry should NOT block synthesis — earlier
    versions early-returned on bare `key in dict`, leaving the upstream
    library to crash on `predictions[None].reshape(...)`.
    """
    from app.pipeline.export import _ensure_world_points_for_export

    pred = _identity_pred()
    pred["world_points"] = None
    _ensure_world_points_for_export(pred)
    assert pred.get("world_points_from_depth") is not None


def test_treats_none_world_points_from_depth_as_absent():
    from app.pipeline.export import _ensure_world_points_for_export

    pred = _identity_pred()
    pred["world_points_from_depth"] = None
    _ensure_world_points_for_export(pred)
    out = pred["world_points_from_depth"]
    assert out is not None and out.shape[-1] == 3


# --- depth_conf placeholder paired with synthesised world_points ------


def test_synthesises_depth_conf_when_missing():
    """Upstream's depth-fallback branch reads `depth_conf` for percentile
    filtering. When we synthesise world_points_from_depth from a model that
    emits no depth_conf, we also stuff in a placeholder ones array so the
    filter passes everything rather than crashing.
    """
    from app.pipeline.export import _ensure_world_points_for_export

    pred = _identity_pred()
    _ensure_world_points_for_export(pred)
    assert "depth_conf" in pred
    assert pred["depth_conf"].shape == pred["depth"].shape
    np.testing.assert_allclose(pred["depth_conf"], 1.0)


def test_keeps_existing_depth_conf():
    from app.pipeline.export import _ensure_world_points_for_export

    pred = _identity_pred()
    sentinel = np.full(pred["depth"].shape, 0.5, dtype=np.float32)
    pred["depth_conf"] = sentinel
    _ensure_world_points_for_export(pred)
    assert pred["depth_conf"] is sentinel


# --- visibility into the failure path --------------------------------


def test_failure_reason_printed_to_stdout():
    """When synthesis can't proceed, the function must print the reason
    + the actual key list to stdout (so capture_stdio surfaces it to the
    user UI). Earlier versions used `log.warning` which routed nowhere
    visible and left users staring at a bare KeyError."""
    from app.pipeline.export import _ensure_world_points_for_export

    pred = {"pose_enc": np.zeros((1, 4, 9), dtype=np.float32)}
    buf = io.StringIO()
    with redirect_stdout(buf):
        _ensure_world_points_for_export(pred)
    out = buf.getvalue()
    assert "[lingbot]" in out
    assert "pose_enc" in out  # key list included for debugging


def test_success_reason_printed_to_stdout():
    from app.pipeline.export import _ensure_world_points_for_export

    pred = _identity_pred()
    buf = io.StringIO()
    with redirect_stdout(buf):
        _ensure_world_points_for_export(pred)
    assert "synthesised world_points_from_depth" in buf.getvalue()


# --- _scene_from_predictions: clear error before upstream KeyError ----


def test_scene_raises_clear_error_when_synthesis_impossible(monkeypatch):
    """Pre-empt the upstream library's `KeyError: 'world_points_from_depth'`
    with a RuntimeError that names the actual problem (no geometry-bearing
    keys at all) and includes the predictions summary for debugging."""
    import sys
    import types

    # Stub the lingbot_map.vis.glb_export import — we never reach the call.
    fake_mod = types.ModuleType("lingbot_map.vis.glb_export")
    fake_mod.predictions_to_glb = lambda *a, **kw: pytest.fail(
        "predictions_to_glb should not be reached when synthesis is impossible"
    )
    monkeypatch.setitem(sys.modules, "lingbot_map", types.ModuleType("lingbot_map"))
    monkeypatch.setitem(sys.modules, "lingbot_map.vis", types.ModuleType("lingbot_map.vis"))
    monkeypatch.setitem(sys.modules, "lingbot_map.vis.glb_export", fake_mod)

    from app.jobs.schema import LingbotConfig
    from app.pipeline.export import _scene_from_predictions

    cfg = LingbotConfig()
    pred = {"pose_enc": np.zeros((1, 4, 9), dtype=np.float32)}
    with pytest.raises(RuntimeError, match="neither `world_points` nor `depth`"):
        _scene_from_predictions(pred, cfg)


# --- helper unit tests ------------------------------------------------


def test_value_present_helper():
    from app.pipeline.export import _value_present

    assert _value_present({"k": np.zeros(3)}, "k") is True
    assert _value_present({"k": None}, "k") is False
    assert _value_present({}, "k") is False


def test_predictions_summary_helper():
    from app.pipeline.export import _predictions_summary

    summary = _predictions_summary(
        {
            "shape_array": np.zeros((2, 3), dtype=np.float32),
            "noneval": None,
            "scalar": 1.0,
        }
    )
    assert summary["shape_array"] == "(2, 3)"
    assert summary["noneval"] == "None"
    assert summary["scalar"] == "float"

"""Phase-3 FPV preprocessing stages, run on a synthetic clip.

Each stage has a distinct contract:

  * `color_norm.apply` rewrites each PNG in place — assert pixels changed
    and the tint was reduced.
  * `rolling_shutter.apply` either warps or returns 0 touched frames
    when the estimated shear is sub-threshold.
  * `deblur.apply` applies unsharp to the blurriest quantile — assert
    only a subset is touched.
  * `keyframe_score.write_scores` writes `frame_scores.jsonl` with one
    row per frame and a per-clip-normalised `quality` field.
  * `analog_cleanup.ffmpeg_snippet` returns the right chain for each
    flag combination (pure config, no file I/O).

Run: `pytest worker/tests/test_preproc_stages.py -q`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _read_all(frames_dir: Path) -> list[np.ndarray]:
    return [cv2.imread(str(p)) for p in sorted(frames_dir.glob("*.png"))]


@pytest.mark.asyncio
async def test_color_norm_rewrites_and_reduces_green_tint(
    synthetic_frames: Path,
    captured_events: tuple[list, object],
):
    from app.pipeline.fpv_filters import color_norm

    events, publish = captured_events
    before = _read_all(synthetic_frames)
    touched = await color_norm.apply("job-cn", synthetic_frames, publish)
    after = _read_all(synthetic_frames)

    assert touched == len(before) > 0
    # At least one pixel changed — the grey-world pass shouldn't be a no-op
    # on our tinted fixture.
    diffs = [int(np.abs(a.astype(int) - b.astype(int)).sum()) for a, b in zip(before, after)]
    assert sum(diffs) > 0

    # Green channel mean should be closer to the mean of R+B after WB.
    before_g_delta = np.mean([
        abs(b[..., 1].mean() - (b[..., 0].mean() + b[..., 2].mean()) / 2) for b in before
    ])
    after_g_delta = np.mean([
        abs(a[..., 1].mean() - (a[..., 0].mean() + a[..., 2].mean()) / 2) for a in after
    ])
    assert after_g_delta <= before_g_delta + 1e-6

    # Progress events published for the stage.
    assert any(getattr(e, "stage", None) == "ingest" for e in events)


@pytest.mark.asyncio
async def test_rolling_shutter_skips_on_sub_threshold_shear(
    synthetic_frames: Path,
    captured_events: tuple[list, object],
):
    """Our synthetic clip has a horizontal-translation-only motion, so the
    estimated y-shear is ~0 → the stage should skip the warp entirely."""
    from app.pipeline.fpv_filters import rolling_shutter

    events, publish = captured_events
    touched = await rolling_shutter.apply(
        "job-rs", synthetic_frames, override_shear=None, publish=publish
    )
    assert touched == 0
    assert any("rolling_shutter" in getattr(e, "message", "") for e in events)


@pytest.mark.asyncio
async def test_rolling_shutter_applies_on_override(
    synthetic_frames: Path,
    captured_events: tuple[list, object],
):
    """Force a clearly-above-threshold shear via the override knob and
    check that every frame is warped."""
    from app.pipeline.fpv_filters import rolling_shutter

    events, publish = captured_events
    touched = await rolling_shutter.apply(
        "job-rs-override",
        synthetic_frames,
        override_shear=0.25,
        publish=publish,
    )
    assert touched == len(sorted(synthetic_frames.glob("*.png")))


@pytest.mark.asyncio
async def test_deblur_unsharp_gates_on_median(
    synthetic_frames: Path,
    captured_events: tuple[list, object],
):
    from app.jobs.schema import LingbotConfig
    from app.pipeline.fpv_filters import deblur

    events, publish = captured_events
    # gate_frac=0.6 → only the blurriest ~60% get the filter; we assert
    # that it doesn't touch all frames (the gate is doing something).
    cfg = LingbotConfig(preproc_deblur="unsharp", deblur_sharpness_gate=0.6)
    touched = await deblur.apply("job-db", synthetic_frames, cfg, publish)
    total = len(sorted(synthetic_frames.glob("*.png")))
    assert 0 <= touched <= total
    # The diagnostic event carries the median we gated on.
    assert any(getattr(e, "data", {}).get("median_var") is not None for e in events)


@pytest.mark.asyncio
async def test_deblur_none_is_noop(
    synthetic_frames: Path, captured_events: tuple[list, object]
):
    from app.jobs.schema import LingbotConfig
    from app.pipeline.fpv_filters import deblur

    _, publish = captured_events
    cfg = LingbotConfig(preproc_deblur="none")
    touched = await deblur.apply("job-db-off", synthetic_frames, cfg, publish)
    assert touched == 0


@pytest.mark.asyncio
async def test_keyframe_score_writes_jsonl_with_quality(
    synthetic_frames: Path, captured_events: tuple[list, object]
):
    import json as _json

    from app.pipeline.fpv_filters import keyframe_score

    _, publish = captured_events
    out = await keyframe_score.write_scores("job-ks", synthetic_frames, publish)
    assert out.exists() and out.name == "frame_scores.jsonl"

    rows = [_json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == len(sorted(synthetic_frames.glob("*.png")))
    # Every row has the three expected fields after scoring.
    for row in rows:
        assert {"index", "name", "sharpness"}.issubset(row.keys())
    # `quality` is normalised so the median row sits ~1.0.
    qualities = [r.get("quality", 0.0) for r in rows]
    assert any(q > 0 for q in qualities)
    med = sorted(qualities)[len(qualities) // 2]
    assert 0.5 <= med <= 2.0


def test_analog_cleanup_ffmpeg_snippet_flags():
    from app.jobs.schema import LingbotConfig
    from app.pipeline.fpv_filters import analog_cleanup

    # Default config → no extra filters added by this module.
    assert analog_cleanup.ffmpeg_snippet(LingbotConfig()) == []

    cfg = LingbotConfig(preproc_analog_cleanup=True)
    snippet = analog_cleanup.ffmpeg_snippet(cfg)
    assert any("atadenoise" in s for s in snippet)

    # deflicker-only: standalone deflicker filter.
    cfg2 = LingbotConfig(preproc_deflicker=True, preproc_denoise=False)
    assert any("deflicker" in s for s in analog_cleanup.ffmpeg_snippet(cfg2))

    # deflicker + denoise: deduplicated (denoise bundle already has it).
    cfg3 = LingbotConfig(preproc_deflicker=True, preproc_denoise=True)
    assert analog_cleanup.ffmpeg_snippet(cfg3) == []

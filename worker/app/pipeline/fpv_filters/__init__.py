"""Analog-FPV preprocessing stages.

Each module implements one stage from the Phase 3 pipeline. Stages are
applied in a fixed order by `pipeline.ingest`:

  1. analog_cleanup.ffmpeg_snippet  — extra ffmpeg filters (atadenoise, etc).
  2. (ffmpeg extraction writes PNGs.)
  3. color_norm.apply               — grey-world white balance.
  4. rolling_shutter.apply          — global skew correction.
  5. deblur.apply                   — unsharp mask / NAFNet (config-selected).
  6. keyframe_score.write_scores    — sharpness + motion scoring to JSONL.

All `apply(...)` helpers are idempotent (running twice produces the same
output) so retries / reexports skip work via the caller's cache. Each
stage reads the subset of fields it cares about from the config and
returns the number of frames it touched for progress reporting.
"""

from app.pipeline.fpv_filters import (
    analog_cleanup,
    color_norm,
    deblur,
    keyframe_score,
    rolling_shutter,
)

__all__ = [
    "analog_cleanup",
    "color_norm",
    "deblur",
    "keyframe_score",
    "rolling_shutter",
]

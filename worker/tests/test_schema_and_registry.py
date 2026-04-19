"""Phase-1 safety net: discriminated-union parsing + worker-class routing.

These tests lock in the invariants that the API + worker claim loop rely
on:

  * Legacy job rows (created before Phase 1, no `processor` field) parse
    back as `LingbotConfig`.
  * Each `ProcessorId` routes to its expected worker class.
  * Each SLAM backend's config is keyed by a unique discriminator value.
  * `processor_kind` groups the six concrete processors into the three
    viewer-facing kinds the frontend switches on.
"""

from __future__ import annotations

import pytest


def test_legacy_row_parses_as_lingbot():
    from app.jobs.schema import LingbotConfig, parse_job_config

    # Representative old-row blob — no `processor` key, just the lingbot
    # fields that existed pre-Phase-1.
    legacy = {"model_id": "lingbot-map", "fps": 10.0, "image_size": 518}
    cfg = parse_job_config(legacy)
    assert isinstance(cfg, LingbotConfig)
    assert cfg.processor == "lingbot"
    assert cfg.fps == 10.0


def test_every_processor_has_a_worker_class():
    from app.processors import WORKER_CLASSES

    expected_ids = {
        "lingbot",
        "droid_slam",
        "mast3r_slam",
        "dpvo",
        "monogs",
        "gsplat",
    }
    assert set(WORKER_CLASSES.keys()) == expected_ids
    # Lingbot → its own container; all four SLAM backends → slam; gsplat → gs.
    assert WORKER_CLASSES["lingbot"] == "lingbot"
    for sid in ("droid_slam", "mast3r_slam", "dpvo", "monogs"):
        assert WORKER_CLASSES[sid] == "slam"
    assert WORKER_CLASSES["gsplat"] == "gs"


def test_processor_kind_grouping():
    from app.jobs.schema import (
        DpvoConfig,
        DroidSlamConfig,
        GsplatConfig,
        LingbotConfig,
        Mast3rSlamConfig,
        MonogsConfig,
        processor_kind,
    )

    assert processor_kind(LingbotConfig()) == "reconstruction"
    assert processor_kind(DroidSlamConfig()) == "slam"
    assert processor_kind(Mast3rSlamConfig()) == "slam"
    assert processor_kind(DpvoConfig()) == "slam"
    assert processor_kind(MonogsConfig()) == "slam"
    assert processor_kind(GsplatConfig(source_job_id="abc")) == "gsplat"


def test_discriminator_roundtrip_preserves_backend():
    from app.jobs.schema import (
        DroidSlamConfig,
        MonogsConfig,
        dump_job_config,
        parse_job_config,
    )

    cfg = DroidSlamConfig(buffer_size=256, global_ba_iters=10)
    raw = dump_job_config(cfg)
    rebuilt = parse_job_config(raw)
    assert isinstance(rebuilt, DroidSlamConfig)
    assert rebuilt.buffer_size == 256
    assert rebuilt.global_ba_iters == 10

    mono = MonogsConfig(refine_iters=25, prune_opacity=0.01)
    rebuilt_mono = parse_job_config(dump_job_config(mono))
    assert isinstance(rebuilt_mono, MonogsConfig)
    assert rebuilt_mono.refine_iters == 25


def test_gsplat_config_requires_source_job():
    """The API bakes this in before enqueueing; the schema-level check is
    the backstop."""
    from pydantic import ValidationError

    from app.jobs.schema import GsplatConfig

    with pytest.raises(ValidationError):
        GsplatConfig()  # type: ignore[call-arg]


def test_load_processor_returns_expected_classes():
    from app.processors import load_processor

    # We intentionally only exercise the processors whose modules are
    # import-clean on CPU (no torch / CUDA extensions). That covers
    # lingbot's registration (the concrete module pulls torch lazily) and
    # all four SLAM backends' simulated-tracker fallback, plus gsplat.
    for pid in ("mast3r_slam", "droid_slam", "dpvo", "monogs", "gsplat"):
        cls = load_processor(pid)
        assert cls.id == pid

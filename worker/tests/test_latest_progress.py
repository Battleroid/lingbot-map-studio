"""Smoke tests for `events.latest_progress` — the tail-reader the
JobList table relies on for per-row progress without a per-row WS
subscription. Should be cheap, robust to corruption, and correct."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.jobs import events


def _write_events(job_id: str, payloads: list[dict]) -> Path:
    """Materialize a fake events.jsonl for `job_id` and return the path."""
    job_dir = settings.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "events.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for p in payloads:
            fh.write(json.dumps(p) + "\n")
    return path


def test_returns_none_when_events_file_missing(tmp_data_dir):
    assert events.latest_progress("never-existed") is None


def test_returns_none_when_no_event_carries_progress(tmp_data_dir):
    _write_events(
        "no-progress",
        [
            {"id": 1, "job_id": "no-progress", "stage": "queue", "message": "claimed"},
            {"id": 2, "job_id": "no-progress", "stage": "ingest", "message": "starting"},
        ],
    )
    assert events.latest_progress("no-progress") is None


def test_returns_most_recent_progress_value(tmp_data_dir):
    _write_events(
        "happy",
        [
            {"id": 1, "job_id": "happy", "stage": "ingest", "progress": 0.1},
            {"id": 2, "job_id": "happy", "stage": "inference", "progress": 0.42},
            {"id": 3, "job_id": "happy", "stage": "inference", "progress": 0.93},
        ],
    )
    assert events.latest_progress("happy") == 0.93


def test_skips_events_without_progress_when_walking_back(tmp_data_dir):
    """A trailing event with no `progress` field shouldn't shadow the
    real most-recent progress value emitted earlier."""
    _write_events(
        "interleaved",
        [
            {"id": 1, "job_id": "interleaved", "stage": "ingest", "progress": 0.2},
            {"id": 2, "job_id": "interleaved", "stage": "inference", "progress": 0.7},
            {"id": 3, "job_id": "interleaved", "stage": "inference", "message": "fps tick"},
            {"id": 4, "job_id": "interleaved", "stage": "inference", "message": "vram tick"},
        ],
    )
    assert events.latest_progress("interleaved") == 0.7


def test_handles_corrupt_lines_without_crashing(tmp_data_dir):
    job_dir = settings.job_dir("corrupt")
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "events.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": 1, "job_id": "corrupt", "stage": "ingest", "progress": 0.5}) + "\n")
        fh.write("{ this is not valid json\n")
        fh.write(json.dumps({"id": 2, "job_id": "corrupt", "stage": "inference", "progress": 0.8}) + "\n")
    assert events.latest_progress("corrupt") == 0.8


def test_only_reads_tail_for_long_files(tmp_data_dir):
    """Even a multi-MB events.jsonl should resolve in ms because the
    helper only reads the trailing window. Sanity-check by stuffing a
    large prefix of progress=0.0 events and asserting we still see the
    final 0.99 from the tail."""
    payloads = [
        {"id": i, "job_id": "long", "stage": "inference", "progress": 0.0}
        for i in range(2000)
    ]
    payloads.append(
        {"id": 9999, "job_id": "long", "stage": "inference", "progress": 0.99},
    )
    _write_events("long", payloads)
    assert events.latest_progress("long") == 0.99


def test_treats_partial_first_line_as_skipped(tmp_data_dir):
    """The 16 KB tail almost always slices mid-line at the start. The
    helper drops that first (partial) line so we don't try to parse half
    an event. Construct an events.jsonl that exercises the boundary."""
    job_dir = settings.job_dir("boundary")
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "events.jsonl"
    big = json.dumps({"id": 1, "job_id": "boundary", "stage": "ingest", "progress": 0.05})
    padding = "x" * 20000  # forces the file beyond the 16 KB tail window
    with path.open("w", encoding="utf-8") as fh:
        fh.write(big + " " + padding + "\n")
        fh.write(json.dumps({"id": 2, "job_id": "boundary", "stage": "inference", "progress": 0.6}) + "\n")
    # The first line is skipped (it's mid-line in the tail window), and
    # we successfully parse the second.
    assert events.latest_progress("boundary") == 0.6

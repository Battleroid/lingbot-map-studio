# FPV preprocessing pipeline

Low-bitrate analog FPV footage — what actually comes out of a DVR recording
an analog receiver — fails reconstruction in very specific ways: chroma
noise becomes false geometry, OSD telemetry text fixes itself in camera
space, rolling-shutter skew warps the scale, and motion blur starves
keyframe selection. The preprocessing pipeline is a set of toggleable
stages tuned for that failure mode.

Each stage is idempotent per-frame and writes into a content-addressed
cache under `data/preproc_cache/{hash}/`, so retries and reexports skip
work.

## Stages

### 1. Analog noise cleanup

- `preproc_denoise` → `hqdn3d` (temporal + spatial denoise, cheap).
- `preproc_analog_cleanup` → `atadenoise` (heavier, tuned for chroma-noise
  / dot crawl / VHS fringing — **expensive**, only enable for clips where
  hqdn3d leaves artefacts).

No-op on clean digital clips.

### 2. Deflicker

- `preproc_deflicker` → ffmpeg `deflicker` filter + median-luma
  normalization across frames.

Useful when brightness jitter is the dominant artefact. Auto-skipped when
`preproc_denoise + deflicker` is already bundled.

### 3. Color normalization

- `preproc_color_norm` → per-frame grey-world white-balance + 1/99
  percentile histogram stretch.

Recovers natural color on green/magenta-tinted analog feeds. Cheap, CPU-only.

### 4. Rolling-shutter correction

- `preproc_rs_correction` → estimate global y-shear from optical flow
  between rows, apply the inverse affine warp.
- `rs_shear_px_per_row` → override the estimate (blank = auto). Values
  smaller than ±0.02 px/row skip the warp entirely.

v1 handles the dominant-skew case only. Full per-row RS needs gyro data
and is out of scope.

### 5. Motion deblur

- `preproc_deblur: "none" | "unsharp" | "nafnet"`.
- `deblur_sharpness_gate` → apply the deblur filter only to frames whose
  Laplacian variance is below this fraction of the clip median. 1.0 =
  every frame, 0.6 (default) = the blurriest ~60%.

`nafnet` is a learned single-image deblur; the checkpoint lazy-downloads
via the existing cache, and falls back to `unsharp` if absent.

### 6. Fisheye unwrap

- `preproc_fisheye` → `v360` filter.
- `fisheye_in_fov` / `fisheye_out_fov` → source lens HFOV / target
  diagonal FOV after unwrap.

FPV micro cams (2.1-2.5 mm) are 150-170° — leaving them distorted feeds
the reconstructor false geometry.

Reference values:
- Caddx Ratel 2.1 → 165°
- RunCam Phoenix 2 → 155°
- Foxeer Predator Micro → 160°

### 7. OSD / HUD mask + inpaint

- `preproc_osd_mask` → detect static pixels (telemetry text, timer,
  battery, home arrow) and inpaint them out.
- `osd_mask_samples` — how many frames to sample.
- `osd_mask_std_threshold` — per-pixel stddev cutoff (0-255 scale).
- `osd_mask_dilate` — grow the mask to catch anti-aliased text edges.
- `osd_detect_text` — flag regions near edges in most sampled frames
  (catches changing numeric HUD values).
- `osd_edge_persist_frac` — stricter/looser edge-persistence gate.

The resulting mask is also exposed to SLAM processors as a per-frame
ignore mask, so their feature matcher doesn't anchor on baked-in text.

### 8. Keyframe scoring

- `preproc_keyframe_score` → compute per-frame sharpness (Laplacian
  variance) + motion magnitude (optical-flow L2).
- `keyframe_min_sharpness_frac` / `keyframe_min_motion_px` → ingest-side
  gates.

Writes `frame_scores.jsonl`; SLAM backends with
`keyframe_policy=score_gated` read this instead of re-scoring. Cheap to
leave on — backends that don't consume the file just ignore it.

## Presets

| Preset | Stages on |
| --- | --- |
| `none` | (nothing) |
| `analog fpv (default)` | denoise + deflicker + osd_mask + color_norm + rs_correction + unsharp deblur + keyframe_score |
| `aggressive` | all of the above + analog_cleanup (atadenoise) |

Change presets at any time before probing; `PreprocPreview` on the home
page renders before/after frames for each enabled stage using the preview
endpoint.

## When to turn things off

- Clean digital clips (a phone, an action cam) → default to `none` or
  just `keyframe_score`. The analog stages add time without a quality
  gain.
- Non-FPV content → turn off `fisheye` and `osd_mask`.
- You see the reconstruction baking-in edge artefacts → raise
  `osd_mask_std_threshold` or disable `osd_detect_text`.
- Mass of micro-geometry appears "tilted" → `rs_correction` is
  over-correcting; override `rs_shear_px_per_row` toward 0 or disable.

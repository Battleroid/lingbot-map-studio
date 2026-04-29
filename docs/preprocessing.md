# Preprocessing pipeline

The studio's default footage profile is `hi-def · no preproc` — a clip from
a phone, mirrorless, action cam, or HD drone goes straight to the
reconstruction model with nothing in between. The preprocessing pipeline
exists for the sources where the model does need help: low-bitrate analog
captures, fisheye lenses, footage with persistent HUD overlays, rolling-
shutter skew on fast pans.

Each stage is idempotent per-frame and writes into a content-addressed
cache under `data/preproc_cache/{hash}/`, so retries and reexports skip
work.

## Footage profiles

The dropdown at the top of every config panel is a one-click way to
enable a stage bundle. The profile is derived from the toggle state, not
stored — touching an individual toggle flips the picker to `custom`.

| Profile | Stages enabled |
| --- | --- |
| `hi-def · no preproc` (default) | (nothing) |
| `fpv · analog` | denoise + deflicker + osd_mask + color_norm + rs_correction + unsharp deblur + keyframe_score |
| `fpv · aggressive` | all of the above + atadenoise |
| `custom` | manual mix; only shown when toggles don't match a named profile |

The probe heuristic (`worker/app/pipeline/probe.py:_is_analog_fpv`) auto-
selects `fpv · analog` after upload when the clip looks low-fi (≤720p
with low bitrate, or analog-era codec). Drop a 1080p+ phone clip and the
profile stays at `hi-def`.

## Stages

### 1. Fisheye unwrap

- `preproc_fisheye` → `v360` filter.
- `fisheye_in_fov` / `fisheye_out_fov` → source lens HFOV / target
  diagonal FOV after unwrap.

Anything wider than ~120° gives the reconstructor false geometry if left
distorted. Common cameras:

- GoPro Wide → 122°
- Insta360 X-series → 170°
- Caddx Ratel 2.1 → 165° (FPV)
- RunCam Phoenix 2 → 155° (FPV)
- Foxeer Predator Micro → 160° (FPV)

### 2. Temporal denoise + deflicker

- `preproc_denoise` → `hqdn3d` (temporal + spatial denoise) + ffmpeg
  `deflicker`.
- `preproc_deflicker` → standalone `deflicker` without the hqdn3d pair —
  useful when brightness jitter is the dominant artefact and you don't
  want hqdn3d's spatial blur.

Useful for any noisy or low-light source: phone clips at night, drone
footage at dusk, low-bitrate streams.

### 3. Heavier atadenoise

- `preproc_analog_cleanup` → `atadenoise` (heavier, tuned for chroma
  noise / dot crawl).

Expensive — only enable for genuinely rough captures (analog DVR rips,
very low-bitrate streams) where the standard denoise still leaves
artifacts. No-op on clean digital clips.

### 4. Static-overlay mask + inpaint

- `preproc_osd_mask` → detect static pixels (telemetry text, timer,
  battery indicator, watermark, station logo) and inpaint them out.
- `osd_mask_samples` — how many frames to sample.
- `osd_mask_std_threshold` — per-pixel stddev cutoff (0-255 scale).
- `osd_mask_dilate` — grow the mask to catch anti-aliased text edges.
- `osd_detect_text` — flag regions near edges in most sampled frames
  (catches changing numeric HUD values).
- `osd_edge_persist_frac` — stricter/looser edge-persistence gate.

Worth enabling for any footage with a persistent overlay, not just FPV
HUDs — GoPro battery indicators, phone screen-recordings with status
bars, broadcast watermarks. Without this they become false geometry
fixed in camera space.

The resulting mask is also exposed to SLAM processors as a per-frame
ignore mask, so their feature matcher doesn't anchor on baked-in text.

### 5. White-balance + histogram stretch

- `preproc_color_norm` → per-frame grey-world WB + 1/99 percentile
  histogram stretch.

Recovers natural colour on any source with a colour cast — tungsten
indoor footage, green-tinted analog feeds, magenta-cast drone clips.
Cheap, CPU-only.

### 6. Rolling-shutter correction

- `preproc_rs_correction` → estimate global y-shear from optical flow
  between rows, apply the inverse affine warp.
- `rs_shear_px_per_row` → override the estimate (blank = auto). Values
  smaller than ±0.02 px/row skip the warp entirely.

Most CMOS cameras (phones, action cams, FPV digital cameras, drones with
electronic shutters) read the sensor line-by-line; fast pans show
characteristic skew. v1 handles the dominant-skew case only — full
per-row RS needs gyro data and is out of scope.

### 7. Motion deblur

- `preproc_deblur: "none" | "unsharp" | "nafnet"`.
- `deblur_sharpness_gate` → apply the deblur filter only to frames whose
  Laplacian variance is below this fraction of the clip median. 1.0 =
  every frame, 0.6 (default) = the blurriest ~60%.

`nafnet` is a learned single-image deblur; the checkpoint lazy-downloads
via the existing cache, and falls back to `unsharp` if absent.

### 8. Keyframe scoring

- `preproc_keyframe_score` → compute per-frame sharpness (Laplacian
  variance) + motion magnitude (optical-flow L2).
- `keyframe_min_sharpness_frac` / `keyframe_min_motion_px` → ingest-side
  gates.

Writes `frame_scores.jsonl`; SLAM backends with
`keyframe_policy=score_gated` read this instead of re-scoring. Cheap to
leave on — backends that don't consume the file just ignore it.

## When to turn things off

- Clean digital clips (phone, mirrorless, action cam, HD drone) → leave
  the profile on `hi-def`. The cleanup stages add time without a
  quality gain on already-clean footage.
- Non-FPV / non-fisheye content → keep `fisheye` off. The unwrap is
  destructive on a near-rectilinear source.
- No persistent overlays in the footage → keep `osd_mask` off (it
  always finds *something* to mask).
- You see the reconstruction baking-in edge artefacts → raise
  `osd_mask_std_threshold` or disable `osd_detect_text`.
- Mass of micro-geometry appears "tilted" → `rs_correction` is
  over-correcting; override `rs_shear_px_per_row` toward 0 or disable.

## FPV-specific notes

Low-bitrate analog FPV footage — what comes out of a DVR recording an
analog receiver — fails reconstruction in very specific ways: chroma
noise becomes false geometry, OSD telemetry text fixes itself in camera
space, rolling-shutter skew warps the scale, and motion blur starves
keyframe selection. The `fpv · analog` profile is tuned for this failure
mode.

Use `fpv · aggressive` (adds atadenoise) when the standard bundle still
leaves visible chroma noise or dot crawl — common on VHS rips or analog
receivers with weak signal. The atadenoise stage is the single most
expensive step in the pipeline; reserve it for sources that need it.

When tuning fisheye unwrap for an FPV micro cam, the source HFOV usually
falls in the 150-170° range. Check the manufacturer spec or measure with
a known straight edge in frame.

# lingbot-map-studio

Browser studio for three-mode 3D reconstruction from local video on a local GPU:

1. **Lingbot** — [lingbot-map](https://github.com/Robbyant/lingbot-map) feed-forward reconstruction. Upload video(s), get a point cloud + textured mesh + camera path, clean up the mesh with lasso-cull / fill-holes / decimate / smooth, export GLB / PLY / OBJ.
2. **SLAM** — DROID-SLAM, MASt3R-SLAM, DPVO, or MonoGS/Photo-SLAM. Tuned for low-quality analog FPV footage via an upstream preprocessing pipeline (deblur, rolling-shutter correction, color normalization, analog-noise cleanup, keyframe scoring). Outputs pose graph + sparse/dense cloud + optional Poisson mesh; MonoGS additionally emits a Gaussian-Splat scene.
3. **Gaussian Splat training** — a `gsplat`-based trainer that consumes a completed SLAM (or lingbot) job's frames + poses + initial cloud. Renders the growing splat natively in the same three.js canvas during training.

## Requirements

- Linux or WSL2 host with an NVIDIA RTX 3000 series or newer (Ampere/Ada/Hopper — bf16 path).
- NVIDIA driver with CUDA 12.x support.
- `nvidia-container-toolkit` installed and the `nvidia` Docker runtime configured.
- Docker 25+ with Compose v2.
- ~15 GB free disk for the CUDA base image + checkpoints + per-job artifacts.

## Quick start

```bash
cp .env.example .env
docker compose build
docker compose up
```

Open http://localhost:3000.

## What to expect on the first job

The worker lazy-downloads checkpoints from `huggingface.co/robbyant/lingbot-map` into a named Docker volume. The download progress streams into the log pane on the job page. Subsequent jobs reuse the cached checkpoint.

## Architecture

```
┌───────────────┐   POST /api/jobs (enqueues)
│   web (3000)  │──────────────┐
└───────────────┘              ▼
       ▲                ┌──────────────┐
       │ WS /stream     │  api (8000)  │  SQLite job_queue
       │ events         └──────────────┘
       │                  ▲    ▲    ▲
       │                  │    │    │ claim_next(worker_class)
       ▼                  │    │    │
┌──────────────┐   ┌──────┴──┐ ┌─┴───────┐ ┌─┴──────┐
│ shared data/ │◀──│ lingbot │ │  slam   │ │   gs   │
│  models/     │   │ worker  │ │ worker  │ │ worker │
└──────────────┘   └─────────┘ └─────────┘ └────────┘
```

- **API** (`worker/app/main.py`) serves HTTP + WebSocket only. On `POST /api/jobs` it validates the discriminated-union config and writes a row into `job_queue` with the appropriate `worker_class`.
- **Workers** (`worker/app/worker_main.py`) each loop on `claim_next_job(worker_class)`. Three separate images pin incompatible CUDA/torch matrices: `worker-lingbot` (torch 2.9/cu128), `worker-slam` (backend-specific CUDA extensions), `worker-gs` (gsplat-matched wheels).
- **Cross-process events** go through the SQLite `job_events` table; cancellation via a polled flag on the job row.

## Modes

### Lingbot

Same feed-forward reconstruction as before. Presets:

- **Low-fi drone** — sky masking on, higher confidence threshold, more aggressive keyframe dropping.
- **High-fi** — sky masking off, lower confidence threshold, more scale frames and camera iterations.
- **FPV drone** — adds fisheye unwrap + OSD mask + denoise.

Knobs: `model_id`, `mode` (streaming/windowed), `window_size`, `overlap_size`, `image_size`, `fps`, `first_k`, `stride`, `mask_sky`, `conf_threshold`, `keyframe_interval`, `num_scale_frames`, `camera_num_iterations`, `use_sdpa`, `offload_to_cpu`, plus the FPV preprocessing block.

Post-inference, `conf_threshold`, `mask_sky`, and `show_cam` can be re-applied via `POST /api/jobs/{id}/reexport` without re-running the GPU pass (cached pred tensors).

### SLAM

Four backends, all behind the same `Processor` / `SlamSession` interface:

| Backend | Best for | Notes |
| --- | --- | --- |
| `mast3r_slam` | Analog FPV (default) | Calibration-free. Robust to bad intrinsics. |
| `droid_slam` | High-fidelity indoor/small scenes | Dense; highest VRAM. |
| `dpvo` | Long clips on small cards | Patch-based deep VO; sparse cloud. |
| `monogs` | "I want a splat now" | Photo-SLAM. Emits a Gaussian-Splat scene incrementally. |

Shared config: `max_frames`, `downscale`, `stride`, `fps`, `calibration` (auto / manual fx/fy/cx/cy), `keyframe_policy` (score_gated / translation / hybrid), `partial_snapshot_every`, `run_poisson_mesh`, plus the FPV preprocessing block. Per-backend configs add: DROID `buffer_size` + `global_ba_iters`; MASt3R `match_threshold` + `window_size`; DPVO `patch_per_frame` + `buffer_keyframes`; MonoGS `refine_iters` + `prune_opacity`.

VRAM expectations on a 24 GB card:

- DROID-SLAM: 12-20 GB depending on `buffer_size` + input resolution.
- MASt3R-SLAM: 6-10 GB typical.
- DPVO: 2-4 GB — fits on 8 GB cards.
- MonoGS: 8-14 GB, grows with scene complexity.

### Gaussian Splat training

Chains off a `ready` SLAM (or lingbot) job — no upload. The `gsplat` processor reads the source job's `frames/`, `pose_graph.json` (or `camera_path.json` fallback), and `reconstruction.ply` as init.

Knobs: `iterations`, `sh_degree`, `densify_interval`, `prune_interval`, `prune_opacity`, `init_from` (point_cloud / random), `initial_resolution`, `upsample_at_iter`, `preview_every_iters`, `bake_mesh_after`.

Live preview: every `preview_every_iters` training steps the trainer writes `partial_splat_NNNN.ply` and emits a `partial_splat` artifact event; the viewer's `SplatLayer` swaps in the latest without stealing the user's camera.

## FPV preprocessing pipeline

Order (each stage independently toggleable via the config panel):

1. **Analog noise cleanup** — `hqdn3d` + `atadenoise` for VHS/chroma-noise kill.
2. **Deflicker / exposure normalization** — ffmpeg `deflicker` + median-luma pass.
3. **Color normalization** — grey-world WB + histogram stretch.
4. **Rolling-shutter correction** — global y-shear estimate from optical flow, applied as an inverse affine warp.
5. **Motion deblur** — `unsharp` (classical, default) or `nafnet` (learned, checkpoint lazy-downloaded).
6. **Fisheye unwrap** — existing `v360` filter.
7. **OSD/HUD mask + inpaint** — static-pixel detector; mask is also exposed to SLAM backends as per-frame ignore.
8. **Keyframe scoring** — per-frame Laplacian variance + optical-flow L2; written to `frame_scores.jsonl` and consumed by SLAM backends whose `keyframe_policy=score_gated`.

Presets: `none`, `analog fpv (default)`, `aggressive`. See `docs/fpv_preprocessing.md`.

## Berkeley Mono

If you have a licensed [Berkeley Mono](https://usgraphics.com/products/berkeley-mono) woff2 file, drop it at `web/public/fonts/berkeley-mono.woff2` and set `BERKELEY_MONO=1` in the web image build args. Otherwise the UI falls back to Roboto Mono served from Google Fonts by `next/font`.

## Layout

- `worker/` — Python 3.11 FastAPI service + worker claim loop.
  - `app/processors/` — per-mode processors (`lingbot`, `slam/*`, `gsplat/*`) behind a shared `Processor` interface.
  - `app/pipeline/` — ingest + FPV preprocessing stages + checkpoint cache + VRAM watchdog.
  - `app/jobs/` — schema (discriminated-union `JobConfig`), runner, cancel token, events, store.
  - `app/mesh/` — pymeshlab-backed mesh ops (cull / fill_holes / decimate / smooth / Poisson).
  - `Dockerfile.lingbot`, `Dockerfile.slam`, `Dockerfile.gs` — one image per worker class.
- `web/` — Next.js 16 (app router) + react-three-fiber viewer.
  - `src/components/ModePicker.tsx`, `ConfigPanel.tsx`, `SlamConfigPanel.tsx`, `GsplatConfigPanel.tsx` — mode-aware job creation.
  - `src/components/ToolsPanel.tsx` + `tools/*` — per-mode tool sets.
  - `src/components/Viewer/{PointCloud,MeshLayer,CameraPath,SplatLayer}.tsx` — composable scene layers.
- `data/` — bind-mounted upload/artifact store (ignored by git).
- `models/` — HF checkpoint volume (Docker-managed).
- `docs/` — `processors.md`, `fpv_preprocessing.md`.

## Ports

- `8000` — worker API + WebSocket.
- `3000` — web UI.

## License

Apache 2.0 (follows lingbot-map upstream).

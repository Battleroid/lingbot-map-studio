# lingbot-map-studio

Browser studio for [lingbot-map](https://github.com/Robbyant/lingbot-map) — upload a video (or several covering one scene), run feed-forward 3D reconstruction on a local RTX-class GPU, view point cloud / colored point cloud / wireframe / textured mesh in the browser, clean up the result with lasso-cull / fill-holes / decimate / smooth, and download GLB / PLY / OBJ.

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

## Configuration presets

- **Low-fi drone** — sky masking on, higher confidence threshold, more aggressive keyframe dropping.
- **High-fi** — sky masking off, lower confidence threshold, more scale frames and camera iterations.

All knobs exposed in the UI:
`model_id`, `mode` (streaming/windowed), `window_size`, `overlap_size`, `image_size`, `fps`, `first_k`, `stride`, `mask_sky`, `conf_threshold`, `keyframe_interval`, `num_scale_frames`, `camera_num_iterations`, `use_sdpa`, `offload_to_cpu`.

Post-inference, `conf_threshold`, `mask_sky`, and `show_cam` can be re-applied via `POST /api/jobs/{id}/reexport` without re-running the GPU pass (cached pred tensors).

## Berkeley Mono

If you have a licensed [Berkeley Mono](https://usgraphics.com/products/berkeley-mono) woff2 file, drop it at `web/public/fonts/berkeley-mono.woff2` and set `BERKELEY_MONO=1` in the web image build args. Otherwise the UI falls back to Roboto Mono served from Google Fonts by `next/font`.

## Layout

- `worker/` — Python 3.11 FastAPI service: ingest, GPU inference, mesh ops, event stream.
- `web/` — Next.js 15 (app router) + react-three-fiber viewer.
- `data/` — bind-mounted upload/artifact store (ignored by git).
- `models/` — HF checkpoint volume (Docker-managed).

## Ports

- `8000` — worker API + WebSocket.
- `3000` — web UI.

## License

Apache 2.0 (follows lingbot-map upstream).

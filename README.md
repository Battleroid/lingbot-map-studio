# lingbot-map-studio

Browser studio for three-mode 3D reconstruction from local video on a local
GPU. Drop a clip from your phone, mirrorless, action cam, or HD drone — by
default the studio passes it through to the reconstruction model unmodified.
Optional preprocessing bundles handle rougher sources (analog FPV captures,
heavy chroma noise, rolling-shutter skew) when the probe heuristic detects
them or you opt in manually.

The three modes:

1. **Lingbot** — [lingbot-map](https://github.com/Robbyant/lingbot-map) feed-forward reconstruction. Upload video(s), get a point cloud + textured mesh + camera path, clean up the mesh with lasso-cull / fill-holes / decimate / smooth, export GLB / PLY / OBJ.
2. **SLAM** — DROID-SLAM, MASt3R-SLAM, DPVO, or MonoGS/Photo-SLAM. Outputs pose graph + sparse/dense cloud + optional Poisson mesh; MonoGS additionally emits a Gaussian-Splat scene. MASt3R-SLAM is the safe default — calibration-free and robust to unknown camera intrinsics.
3. **Gaussian Splat training** — a `gsplat`-based trainer that consumes a completed SLAM (or lingbot) job's frames + poses + initial cloud. Renders the growing splat natively in the same three.js canvas during training.

## Requirements

- Linux or WSL2 host with an NVIDIA RTX 3000 series or newer (Ampere/Ada/Hopper — bf16 path).
- NVIDIA driver with CUDA 12.x support.
- `nvidia-container-toolkit` installed and the `nvidia` Docker runtime configured.
- Docker 25+ with Compose v2.
- ~15 GB free disk for the CUDA base image + checkpoints + per-job artifacts.

## Quick start

Pre-built images are published to GHCR after every green CI run. The
default `make up` pulls them — no local build required.

```bash
make doctor   # one-time: confirm docker + gpu + nvidia-container-toolkit
make up       # pull ghcr.io images + start (foreground)
```

Open http://localhost:3000.

If you're hacking on the worker / api / web code and want to build from
source instead of pulling, use `make up-build` (slow first run, ~5 min
for the shared base image alone).

Run `make help` for the full target list (`down`, `logs`, `restart`,
`shell-api`, `clean`, …).

The Makefile is a thin wrapper over Compose. You can drop down to
`docker compose` directly any time:

```bash
docker compose -f docker-compose.yml -f docker-compose.prebuilt.yml pull
docker compose -f docker-compose.yml -f docker-compose.prebuilt.yml up
```

## Scanning from a phone — HTTPS setup with mkcert

The `/capture` page uses `navigator.mediaDevices.getUserMedia` to pull
frames off the phone's camera and stream them to the SLAM tracker live.
Mobile browsers refuse camera access over plain `http://` (only
`localhost` is exempt, which you can't loopback into from a phone), so
the studio needs to be reachable over `https://` for the capture flow
to work. The default `make up` keeps everything on plain HTTP — that's
fine for the upload flow on the same machine. Use `make up-https`
when you actually want to scan from a phone.

The recipe is **mkcert** (a tiny local CA) + **Caddy** as a reverse
proxy. Caddy is already wired into `docker-compose.yml` under
`profiles: [https]`; the only manual step is generating + trusting a
cert pair.

**One-time on the host (laptop / desktop running the studio):**

```bash
# 1. install mkcert (Debian/Ubuntu)
sudo apt-get install -y libnss3-tools
curl -L "https://github.com/FiloSottile/mkcert/releases/latest/download/mkcert-v1.4.4-linux-amd64" \
  -o /usr/local/bin/mkcert
sudo chmod +x /usr/local/bin/mkcert
mkcert -install

# 2. find the host's LAN IP (e.g. 192.168.1.42)
ip -4 addr show | awk '/inet 192/ {print $2}'

# 3. generate a cert pair valid for both a friendly name and the LAN IP
mkdir -p caddy/certs
mkcert -cert-file caddy/certs/cert.pem \
       -key-file  caddy/certs/key.pem \
       studio.local 192.168.1.42

# 4. start the stack with the https profile
STUDIO_HOSTNAME=studio.local make up-https
```

**One-time on the phone (Android — iOS notes below):**

1. Copy `$(mkcert -CAROOT)/rootCA.pem` from the host to the phone (USB,
   AirDrop equivalent, or `python3 -m http.server` over LAN).
2. Settings → Security → Encryption & credentials → Install a
   certificate → **CA certificate** → pick the `rootCA.pem` file.
   Confirm "Install anyway".
3. Add a hosts entry for `studio.local` → host LAN IP. The simplest
   way is to skip the friendly name and just open the URL by IP
   instead (`https://192.168.1.42/capture`); the cert covers both
   names so either works.

Open `https://studio.local/capture` (or the IP form) from the phone's
browser. The browser should accept the cert without warnings; the page
will prompt for camera access, you tap allow, and the live preview
kicks off.

**iOS:** the flow is the same but the cert install lives at General →
VPN & Device Management, and you also have to flip a separate switch
under General → About → Certificate Trust Settings to fully trust the
mkcert root. Apple does this on purpose to make user-installed CAs
slightly less dangerous.

**Why mkcert and not Let's Encrypt / Caddy auto-HTTPS?** Both
alternatives need a publicly-resolvable hostname and inbound port 80.
The studio is intended to run on a LAN behind a home router, so
self-signed via a locally-trusted CA is the path with the fewest
moving parts. If you do expose the studio to the public internet
(via Cloudflare Tunnel, Tailscale Funnel, or a real VPS) you can
delete the `tls /certs/...` line from `caddy/Caddyfile` and Caddy
will provision a Let's Encrypt cert automatically.

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
- **FPV drone** — adds fisheye unwrap + OSD mask + denoise (for analog FPV captures).

Knobs: `model_id`, `mode` (streaming/windowed), `window_size`, `overlap_size`, `image_size`, `fps`, `first_k`, `stride`, `mask_sky`, `conf_threshold`, `keyframe_interval`, `num_scale_frames`, `camera_num_iterations`, `use_sdpa`, `offload_to_cpu`, plus the preprocessing block (off by default, see [docs/preprocessing.md](docs/preprocessing.md)).

Post-inference, `conf_threshold`, `mask_sky`, and `show_cam` can be re-applied via `POST /api/jobs/{id}/reexport` without re-running the GPU pass (cached pred tensors).

### SLAM

Four backends, all behind the same `Processor` / `SlamSession` interface:

| Backend | Best for | Notes |
| --- | --- | --- |
| `mast3r_slam` | Default — most footage | Calibration-free. Robust to unknown / inaccurate intrinsics. |
| `droid_slam` | High-fidelity indoor/small scenes | Dense; highest VRAM. |
| `dpvo` | Long clips on small cards | Patch-based deep VO; sparse cloud. |
| `monogs` | "I want a splat now" | Photo-SLAM. Emits a Gaussian-Splat scene incrementally. |

Shared config: `max_frames`, `downscale`, `stride`, `fps`, `calibration` (auto / manual fx/fy/cx/cy), `keyframe_policy` (score_gated / translation / hybrid), `partial_snapshot_every`, `run_poisson_mesh`, plus the preprocessing block (off by default; opt into the FPV bundle when needed — see [docs/preprocessing.md](docs/preprocessing.md)). Per-backend configs add: DROID `buffer_size` + `global_ba_iters`; MASt3R `match_threshold` + `window_size`; DPVO `patch_per_frame` + `buffer_keyframes`; MonoGS `refine_iters` + `prune_opacity`.

VRAM expectations on a 24 GB card:

- DROID-SLAM: 12-20 GB depending on `buffer_size` + input resolution.
- MASt3R-SLAM: 6-10 GB typical.
- DPVO: 2-4 GB — fits on 8 GB cards.
- MonoGS: 8-14 GB, grows with scene complexity.

### Gaussian Splat training

Chains off a `ready` SLAM (or lingbot) job — no upload. The `gsplat` processor reads the source job's `frames/`, `pose_graph.json` (or `camera_path.json` fallback), and `reconstruction.ply` as init.

Knobs: `iterations`, `sh_degree`, `densify_interval`, `prune_interval`, `prune_opacity`, `init_from` (point_cloud / random), `initial_resolution`, `upsample_at_iter`, `preview_every_iters`, `bake_mesh_after`.

Live preview: every `preview_every_iters` training steps the trainer writes `partial_splat_NNNN.ply` and emits a `partial_splat` artifact event; the viewer's `SplatLayer` swaps in the latest without stealing the user's camera.

## Preprocessing pipeline

The default footage profile is `hi-def · no preproc` — phone, mirrorless,
action cam, and HD drone clips pass straight through to the reconstruction
model. Pick `fpv · analog` (or `fpv · aggressive`) from the dropdown at the
top of any config panel to enable the cleanup bundle for low-bitrate analog
captures (DVR rips, analog receivers). Individual stages can also be
toggled under the panel's "advanced" disclosure.

Stages, in pipeline order:

1. **Fisheye unwrap** — `v360` filter for any lens wider than ~120° (action cams in superview, FPV micro cams 150-170°).
2. **Temporal denoise + deflicker** — `hqdn3d` + ffmpeg `deflicker` + median-luma pass. Useful for any noisy / low-light source.
3. **Heavier atadenoise** — `atadenoise` tuned for chroma noise and dot crawl. Reserve for genuinely rough captures.
4. **Static-overlay mask + inpaint** — detect persistent overlays (FPV HUD, GoPro battery indicator, watermarks, station logos) and inpaint them out.
5. **White-balance + histogram stretch** — grey-world WB + 1/99-percentile stretch. Recovers natural colour on tungsten / tinted footage.
6. **Rolling-shutter correction** — global y-shear estimate from optical flow, applied as an inverse affine warp. Useful for any CMOS source (phones, action cams, FPV digital, drones).
7. **Motion deblur** — `unsharp` (classical, default) or `nafnet` (learned, checkpoint lazy-downloaded).
8. **Keyframe scoring** — per-frame Laplacian variance + optical-flow L2; written to `frame_scores.jsonl` and consumed by SLAM backends whose `keyframe_policy=score_gated`.

Profiles: `hi-def · no preproc` (default), `fpv · analog`, `fpv · aggressive`, `custom`. See [docs/preprocessing.md](docs/preprocessing.md) for per-stage details and FPV-specific notes.

The probe heuristic auto-selects the `fpv · analog` profile when an
uploaded clip looks low-fi (≤720p with low bitrate, or analog-era codec).
Drop a 1080p+ phone clip and the dropdown stays at `hi-def`.

## Berkeley Mono

If you have a licensed [Berkeley Mono](https://usgraphics.com/products/berkeley-mono) woff2 file, drop it at `web/public/fonts/berkeley-mono.woff2` and set `BERKELEY_MONO=1` in the web image build args. Otherwise the UI falls back to Roboto Mono served from Google Fonts by `next/font`.

## Layout

- `worker/` — Python 3.11 FastAPI service + worker claim loop.
  - `app/processors/` — per-mode processors (`lingbot`, `slam/*`, `gsplat/*`) behind a shared `Processor` interface.
  - `app/pipeline/` — ingest + preprocessing stages + checkpoint cache + VRAM watchdog.
  - `app/jobs/` — schema (discriminated-union `JobConfig`), runner, cancel token, events, store.
  - `app/mesh/` — pymeshlab-backed mesh ops (cull / fill_holes / decimate / smooth / Poisson).
  - `Dockerfile.lingbot`, `Dockerfile.slam`, `Dockerfile.gs` — one image per worker class.
- `web/` — Next.js 16 (app router) + react-three-fiber viewer.
  - `src/components/ModePicker.tsx`, `ConfigPanel.tsx`, `SlamConfigPanel.tsx`, `GsplatConfigPanel.tsx` — mode-aware job creation.
  - `src/components/ToolsPanel.tsx` + `tools/*` — per-mode tool sets.
  - `src/components/Viewer/{PointCloud,MeshLayer,CameraPath,SplatLayer}.tsx` — composable scene layers.
- `data/` — bind-mounted upload/artifact store (ignored by git).
- `models/` — HF checkpoint volume (Docker-managed).
- `docs/` — `processors.md`, `preprocessing.md`.

## Ports

- `8000` — worker API + WebSocket.
- `3000` — web UI.

## License

Apache 2.0 (follows lingbot-map upstream).

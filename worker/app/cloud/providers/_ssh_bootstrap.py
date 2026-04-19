"""Shared user-data / bootstrap script builder.

Providers that hand us a bare VM (Lambda Labs, Vast.ai when using SSH
mode, Paperspace Core, AWS EC2, GCP GCE, Azure VM) all need the same
boot sequence: make sure docker exists, pull the remote-worker image,
run it with the dispatcher's env injected. RunPod and RunPod Serverless
don't need this — their pod API owns the container lifecycle.

The returned script is cloud-init-compatible: it's a `#!/usr/bin/env
bash` header followed by idempotent `set -euo pipefail` steps. Each
provider pastes it verbatim into whatever field their API calls
"user_data" (EC2, Lambda), "startup-script" (GCP), "onstart" (Vast SSH
mode), etc.

The env map is built from the `LaunchSpec` standard four — broker URL,
job token, worker mode=remote, worker class — plus whatever per-spec
extras live in `spec.env`. Kept in this module so every SSH-bootstrapped
provider picks up env additions automatically.
"""

from __future__ import annotations

import shlex
from typing import Iterable

from app.cloud.providers.base import LaunchSpec


def _env_map(spec: LaunchSpec) -> dict[str, str]:
    """Standard studio env + per-spec extras, in that order so adapter
    overrides win. The remote worker entrypoint reads these four names
    directly; everything else is opaque pass-through."""
    env = {
        "STUDIO_BROKER_URL": spec.broker_url,
        "STUDIO_JOB_TOKEN": spec.job_token,
        "WORKER_MODE": "remote",
        "WORKER_CLASS": spec.worker_class,
    }
    env.update(spec.env)
    return env


def _render_env_flags(env: dict[str, str]) -> str:
    """Turn an env dict into a series of `-e KEY=VALUE` args, shell-quoted
    so embedded spaces / quotes / newlines in the job token don't escape
    the docker command."""
    parts: list[str] = []
    for key, value in env.items():
        parts.append(f"-e {shlex.quote(key)}={shlex.quote(value)}")
    return " ".join(parts)


def build_bootstrap_script(
    spec: LaunchSpec,
    *,
    container_name: str | None = None,
    extra_docker_args: Iterable[str] = (),
) -> str:
    """Build the bash user-data script for a bare-VM provider.

    Idempotent: safe to re-run (reboot, manual SSH). Installs docker via
    the convenience script if `docker` isn't on PATH, enables + starts
    the daemon, pulls our image, and `docker run`s it with the full env
    map. We use `--restart=no` so a successful job exit lets the orphan
    sweeper + provider-side terminate path wind things down; the remote
    worker is expected to shut down cleanly once it finalizes its one
    job.

    `container_name` defaults to `lingbot-<job_id>` so the ops team can
    `docker ps | grep lingbot-` on the host and find the thing. Extra
    `docker run` flags (e.g. `--gpus all`, `--shm-size=8g`) come in via
    `extra_docker_args` — providers know their own GPU runtime quirks.
    """
    env = _env_map(spec)
    env_flags = _render_env_flags(env)
    name = container_name or f"lingbot-{spec.job_id}"
    extras = " ".join(extra_docker_args)
    image = shlex.quote(spec.image)
    name_q = shlex.quote(name)

    return f"""#!/usr/bin/env bash
set -euo pipefail

log() {{ echo "[lingbot-bootstrap] $*"; }}

# --- docker install (no-op if already present) --------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "docker not found, installing via convenience script"
  curl -fsSL https://get.docker.com | sh
fi

# --- make sure the daemon is up ----------------------------------------
if command -v systemctl >/dev/null 2>&1; then
  systemctl enable --now docker || true
fi

# --- pull + run ---------------------------------------------------------
log "pulling image {image}"
docker pull {image}

log "starting remote worker container {name_q}"
docker run -d \\
  --name {name_q} \\
  --restart=no \\
  {extras} \\
  {env_flags} \\
  {image}

log "bootstrap complete"
"""


__all__ = ["build_bootstrap_script"]

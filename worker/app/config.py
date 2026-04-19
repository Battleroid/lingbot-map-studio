from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: Path = Path("/data")
    models_dir: Path = Path("/models")
    log_level: str = "info"
    use_flashinfer: bool = False
    hf_repo_id: str = "robbyant/lingbot-map"
    default_model_id: str = "lingbot-map"
    event_replay_size: int = 2048
    cors_origins: list[str] = ["*"]
    # Process-wide hard cap on CUDA memory, as a fraction of total device VRAM.
    # When exceeded, PyTorch raises torch.cuda.OutOfMemoryError cleanly — a
    # much better outcome than WSL2 paging VRAM and hanging the host.
    vram_limit_fraction: float = 0.85
    # Watchdog poll interval and default soft-limit used when a job doesn't
    # supply its own cap. In GB.
    vram_watchdog_interval_s: float = 2.0
    # Soft limit set below the hard process cap (~0.85 × 24 = 20.4 GB) so the
    # watchdog can abort with a friendly message *before* PyTorch raises
    # torch.cuda.OutOfMemoryError.
    vram_default_soft_limit_gb: float = 19.0

    # --- Cloud execution (Phase R1) ---
    # How a remote worker reaches this API. Used when dispatching a job to a
    # rented pod; the pod's env gets this as STUDIO_BROKER_URL.
    cloud_broker_public_url: str = "http://localhost:8000"
    # Shared secret that signs the per-job HMAC tokens issued to remote
    # workers. Rotate by changing this and letting in-flight jobs finish
    # under the old key (tokens carry their expiry; the signer key is
    # checked on every request).
    cloud_broker_hmac_key: str = "change-me-in-production"
    # Lifetime of a per-job broker token. Long enough for a cold-start +
    # ingest + inference + export on a slow provider; short enough that a
    # leaked token times out before it's useful.
    cloud_broker_token_ttl_s: int = 6 * 60 * 60
    # Studio-wide hard upper bound on per-job cloud spend. Dispatcher
    # refuses to launch if the estimate exceeds this regardless of the
    # job-level cap. Defaults to $50.
    cloud_cost_cap_cents_default: int = 5000

    model_config = SettingsConfigDict(env_file=None, case_sensitive=False)

    def ensure_dirs(self) -> None:
        (self.data_dir / "jobs").mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / "jobs" / job_id

    def job_uploads(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "uploads"

    def job_frames(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "frames"

    def job_artifacts(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "artifacts"

    def sqlite_path(self) -> Path:
        return self.data_dir / "studio.db"


settings = Settings()

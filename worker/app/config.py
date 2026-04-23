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

    # --- Per-provider credentials (Phase R2+). ---
    # Each provider adapter reads only the fields it needs; unset
    # fields leave that provider unregistered (its import-time self-
    # registration checks the key and bails if missing). This keeps
    # the fake provider the only one available in a fresh dev install
    # so tests + CI stay lean.
    runpod_api_key: str = ""
    runpod_api_base: str = "https://rest.runpod.io/v1"
    # Vast.ai — account-level API key.
    vast_api_key: str = ""
    vast_api_base: str = "https://console.vast.ai/api/v0"
    # Lambda Labs — account-level API key.
    lambda_labs_api_key: str = ""
    lambda_labs_api_base: str = "https://cloud.lambdalabs.com/api/v1"
    # Paperspace — account-level API key. Gradient + Core share auth.
    paperspace_api_key: str = ""
    paperspace_api_base: str = "https://api.paperspace.io"
    # AWS / GCP / Azure lean on their own SDKs for credentials; the
    # URLs below are only for SDK overrides (LocalStack, private
    # endpoints, gov-cloud). Empty = use SDK defaults.
    aws_endpoint_url: str = ""
    aws_region_default: str = "us-east-1"
    gcp_service_account_json: str = ""
    gcp_region_default: str = "us-central1"
    azure_subscription_id: str = ""
    azure_region_default: str = "eastus"

    # --- Artifact storage transport (Phase R5). ---
    # Remote workers upload artifacts through one of two backends:
    # - `broker` (default): chunked PUT to /api/worker/artifacts/{name},
    #   same volume that already served local jobs.
    # - `minio`: dispatcher-issued pre-signed PUT URL to an S3-compatible
    #   bucket. The studio pulls bytes back locally on finalize so the
    #   viewer still reads from the artifacts dir. Enabled by setting
    #   `cloud_storage="minio"` and populating the minio_* fields.
    cloud_storage: str = "broker"
    minio_endpoint_url: str = "http://minio:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "lingbot-artifacts"
    minio_region: str = "us-east-1"
    # TTL for pre-signed PUT URLs handed to remote workers. Long enough
    # to upload a multi-hundred-MB splat on a slow uplink; short enough
    # that a leaked URL times out before it's useful.
    minio_presign_ttl_s: int = 15 * 60

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

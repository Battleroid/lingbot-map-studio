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

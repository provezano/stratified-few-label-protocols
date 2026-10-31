"""Project-wide settings (environment variables or a local ``.env`` file)."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Paths and execution parameters shared by the CLI and the runner."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SIMAUDIT_",
        case_sensitive=False,
        extra="ignore",
    )

    data_cache_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "cache",
        description="Where OpenML downloads are cached (outside git).",
    )
    registry_path: Path = Field(
        default=PROJECT_ROOT / "data" / "registry.yaml",
        description="Pinned dataset registry (IDs, versions, targets, groups).",
    )
    results_dir: Path = Field(
        default=PROJECT_ROOT / "results",
        description="Root directory for per-job result files (outside git).",
    )
    n_workers: int = Field(
        default=11,
        ge=1,
        description="Parallel worker processes (M3 Max: 12 P-cores, one kept free).",
    )
    base_seed: int = Field(
        default=20261020,
        description="Root of the SeedSequence tree for splits and searches.",
    )
    tabclustpfn_root: Path = Field(
        default=PROJECT_ROOT.parent
        / "third_party_src"
        / "tabclustpfn_src"
        / "x"
        / "TabClustPFN-d5d25a9ae4db922c2efb1b07ee3fc7a818188819",
        description="TabClustPFN code at the pinned commit (research-only, no git).",
    )
    tabclustpfn_checkpoint: Path = Field(
        default=PROJECT_ROOT.parent
        / "third_party_src"
        / "tabclustpfn_ckpt"
        / "step-10000.ckpt",
        description="TabClustPFN weights; SHA-256 pinned in selection/lucos.py.",
    )


settings = Settings()

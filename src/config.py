"""Typed access to params.yaml plus repo-relative path helpers.

Every module reads configuration through :func:`load_config` so training, serving
and tests share one definition of features, splits and the target.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = REPO_ROOT / "params.yaml"

load_dotenv(REPO_ROOT / ".env")


DateRange = tuple[date, date]


class SplitConfig(BaseModel):
    date_column: str
    train: DateRange
    valid: DateRange
    test: DateRange

    @field_validator("train", "valid", "test", mode="before")
    @classmethod
    def _parse_dates(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return tuple(v if isinstance(v, date) else date.fromisoformat(str(v)) for v in value)
        return value


class DataConfig(BaseModel):
    raw_path: str
    interim_path: str
    processed_dir: str
    subsample_rows: int | None = None


class TargetConfig(BaseModel):
    source_column: str
    positive_labels: list[str]
    negative_labels: list[str]

    @property
    def keep_labels(self) -> set[str]:
        return set(self.positive_labels) | set(self.negative_labels)


class FeatureConfig(BaseModel):
    numeric: list[str]
    categorical: list[str]
    baseline: list[str]

    @property
    def all_model_columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]


class EvaluationConfig(BaseModel):
    precision_target: float = 0.5
    psi_threshold_retrain: float = 0.20


class MlflowConfig(BaseModel):
    tracking_uri: str
    experiment: str
    registered_model: str


class Config(BaseModel):
    seed: int = 42
    data: DataConfig
    split: SplitConfig
    target: TargetConfig
    features: FeatureConfig
    model: dict = Field(default_factory=dict)
    evaluation: EvaluationConfig = EvaluationConfig()
    mlflow: MlflowConfig

    # --- path helpers -----------------------------------------------------
    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def raw_path(self) -> Path:
        return self.path(self.data.raw_path)

    @property
    def interim_path(self) -> Path:
        return self.path(self.data.interim_path)

    @property
    def processed_dir(self) -> Path:
        return self.path(self.data.processed_dir)

    @property
    def lgbm_params(self) -> dict:
        return dict(self.model.get("lgbm", {}))


@lru_cache(maxsize=4)
def load_config(params_path: str | Path = DEFAULT_PARAMS) -> Config:
    """Load and validate params.yaml. Cached per path."""
    with open(params_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)

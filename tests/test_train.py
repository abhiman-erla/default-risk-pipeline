"""Training smoke test: trains on a small synthetic sample, asserts a metric floor.

Runs against a temp file-based MLflow store so it needs no server.
"""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from src.config import Config
from src.evaluate import evaluate_predictions
from src.features import prepare, split_xy, temporal_split
from src.train import build_estimator, train_model
from src.validate import validate_model_frame

PR_AUC_FLOOR = 0.35  # base rate ~0.2; anything that learns clears this


@pytest.fixture()
def _local_mlflow(tmp_path, monkeypatch: pytest.MonkeyPatch, cfg: Config) -> None:
    uri = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setattr(cfg.mlflow, "tracking_uri", uri)
    mlflow.set_tracking_uri(uri)


def _frames(raw_frame: pd.DataFrame, cfg: Config):
    frame = validate_model_frame(prepare(raw_frame, cfg), cfg)
    return temporal_split(frame, cfg)


def test_estimator_beats_floor_baseline(raw_frame: pd.DataFrame, cfg: Config) -> None:
    train, valid, _ = _frames(raw_frame, cfg)
    cols = cfg.features.baseline
    x_tr, y_tr = split_xy(train, cfg, cols)
    x_va, y_va = split_xy(valid, cfg, cols)

    est = build_estimator("baseline", cfg, cols).fit(x_tr, y_tr)
    m = evaluate_predictions(y_va, est.predict_proba(x_va)[:, 1])
    assert m.pr_auc > PR_AUC_FLOOR


def test_both_models_clear_floor_and_agree(raw_frame: pd.DataFrame, cfg: Config) -> None:
    # On real Lending Club data LightGBM lifts PR-AUC over the linear baseline;
    # on this synthetic frame we only assert both learn real signal and land
    # in the same ballpark (the true DGP here is mostly linear).
    train, valid, _ = _frames(raw_frame, cfg)

    def score(name: str, cols: list[str]) -> float:
        x_tr, y_tr = split_xy(train, cfg, cols)
        x_va, y_va = split_xy(valid, cfg, cols)
        est = build_estimator(name, cfg, cols).fit(x_tr, y_tr)
        return evaluate_predictions(y_va, est.predict_proba(x_va)[:, 1]).pr_auc

    base = score("baseline", cfg.features.baseline)
    lgbm = score("lgbm", cfg.features.all_model_columns)
    assert base > PR_AUC_FLOOR
    assert lgbm > PR_AUC_FLOOR
    assert abs(lgbm - base) < 0.15


def test_reproducible(raw_frame: pd.DataFrame, cfg: Config) -> None:
    train, valid, _ = _frames(raw_frame, cfg)
    cols = cfg.features.all_model_columns
    x_tr, y_tr = split_xy(train, cfg, cols)
    x_va, _ = split_xy(valid, cfg, cols)

    p1 = build_estimator("lgbm", cfg, cols).fit(x_tr, y_tr).predict_proba(x_va)[:, 1]
    p2 = build_estimator("lgbm", cfg, cols).fit(x_tr, y_tr).predict_proba(x_va)[:, 1]
    assert (p1 == p2).all()


@pytest.mark.usefixtures("_local_mlflow")
def test_train_model_logs_run(raw_frame, cfg: Config, tmp_path, monkeypatch) -> None:
    parquet = tmp_path / "loans.parquet"
    raw_frame.to_parquet(parquet, index=False)
    monkeypatch.setattr(type(cfg), "interim_path", property(lambda self: parquet))

    metrics = train_model("baseline", cfg, sample=5000)
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["pr_auc"] > PR_AUC_FLOOR

    runs = mlflow.search_runs(experiment_names=[cfg.mlflow.experiment])
    assert len(runs) >= 1
    assert "metrics.pr_auc" in runs.columns

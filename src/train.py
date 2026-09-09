"""Fit the baseline and LightGBM models; log everything to MLflow.

Each call is one MLflow run: params, metrics (from src.evaluate), the fitted
sklearn pipeline as the model artifact, and PR / calibration plots. Randomness
is pinned so two runs of the same command reproduce the reported numbers.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config, load_config
from src.evaluate import calibration_figure, evaluate_predictions, pr_curve_figure
from src.features import prepare, split_xy, temporal_split
from src.validate import validate_model_frame

log = logging.getLogger("train")

ModelName = Literal["baseline", "lgbm"]


# --------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------
def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_estimator(name: ModelName, cfg: Config, feature_cols: list[str]) -> Pipeline:
    numeric = [c for c in cfg.features.numeric if c in feature_cols]
    categorical = [c for c in cfg.features.categorical if c in feature_cols]
    pre = _preprocessor(numeric, categorical)

    if name == "baseline":
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=cfg.seed
        )
    else:
        from lightgbm import LGBMClassifier

        params = {**cfg.lgbm_params, "random_state": cfg.seed, "verbosity": -1}
        clf = LGBMClassifier(**params)

    return Pipeline([("pre", pre), ("clf", clf)])


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_frames(cfg: Config, sample: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(cfg.interim_path)
    if sample is not None and sample < len(raw):
        raw = raw.sample(n=sample, random_state=cfg.seed).reset_index(drop=True)
    frame = validate_model_frame(prepare(raw, cfg), cfg)
    train, valid, test = temporal_split(frame, cfg)
    log.info("frames -> train=%s valid=%s test=%s", len(train), len(valid), len(test))
    if min(len(train), len(valid)) == 0:
        raise ValueError("empty train or valid split; check split windows vs. issue_d range")
    return train, valid, test


# --------------------------------------------------------------------------
# Train + log
# --------------------------------------------------------------------------
def train_model(
    name: ModelName,
    cfg: Config | None = None,
    *,
    sample: int | None = None,
    register: bool = False,
) -> dict[str, float]:
    cfg = cfg or load_config()

    feature_cols = cfg.features.baseline if name == "baseline" else cfg.features.all_model_columns
    train, valid, _ = load_frames(cfg, sample)
    x_train, y_train = split_xy(train, cfg, feature_cols)
    x_valid, y_valid = split_xy(valid, cfg, feature_cols)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run(run_name=name) as run:
        est = build_estimator(name, cfg, feature_cols)
        est.fit(x_train, y_train)

        p_valid = est.predict_proba(x_valid)[:, 1]
        metrics = evaluate_predictions(
            y_valid,
            p_valid,
            loan_amount=x_valid.get("loan_amnt"),
            precision_target=cfg.evaluation.precision_target,
        )

        mlflow.set_tags({"model": name, "split": "temporal", "stage": "validation"})
        mlflow.log_params(
            {
                "model": name,
                "seed": cfg.seed,
                "n_features": len(feature_cols),
                "n_train": len(x_train),
                "n_valid": len(x_valid),
                "sample": sample or "full",
                **({f"lgbm__{k}": v for k, v in cfg.lgbm_params.items()} if name == "lgbm" else {}),
            }
        )
        mlflow.log_metrics(metrics.as_dict())

        _log_figures(y_valid.to_numpy(), p_valid)

        signature = infer_signature(x_valid, p_valid)
        model_info = mlflow.sklearn.log_model(
            est,
            artifact_path="model",
            signature=signature,
            input_example=x_valid.head(3),
            registered_model_name=cfg.mlflow.registered_model if register else None,
        )
        log.info("run %s  pr_auc=%.4f  brier=%.4f", run.info.run_id, metrics.pr_auc, metrics.brier)
        if register:
            log.info("registered %s -> %s", cfg.mlflow.registered_model, model_info.model_uri)

    return metrics.as_dict()


def _log_figures(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmp:
        for fname, fig in {
            "pr_curve.png": pr_curve_figure(y_true, y_prob),
            "calibration.png": calibration_figure(y_true, y_prob),
        }.items():
            path = Path(tmp) / fname
            fig.savefig(path, dpi=120)
            plt.close(fig)
            mlflow.log_artifact(str(path), artifact_path="plots")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a default-risk model and log to MLflow.")
    parser.add_argument("--model", choices=["baseline", "lgbm", "both"], default="both")
    parser.add_argument("--sample", type=int, default=None, help="row cap before splitting")
    parser.add_argument("--register", action="store_true", help="register the run in MLflow")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    names: list[ModelName] = ["baseline", "lgbm"] if args.model == "both" else [args.model]
    results = {n: train_model(n, sample=args.sample, register=args.register) for n in names}

    print("\nvalidation PR-AUC")
    for n, m in results.items():
        print(
            f"  {n:9s} {m['pr_auc']:.4f}  (recall@p={m['precision_target']:.2f}: "
            f"{m['recall_at_precision']:.3f}, brier {m['brier']:.4f})"
        )
    if {"baseline", "lgbm"} <= results.keys():
        lift = results["lgbm"]["pr_auc"] - results["baseline"]["pr_auc"]
        print(f"  lgbm lift over baseline: {lift:+.4f}")


if __name__ == "__main__":
    main()

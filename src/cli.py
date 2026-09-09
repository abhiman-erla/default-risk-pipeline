"""`drp` command-line entrypoint -- thin wrapper over the src modules."""

from __future__ import annotations

import logging

import typer

app = typer.Typer(add_completion=False, help="Default-risk pipeline CLI.")


@app.callback()
def _setup(verbose: bool = typer.Option(True, help="INFO logging")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )


@app.command()
def ingest() -> None:
    """Pull raw data, sanity-check, subsample, write interim parquet."""
    from src.ingest import ingest as _ingest

    _ingest()


@app.command()
def validate() -> None:
    """Run the pandera schema + leakage audit over the prepared frame."""
    import pandas as pd

    from src.config import load_config
    from src.features import prepare
    from src.validate import validate_model_frame

    cfg = load_config()
    frame = validate_model_frame(prepare(pd.read_parquet(cfg.interim_path), cfg), cfg)
    typer.echo(f"OK: {len(frame):,} rows validated")


@app.command()
def train(
    model: str = typer.Option("both", help="baseline | lgbm | both"),
    sample: int = typer.Option(None, help="row cap before splitting"),
    register: bool = typer.Option(False, help="register the run in MLflow"),
) -> None:
    """Fit model(s) and log an MLflow run."""
    from src.train import ModelName, train_model

    names = ["baseline", "lgbm"] if model == "both" else [model]
    for name in names:
        m = train_model(name, sample=sample, register=register)  # type: ignore[arg-type]
        typer.echo(f"{name}: pr_auc={m['pr_auc']:.4f} brier={m['brier']:.4f}")
    _ = ModelName  # keep import referenced for readers


@app.command()
def promote(version: str, stage: str) -> None:
    """Transition a registered model version to Staging/Production/Archived."""
    from src.register import promote as _promote

    _promote(version, stage)


@app.command()
def rollback() -> None:
    """Re-point Production at the previous (Archived) version."""
    from src.register import rollback as _rollback

    typer.echo(f"Production is now v{_rollback()}")


if __name__ == "__main__":
    app()

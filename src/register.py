"""Promote a model version through the registry lifecycle.

Staging -> Production -> Archived, as the spec's phases 5/6 expect. Pinned to
MLflow 2.x so classic stages are available; the equivalent alias-based flow is
noted in the README for a 3.x upgrade.
"""

from __future__ import annotations

import argparse
import logging

from mlflow.tracking import MlflowClient

from src.config import Config, load_config

log = logging.getLogger("register")

Stage = str  # "Staging" | "Production" | "Archived"


def _client(cfg: Config) -> MlflowClient:
    return MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)


def latest_version(cfg: Config, stage: Stage | None = None) -> str | None:
    client = _client(cfg)
    stages = [stage] if stage else None
    versions = client.get_latest_versions(cfg.mlflow.registered_model, stages=stages)
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version)).version


def promote(
    version: str,
    stage: Stage,
    cfg: Config | None = None,
    *,
    archive_existing: bool = True,
) -> None:
    cfg = cfg or load_config()
    client = _client(cfg)
    name = cfg.mlflow.registered_model
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing and stage == "Production",
    )
    run_id = client.get_model_version(name, version).run_id
    client.set_model_version_tag(name, version, "promoted_from_run", run_id or "")
    log.info("%s v%s -> %s", name, version, stage)


def rollback(cfg: Config | None = None) -> str:
    """Re-point Production at the most recent Archived version. One step.

    The current Production version is archived in the same transition, so there
    is always exactly one Production version.
    """
    cfg = cfg or load_config()
    client = _client(cfg)
    name = cfg.mlflow.registered_model
    archived = client.get_latest_versions(name, stages=["Archived"])
    if not archived:
        raise RuntimeError("no archived version to roll back to")
    current = client.get_latest_versions(name, stages=["Production"])
    target = max(archived, key=lambda v: int(v.version)).version
    if current and current[0].version == target:
        raise RuntimeError(f"v{target} is already Production; nothing to roll back")
    promote(target, "Production", cfg, archive_existing=True)
    log.info("rolled back: %s v%s is now Production (was v%s)",
             name, target, current[0].version if current else "none")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote / roll back a registered model version.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("version")
    p_promote.add_argument("stage", choices=["Staging", "Production", "Archived"])

    sub.add_parser("rollback")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.cmd == "promote":
        promote(args.version, args.stage)
    else:
        rollback()


if __name__ == "__main__":
    main()

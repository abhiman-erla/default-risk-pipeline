.DEFAULT_GOAL := help
.PHONY: help setup up down logs ingest validate train train-sample test lint fmt type check clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## create the venv and install deps
	uv sync --dev

up:  ## start MLflow + Postgres
	docker compose up -d --build

down:  ## stop the stack
	docker compose down

logs:  ## tail the MLflow server
	docker compose logs -f mlflow

ingest:  ## raw -> interim parquet
	uv run python -m src.ingest

validate:  ## schema + leakage audit
	uv run python -m src.validate

train:  ## fit baseline + lgbm, log to MLflow
	uv run python -m src.train --model both

train-sample:  ## quick 20k-row run
	uv run python -m src.train --model both --sample 20000

test:  ## pytest
	uv run pytest

lint:  ## ruff check
	uv run ruff check .

fmt:  ## ruff format
	uv run ruff format .

type:  ## mypy
	uv run mypy src

check: lint type test  ## everything CI runs

clean:  ## drop local caches and interim data
	rm -rf .pytest_cache .mypy_cache .ruff_cache data/interim/*.parquet

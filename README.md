# Default Risk Pipeline

Predict which funded Lending Club loans will charge off, and wrap that model in
the machinery a team expects: versioned data, tracked experiments, a model
registry, a served API, drift monitoring, and CI-gated retraining.

Status: **Phase 1 — baseline model + experiment tracking.** (Phase 0 foundations
— leakage audit, temporal split, feature transforms — are folded in here since
Phase 1 stands on them.)

---

## Get the data

The dataset is not in the repo. Download it from Kaggle (free account required):

- Dataset: **Lending Club** by *wordsforthewise* — https://www.kaggle.com/datasets/wordsforthewise/lending-club
- File you need: `accepted_2007_to_2018Q4.csv.gz` (~400 MB gzipped, ~2.26 M rows). You do **not** need `rejected_*`.

**Web:** open the dataset page → *Download* (you get a zip of both files) → extract
`accepted_2007_to_2018Q4.csv.gz` and keep it gzipped.

**CLI:**
```bash
pip install kaggle                       # then put kaggle.json in ~/.kaggle/
kaggle datasets download -d wordsforthewise/lending-club \
  -f accepted_2007_to_2018Q4.csv.gz -p data/raw/
```

Final location (gzipped, unchanged):
```
data/raw/accepted_2007_to_2018Q4.csv.gz
```
`src/ingest.py` reads the `.gz` directly.

---

## Setup

```bash
uv sync --dev            # create .venv, install everything
cp .env.example .env     # MLFLOW_TRACKING_URI=http://localhost:5000
docker compose up -d --build   # MLflow server + Postgres
```
MLflow UI: http://localhost:5000

---

## Run Phase 1

```bash
uv run python -m src.ingest                 # raw .gz -> data/interim/loans.parquet (subsampled to 300k)
uv run python -m src.validate               # pandera schema + leakage audit
uv run python -m src.train --model both     # logistic baseline + LightGBM, one MLflow run each
```
Or `make ingest validate train`. Quick iteration: `make train-sample` (20k rows).

Each run logs params, metrics, a PR curve, a calibration curve, and the fitted
sklearn pipeline as the model artifact. Compare `baseline` vs `lgbm` in the
MLflow UI; the training command also prints the LightGBM lift over the baseline.

Register the current run and promote it:
```bash
uv run python -m src.train --model lgbm --register
uv run python -m src.register promote <version> Production
uv run python -m src.register rollback        # archive current Production, restore the previous
```

---

## Results

300k subsample, temporal split — train 148,218 (2015–2016) · validate 20,540
(2017 H1) · test 29,995 held back (2017 H2–2018). Seed 42.

| Model | PR-AUC (val) | Recall @ precision 0.50 | Brier | ROC-AUC |
|---|---|---|---|---|
| Logistic baseline (16 features) | 0.389 | 0.089 | 0.232 | 0.696 |
| LightGBM (defaults, 25 features) | **0.416** | **0.212** | **0.162** | 0.716 |
| LightGBM lift | +0.027 | +0.123 (2.4×) | −0.070 | +0.020 |

- PR-AUC lands in the expected **0.30–0.45** band for a leakage-free feature
  set. A score near 0.9 would mean a post-origination column got through —
  `assert_no_leakage` + `tests/test_features.py` guard that.
- LightGBM's real gain is at the operating point: **recall at precision 0.50
  more than doubles** (0.089 → 0.212).
- Both models are still **poorly calibrated** (baseline Brier 0.232 — the
  `class_weight="balanced"` baseline over-predicts against a ~20% base rate).
  Calibration is a Phase 2 task; the expected-loss numbers only mean something
  after it.

---

## What Phase 1 gives you

| Piece | Where |
|---|---|
| Leakage drop list + assertion (tested) | `src/features.py` (`LEAKAGE_COLUMNS`, `assert_no_leakage`) |
| Temporal split by `issue_d`, pure function | `src/features.py` (`temporal_split`) |
| Deterministic feature transforms | `src/features.py` (`prepare`) |
| Schema + leakage contract | `src/validate.py` |
| Baseline (logistic) + LightGBM, MLflow-tracked | `src/train.py` |
| PR-AUC, recall@precision, Brier, KS, expected-loss-avoided | `src/evaluate.py` |
| Registry promotion / one-step rollback | `src/register.py` |
| MLflow server + Postgres | `docker-compose.yml`, `docker/mlflow/` |
| ruff + mypy + pytest in CI | `.github/workflows/ci.yml` |

### Metrics (`src/evaluate.py`)
- **PR-AUC / average precision** — primary, given ~20% positives.
- **Recall at precision 0.5** — the lender operating point (configurable in `params.yaml`).
- **Brier score** + reliability curve — calibration; the expected-loss number depends on it.
- **KS** — score separation.
- **Expected loss avoided** — Σ P(default)·loan_amnt over loans declined at the chosen threshold.

Expect PR-AUC in the **0.30–0.45** band on a clean feature set. If your first
model scores ~0.9, a post-origination column leaked in — `assert_no_leakage`
and `tests/test_features.py` exist to stop exactly that.

---

## Tests

```bash
uv run pytest
```
`tests/` runs against a synthetic Lending-Club-shaped frame (`tests/conftest.py`),
so CI needs no dataset. Covered: leakage columns absent from the model matrix,
transform correctness on known inputs, temporal split is disjoint and ordered,
training clears a PR-AUC floor, and two runs reproduce bit-for-bit.

---

## Notes

- **MLflow is pinned to 2.x** so the classic `Staging → Production → Archived`
  stage lifecycle works (phases 5–6 lean on it). On a 3.x upgrade, swap
  `transition_model_version_stage` for version aliases (`@staging`, `@production`).
- **Config** lives in `params.yaml`, read through `src/config.py`. Feature lists,
  split windows, the target definition, and model hyperparameters are all there.
  `MLFLOW_TRACKING_URI` (env / `.env`) overrides `params.yaml` — e.g.
  `MLFLOW_TRACKING_URI=file:./mlruns` to run without the server.
- **Windows:** `src/config.py` forces UTF-8 on stdout/stderr so the MLflow
  client's run-URL print (it uses an emoji) doesn't crash on the cp1252 console.
- `dvc.yaml` wires `ingest → train` so `dvc repro` rebuilds from raw with no
  manual steps once you `uv sync --extra pipeline && dvc init`.

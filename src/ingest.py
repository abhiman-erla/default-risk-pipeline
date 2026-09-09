"""Pull raw Lending Club data, sanity-check it, subsample, write parquet.

Reads the gzipped Kaggle CSV directly (no need to decompress on disk). Keeps
only columns the pipeline knows how to use, so the interim parquet stays small.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import Config, load_config
from src.features import LEAKAGE_COLUMNS, is_leakage_column

log = logging.getLogger("ingest")

# Raw columns we read from the CSV: model inputs + fields needed to derive
# features + the target + a few kept only for the leakage-audit test.
_RAW_KEEP = [
    "loan_status",
    "issue_d",
    "term",
    "emp_length",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "addr_state",
    # a handful of post-origination columns, deliberately carried into the
    # interim file so the leakage audit has something to catch
    "total_pymnt",
    "recoveries",
    "out_prncp",
    "last_pymnt_d",
]

MIN_ROWS = 100_000


def _sanity_checks(df: pd.DataFrame) -> None:
    if len(df) < MIN_ROWS:
        raise ValueError(f"raw data has only {len(df):,} rows (< {MIN_ROWS:,}); wrong file?")
    if df["loan_status"].isna().mean() > 0.01:
        raise ValueError("more than 1% of rows have a null loan_status")
    null_share = df.isna().mean()
    noisy = null_share[null_share > 0.6]
    if not noisy.empty:
        log.warning("columns >60%% null: %s", noisy.round(2).to_dict())


def ingest(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    src = cfg.raw_path
    if not src.exists():
        raise FileNotFoundError(
            f"raw file not found at {src}. See README 'Get the data'. "
            "Expected the gzipped Kaggle CSV accepted_2007_to_2018Q4.csv.gz."
        )

    log.info("reading %s", src)
    available = pd.read_csv(src, nrows=0).columns
    usecols = [c for c in _RAW_KEEP if c in available]
    df = pd.read_csv(src, usecols=usecols, low_memory=False)
    log.info("read %s rows x %s cols", f"{len(df):,}", df.shape[1])

    _sanity_checks(df)

    # deterministic subsample of outcome-known rows
    known = df["loan_status"].isin(cfg.target.positive_labels + cfg.target.negative_labels)
    df = df.loc[known].reset_index(drop=True)
    log.info("outcome-known rows: %s", f"{len(df):,}")

    n = cfg.data.subsample_rows
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=cfg.seed).reset_index(drop=True)
        log.info("subsampled to %s rows (seed=%d)", f"{len(df):,}", cfg.seed)

    leak_present = sorted(c for c in df.columns if is_leakage_column(c))
    log.info(
        "carrying %d post-origination columns for the leakage audit: %s",
        len(leak_present),
        leak_present,
    )
    assert LEAKAGE_COLUMNS  # keep import meaningful for readers

    cfg.interim_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.interim_path, index=False)
    log.info("wrote %s", cfg.interim_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ingest()

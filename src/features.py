"""Deterministic feature transforms and the temporal split.

The same functions run in training (`src/train.py`) and, later, in serving.
No fitting happens here beyond parsing strings and arithmetic; anything that
must be learned from the training set (scaling, encoding) lives in the sklearn
pipeline built in `src/train.py`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config import Config

# --------------------------------------------------------------------------
# Leakage audit
# --------------------------------------------------------------------------
# Fields populated *after* the loan is originated. They encode repayment
# behaviour and therefore the target. Dropped before anything else; the
# assertion below is exercised by tests/test_features.py.
LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "out_prncp",
        "out_prncp_inv",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
        "pymnt_plan",
        "hardship_flag",
        "deferral_term",
        "payment_plan_start_date",
    }
)

# Any column beginning with one of these prefixes is also post-origination.
LEAKAGE_PREFIXES: tuple[str, ...] = ("hardship_", "settlement_", "sec_app_")


def is_leakage_column(name: str) -> bool:
    return name in LEAKAGE_COLUMNS or name.startswith(LEAKAGE_PREFIXES)


def assert_no_leakage(columns: Iterable[object]) -> None:
    """Raise if any post-origination column reached the model matrix."""
    offenders = sorted(str(c) for c in columns if is_leakage_column(str(c)))
    if offenders:
        raise AssertionError(f"leakage columns present in model matrix: {offenders}")


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
def build_target(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filter to outcome-known rows and attach an integer ``target`` column.

    Charged Off / Default -> 1, Fully Paid -> 0. Current / Late / In Grace
    Period rows are dropped (outcome not yet known).
    """
    col = cfg.target.source_column
    status = df[col].astype("string").str.strip()
    keep = status.isin(cfg.target.keep_labels)
    out = df.loc[keep].copy()
    out["target"] = status.loc[keep].isin(cfg.target.positive_labels).astype("int8")
    return out


# --------------------------------------------------------------------------
# Derived features
# --------------------------------------------------------------------------
_EMP_LENGTH_RE = re.compile(r"(\d+)")


def _parse_emp_length(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "n/a", "na", "none"}:
        return np.nan
    if text.startswith("< 1"):
        return 0.0
    match = _EMP_LENGTH_RE.search(text)
    return float(match.group(1)) if match else np.nan


def _parse_term_months(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    match = _EMP_LENGTH_RE.search(str(value))
    return float(match.group(1)) if match else np.nan


def _to_month_start(series: pd.Series) -> pd.Series:
    """Lending Club dates look like ``Dec-2015``. Coerce to a timestamp."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series.astype("string").str.strip(), format="%b-%Y", errors="coerce")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fico, credit_history_years, emp_length_years, term_months, issue_d(dt)."""
    out = df.copy()

    issue_dt = _to_month_start(out["issue_d"])
    out["issue_d"] = issue_dt

    if {"fico_range_low", "fico_range_high"}.issubset(out.columns):
        out["fico"] = (
            out["fico_range_low"].astype("float64") + out["fico_range_high"].astype("float64")
        ) / 2.0

    if "earliest_cr_line" in out.columns:
        earliest = _to_month_start(out["earliest_cr_line"])
        out["credit_history_years"] = (issue_dt - earliest).dt.days / 365.25

    if "emp_length" in out.columns:
        out["emp_length_years"] = out["emp_length"].map(_parse_emp_length).astype("float64")

    if "term" in out.columns:
        out["term_months"] = out["term"].map(_parse_term_months).astype("float64")

    return out


# --------------------------------------------------------------------------
# Model frame + split
# --------------------------------------------------------------------------
def select_model_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep feature columns + target + issue_d, and prove no leakage sneaked in."""
    wanted = [*cfg.features.all_model_columns, "target", cfg.split.date_column]
    present = [c for c in wanted if c in df.columns]
    missing = sorted(set(wanted) - set(present))
    if missing:
        raise KeyError(f"expected columns absent after feature build: {missing}")
    frame = df.loc[:, present].copy()
    assert_no_leakage(frame.columns)
    return frame


def prepare(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Raw Lending Club rows -> validated model frame. Pure and deterministic."""
    df = df.drop(columns=[c for c in df.columns if is_leakage_column(str(c))], errors="ignore")
    df = build_target(df, cfg)
    df = add_derived_features(df)
    return select_model_frame(df, cfg)


def temporal_split(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition rows into train / valid / test by ``issue_d`` vintage.

    Pure function: no shuffling, no leakage across time. Rows outside every
    window (or with an unparseable date) are discarded.
    """
    col = cfg.split.date_column
    dates = _to_month_start(df[col])

    def window(bounds: tuple) -> pd.DataFrame:
        lo, hi = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        mask = dates.between(lo, hi)
        return df.loc[mask].copy()

    return window(cfg.split.train), window(cfg.split.valid), window(cfg.split.test)


def split_xy(
    frame: pd.DataFrame, cfg: Config, columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the given feature list (defaults to the full set)."""
    cols = columns or cfg.features.all_model_columns
    return frame.loc[:, cols].copy(), frame["target"].astype("int8")

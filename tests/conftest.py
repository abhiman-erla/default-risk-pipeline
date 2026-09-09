"""Synthetic Lending-Club-shaped data for tests.

Not a general data generator -- just enough columns, in the raw CSV's names and
formats, to exercise the transforms, the leakage audit and a real training run.
Signal is injected (int_rate up, fico down, dti up => higher default odds) so
the model can clear a metric floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.config import Config, load_config

RAW_LEAKAGE_COLS = ["total_pymnt", "recoveries", "out_prncp", "last_pymnt_d"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_raw_frame(n: int = 12_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    months = pd.date_range("2015-01-01", "2018-12-01", freq="MS")
    issue = rng.choice(months, size=n)
    issue_d = pd.Series(issue).dt.strftime("%b-%Y")

    grade = rng.choice(list("ABCDEFG"), size=n, p=[0.18, 0.28, 0.24, 0.15, 0.09, 0.04, 0.02])
    sub_grade = np.char.add(grade.astype(str), rng.integers(1, 6, n).astype(str))
    int_rate = np.round(
        5 + (np.searchsorted(list("ABCDEFG"), grade) * 3.5) + rng.normal(0, 1.5, n), 2
    ).clip(5, 31)
    loan_amnt = np.round(rng.uniform(1000, 40000, n), -2)
    term = rng.choice([" 36 months", " 60 months"], size=n, p=[0.72, 0.28])
    term_m = np.where(term == " 36 months", 36, 60)
    installment = np.round(
        loan_amnt * (int_rate / 1200) / (1 - (1 + int_rate / 1200) ** (-term_m)), 2
    )
    annual_inc = np.round(rng.lognormal(11.0, 0.5, n), -2).clip(8000, 500000)
    dti = np.round(rng.gamma(3.0, 6.0, n), 2).clip(0, 60)
    fico_low = rng.integers(660, 840, n)
    fico = fico_low + 4

    df = pd.DataFrame(
        {
            "issue_d": issue_d,
            "term": term,
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_length": rng.choice(
                ["< 1 year", "1 year", "2 years", "5 years", "10+ years", "n/a"], size=n
            ),
            "earliest_cr_line": pd.Series(
                rng.choice(pd.date_range("1990-01-01", "2012-01-01", freq="MS"), size=n)
            ).dt.strftime("%b-%Y"),
            "fico_range_low": fico_low,
            "fico_range_high": fico_low + 4,
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "installment": installment,
            "annual_inc": annual_inc,
            "dti": dti,
            "delinq_2yrs": rng.poisson(0.3, n),
            "inq_last_6mths": rng.poisson(0.8, n),
            "open_acc": rng.integers(2, 25, n),
            "pub_rec": rng.poisson(0.2, n),
            "revol_bal": np.round(rng.uniform(0, 60000, n), -1),
            "revol_util": np.round(rng.uniform(0, 100, n), 1),
            "total_acc": rng.integers(5, 60, n),
            "mort_acc": rng.integers(0, 6, n),
            "pub_rec_bankruptcies": rng.poisson(0.1, n),
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], size=n, p=[0.4, 0.1, 0.5]),
            "verification_status": rng.choice(
                ["Verified", "Source Verified", "Not Verified"], size=n
            ),
            "purpose": rng.choice(
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "major_purchase",
                    "other",
                ],
                size=n,
                p=[0.55, 0.22, 0.1, 0.06, 0.07],
            ),
            "application_type": rng.choice(["Individual", "Joint App"], size=n, p=[0.94, 0.06]),
            "addr_state": rng.choice(["CA", "TX", "NY", "FL", "IL", "PA"], size=n),
        }
    )

    # ---- target with injected signal ----------------------------------
    # Linear terms + a threshold effect on DTI + a rate x 60-month interaction,
    # so both a linear model and a tree model have something real to fit.
    z_rate = (int_rate - int_rate.mean()) / int_rate.std()
    z_fico = (fico - fico.mean()) / fico.std()
    z_dti = (dti - dti.mean()) / dti.std()
    long_term = (term == " 60 months").astype(float)
    logit = (
        -1.5
        + 1.1 * z_rate
        - 0.7 * z_fico
        + 0.4 * z_dti
        + 0.8 * (dti > 30).astype(float)
        + 0.6 * z_rate * long_term
        + rng.normal(0, 0.8, n)
    )
    default = rng.random(n) < _sigmoid(logit)
    status = np.where(default, "Charged Off", "Fully Paid").astype(object)
    # a few outcome-unknown rows that must be filtered out
    unknown_idx = rng.choice(n, size=n // 20, replace=False)
    status[unknown_idx] = "Current"
    df["loan_status"] = status

    # ---- post-origination columns (leakage; must be dropped) ----------
    df["total_pymnt"] = np.where(default, loan_amnt * 0.4, loan_amnt * 1.15).round(2)
    df["recoveries"] = np.where(default, rng.uniform(0, 3000, n), 0.0).round(2)
    df["out_prncp"] = np.where(default, 0.0, rng.uniform(0, loan_amnt)).round(2)
    df["last_pymnt_d"] = pd.Series(
        rng.choice(pd.date_range("2016-01-01", "2019-06-01", freq="MS"), size=n)
    ).dt.strftime("%b-%Y")

    return df


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture()
def interim_parquet(tmp_path_factory: pytest.TempPathFactory, raw_frame: pd.DataFrame) -> str:
    path = tmp_path_factory.mktemp("data") / "loans.parquet"
    raw_frame.to_parquet(path, index=False)
    return str(path)

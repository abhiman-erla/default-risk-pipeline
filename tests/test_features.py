"""Transform correctness + the leakage guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.config import Config
from src.features import (
    add_derived_features,
    assert_no_leakage,
    build_target,
    is_leakage_column,
    prepare,
    select_model_frame,
    temporal_split,
)

from tests.conftest import RAW_LEAKAGE_COLS


def test_leakage_columns_recognised() -> None:
    for col in RAW_LEAKAGE_COLS:
        assert is_leakage_column(col)
    assert is_leakage_column("hardship_type")
    assert is_leakage_column("settlement_amount")
    assert not is_leakage_column("loan_amnt")
    assert not is_leakage_column("int_rate")


def test_assert_no_leakage_raises() -> None:
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(["loan_amnt", "recoveries"])
    assert assert_no_leakage(["loan_amnt", "int_rate", "fico"]) is None  # clean list: no raise


def test_build_target_filters_and_encodes(raw_frame: pd.DataFrame, cfg: Config) -> None:
    out = build_target(raw_frame, cfg)
    assert set(out["target"].unique()) <= {0, 1}
    assert (out["target"] == 1).sum() > 0
    assert not out["loan_status"].str.contains("Current").any()
    # Charged Off -> 1
    co = out.loc[out["loan_status"] == "Charged Off", "target"]
    assert (co == 1).all()


def test_derived_features_known_values(cfg: Config) -> None:
    df = pd.DataFrame(
        {
            "issue_d": ["Dec-2015"],
            "earliest_cr_line": ["Dec-2005"],
            "fico_range_low": [700],
            "fico_range_high": [704],
            "emp_length": ["10+ years"],
            "term": [" 36 months"],
        }
    )
    out = add_derived_features(df)
    assert out["fico"].iloc[0] == pytest.approx(702.0)
    assert out["term_months"].iloc[0] == 36.0
    assert out["emp_length_years"].iloc[0] == 10.0
    assert out["credit_history_years"].iloc[0] == pytest.approx(10.0, abs=0.1)
    assert out["issue_d"].iloc[0] == pd.Timestamp("2015-12-01")


def test_emp_length_edge_cases(cfg: Config) -> None:
    df = pd.DataFrame(
        {
            "issue_d": ["Jan-2016"] * 3,
            "term": [" 60 months"] * 3,
            "emp_length": ["< 1 year", "n/a", "3 years"],
        }
    )
    out = add_derived_features(df)
    assert out["emp_length_years"].iloc[0] == 0.0
    assert np.isnan(out["emp_length_years"].iloc[1])
    assert out["emp_length_years"].iloc[2] == 3.0
    assert out["term_months"].iloc[0] == 60.0


def test_prepare_produces_leakage_free_frame(raw_frame: pd.DataFrame, cfg: Config) -> None:
    frame = prepare(raw_frame, cfg)
    assert_no_leakage(frame.columns)  # would raise otherwise
    for col in RAW_LEAKAGE_COLS:
        assert col not in frame.columns
    assert "target" in frame.columns
    assert frame[cfg.split.date_column].dtype.kind == "M"


def test_select_model_frame_missing_column_raises(raw_frame: pd.DataFrame, cfg: Config) -> None:
    broken = add_derived_features(build_target(raw_frame, cfg)).drop(columns=["int_rate"])
    with pytest.raises(KeyError, match="int_rate"):
        select_model_frame(broken, cfg)


def test_temporal_split_is_disjoint_and_ordered(raw_frame: pd.DataFrame, cfg: Config) -> None:
    frame = prepare(raw_frame, cfg)
    train, valid, test = temporal_split(frame, cfg)

    assert len(train) and len(valid) and len(test)
    assert train[cfg.split.date_column].max() < valid[cfg.split.date_column].min()
    assert valid[cfg.split.date_column].max() < test[cfg.split.date_column].min()

    # windows come straight from params.yaml
    assert train[cfg.split.date_column].min() >= pd.Timestamp(cfg.split.train[0])
    assert test[cfg.split.date_column].max() <= pd.Timestamp(cfg.split.test[1])

"""Schema + leakage contract for the model frame.

`validate_model_frame` is the single gate every training run and (later) every
serving request passes through. A malformed row raises a typed error here
instead of a 500 or a silent NaN downstream.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.config import Config
from src.features import assert_no_leakage

_HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "ANY", "NONE", "OTHER"]
_VERIFICATION = ["Verified", "Source Verified", "Not Verified"]
_APPLICATION = ["Individual", "Joint App"]


def build_schema(cfg: Config, *, require_target: bool) -> DataFrameSchema:
    """Construct the pandera schema from params.yaml feature lists.

    Numeric features are all nullable: real Lending Club rows carry scattered
    missing credit-bureau values, and the fitted ``SimpleImputer`` in the
    training/serving pipeline is the explicit contract for filling them. The
    schema's job here is dtype + plausible range + allowed categories + the
    leakage guarantee -- not completeness. Range checks skip nulls.
    """
    columns: dict[str, Column] = {}

    for name in cfg.features.numeric:
        num_checks: list[Check] = [Check.ge(-1) if name == "credit_history_years" else Check.ge(0)]
        if name == "int_rate":
            num_checks = [Check.in_range(0, 100)]
        if name == "fico":
            num_checks = [Check.in_range(300, 900)]
        columns[name] = Column(float, checks=num_checks, nullable=True, coerce=True)

    allowed_values = {
        "home_ownership": _HOME_OWNERSHIP,
        "verification_status": _VERIFICATION,
        "application_type": _APPLICATION,
    }
    for name in cfg.features.categorical:
        allowed = allowed_values.get(name)
        cat_checks: list[Check] | None = [Check.isin(allowed)] if allowed else None
        columns[name] = Column(str, checks=cat_checks, nullable=True, coerce=True)

    columns[cfg.split.date_column] = Column("datetime64[ns]", nullable=False, coerce=True)
    if require_target:
        columns["target"] = Column(int, checks=[Check.isin([0, 1])], coerce=True)

    return DataFrameSchema(columns, strict=False, coerce=True)


def validate_model_frame(
    df: pd.DataFrame, cfg: Config, *, require_target: bool = True
) -> pd.DataFrame:
    assert_no_leakage(df.columns)
    schema = build_schema(cfg, require_target=require_target)
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:  # narrow, readable failure
        cases = exc.failure_cases[["column", "check", "failure_case"]].head(20)
        raise ValueError(f"model-frame validation failed:\n{cases.to_string(index=False)}") from exc


if __name__ == "__main__":
    import pandas as pd

    from src.config import load_config
    from src.features import prepare

    cfg = load_config()
    frame = prepare(pd.read_parquet(cfg.interim_path), cfg)
    validate_model_frame(frame, cfg)
    print(f"OK: {len(frame):,} rows validated")

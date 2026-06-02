"""Data contract checks for binary PD validation."""

import polars as pl

from credit_risk_validation.config import ColumnConfig, ValidationOptions
from credit_risk_validation.schemas import CheckResult
from credit_risk_validation.status import Status
from credit_risk_validation.utils.dataframe import optional_float
from credit_risk_validation.utils.validation import ensure_columns_exist


def validate_contract(
    frame: pl.DataFrame,
    columns: ColumnConfig,
    validation: ValidationOptions,
    *,
    name: str = "dataset",
) -> list[CheckResult]:
    """Valida contrato minimo de datos para un dataset."""

    checks: list[CheckResult] = []
    available = set(frame.columns)
    required = [columns.target, columns.pd]
    optional = [
        column
        for column in [columns.score, columns.period, columns.weight, columns.id, *columns.segments]
        if column is not None
    ]
    missing_required = ensure_columns_exist(required, available)
    missing_optional = ensure_columns_exist(optional, available)

    if missing_required:
        checks.append(
            CheckResult(
                f"{name}.required_columns",
                Status.CRITICAL,
                f"Missing required columns: {', '.join(missing_required)}",
                missing_required,
            )
        )
        return checks
    checks.append(CheckResult(f"{name}.required_columns", Status.OK, "Required columns exist"))

    if missing_optional:
        checks.append(
            CheckResult(
                f"{name}.optional_columns",
                Status.WARNING,
                f"Missing optional configured columns: {', '.join(missing_optional)}",
                missing_optional,
            )
        )
    else:
        checks.append(
            CheckResult(f"{name}.optional_columns", Status.OK, "Configured columns exist")
        )

    if len(frame.columns) != len(set(frame.columns)):
        checks.append(
            CheckResult(f"{name}.duplicate_columns", Status.CRITICAL, "Duplicate columns found")
        )
    else:
        checks.append(CheckResult(f"{name}.duplicate_columns", Status.OK, "No duplicate columns"))

    checks.extend(
        [
            _check_size(frame, validation, name),
            _check_target(frame, columns.target, validation, name),
            _check_pd(frame, columns.pd, validation, name),
        ]
    )
    if columns.score and columns.score in available:
        checks.append(_check_constant_numeric(frame, columns.score, name, "score"))

    if columns.weight and columns.weight in available:
        checks.append(_check_weight(frame, columns.weight, name))
    if columns.period and columns.period in available:
        checks.append(_check_period(frame, columns.period, name))
    if columns.id and columns.period and columns.id in available and columns.period in available:
        checks.append(_check_id_period_duplicates(frame, columns.id, columns.period, name))
    for segment in columns.segments:
        if segment in available:
            checks.append(_check_segment_cardinality(frame, segment, name))
    return checks


def _check_size(frame: pl.DataFrame, validation: ValidationOptions, name: str) -> CheckResult:
    if frame.height == 0:
        return CheckResult(f"{name}.empty", Status.CRITICAL, "Dataset is empty", frame.height)
    if frame.height < validation.min_rows:
        return CheckResult(
            f"{name}.min_rows",
            Status.WARNING,
            f"Dataset has {frame.height} rows; configured minimum is {validation.min_rows}",
            frame.height,
        )
    return CheckResult(f"{name}.min_rows", Status.OK, "Dataset size is sufficient", frame.height)


def _check_target(
    frame: pl.DataFrame, target_col: str, validation: ValidationOptions, name: str
) -> CheckResult:
    target = frame[target_col]
    if target.null_count() > 0:
        return CheckResult(f"{name}.target_nulls", Status.CRITICAL, "Target contains nulls")
    unique = set(target.unique().to_list())
    expected = {0, validation.positive_class}
    if not unique.issubset(expected):
        return CheckResult(
            f"{name}.target_binary",
            Status.CRITICAL,
            f"Target must be binary with values {sorted(expected)}",
            sorted(unique),
        )
    if len(unique) < 2:
        return CheckResult(
            f"{name}.target_events",
            Status.INSUFFICIENT_DATA,
            "Target must contain at least one event and one non-event",
            sorted(unique),
        )
    events = int((target == validation.positive_class).sum())
    non_events = frame.height - events
    if events < validation.min_events or non_events < validation.min_non_events:
        return CheckResult(
            f"{name}.target_events",
            Status.INSUFFICIENT_DATA,
            "Insufficient events or non-events",
            {"events": events, "non_events": non_events},
        )
    event_rate = events / frame.height if frame.height else 0.0
    if event_rate < 0.01 or event_rate > 0.99:
        return CheckResult(
            f"{name}.target_imbalance",
            Status.WARNING,
            "Target is highly imbalanced",
            {"events": events, "non_events": non_events, "event_rate": event_rate},
        )
    return CheckResult(
        f"{name}.target_binary",
        Status.OK,
        "Target is binary with sufficient events and non-events",
        {"events": events, "non_events": non_events},
    )


def _check_pd(
    frame: pl.DataFrame, pd_col: str, validation: ValidationOptions, name: str
) -> CheckResult:
    series = frame.select(pl.col(pd_col).cast(pl.Float64, strict=False)).to_series()
    if series.null_count() > 0:
        return CheckResult(
            f"{name}.pd_numeric", Status.CRITICAL, "PD contains nulls or non-numeric values"
        )
    if series.is_infinite().sum() > 0:
        return CheckResult(f"{name}.pd_infinite", Status.CRITICAL, "PD contains infinite values")
    min_pd = optional_float(series.min())
    max_pd = optional_float(series.max())
    if min_pd is None or max_pd is None:
        return CheckResult(
            f"{name}.pd_numeric", Status.CRITICAL, "PD min/max could not be computed"
        )
    if validation.clip_pd.enabled:
        if min_pd < 0 or max_pd > 1:
            return CheckResult(
                f"{name}.pd_range",
                Status.WARNING,
                "PD outside [0, 1]; values will be clipped because clip_pd is enabled",
                {"min": min_pd, "max": max_pd},
            )
    elif min_pd < 0 or max_pd > 1:
        return CheckResult(
            f"{name}.pd_range",
            Status.CRITICAL,
            "PD must be in [0, 1]",
            {"min": min_pd, "max": max_pd},
        )
    if series.n_unique() <= 1:
        return CheckResult(
            f"{name}.pd_constant",
            Status.WARNING,
            "PD is constant; discrimination and calibration evidence may be weak",
            {"value": min_pd},
        )
    return CheckResult(f"{name}.pd_range", Status.OK, "PD is numeric and in valid range")


def _check_constant_numeric(frame: pl.DataFrame, column: str, name: str, label: str) -> CheckResult:
    series = frame.select(pl.col(column).cast(pl.Float64, strict=False)).to_series()
    if series.null_count() > 0:
        return CheckResult(
            f"{name}.{label}_numeric",
            Status.WARNING,
            f"{label.title()} contains nulls or non-numeric values",
        )
    if series.n_unique() <= 1:
        return CheckResult(
            f"{name}.{label}_constant",
            Status.WARNING,
            f"{label.title()} is constant; discrimination evidence may be weak",
            {"value": optional_float(series.min())},
        )
    return CheckResult(f"{name}.{label}_constant", Status.OK, f"{label.title()} varies")


def _check_weight(frame: pl.DataFrame, weight_col: str, name: str) -> CheckResult:
    series = frame.select(pl.col(weight_col).cast(pl.Float64, strict=False)).to_series()
    if series.null_count() > 0:
        return CheckResult(
            f"{name}.weight_numeric", Status.CRITICAL, "Weight contains nulls or non-numeric values"
        )
    if bool((series <= 0).any()):
        return CheckResult(f"{name}.weight_positive", Status.CRITICAL, "Weight must be positive")
    return CheckResult(f"{name}.weight_positive", Status.OK, "Weight is positive")


def _check_period(frame: pl.DataFrame, period_col: str, name: str) -> CheckResult:
    try:
        parsed = frame.select(
            pl.col(period_col).cast(pl.Utf8).str.to_date(strict=False)
        ).to_series()
    except Exception as exc:  # pragma: no cover - defensive against polars parsing internals
        return CheckResult(
            f"{name}.period_parseable", Status.WARNING, f"Period parse failed: {exc}"
        )
    if parsed.null_count() == frame.height:
        return CheckResult(f"{name}.period_parseable", Status.WARNING, "Period is not date-like")
    return CheckResult(f"{name}.period_parseable", Status.OK, "Period is parseable")


def _check_id_period_duplicates(
    frame: pl.DataFrame, id_col: str, period_col: str, name: str
) -> CheckResult:
    duplicate_count = (
        frame.group_by([id_col, period_col])
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.col("len").sum())
        .item()
    )
    if duplicate_count:
        return CheckResult(
            f"{name}.id_period_duplicates",
            Status.CRITICAL,
            "Duplicate id-period records found",
            int(duplicate_count),
        )
    return CheckResult(f"{name}.id_period_duplicates", Status.OK, "No id-period duplicates")


def _check_segment_cardinality(frame: pl.DataFrame, segment_col: str, name: str) -> CheckResult:
    cardinality = frame[segment_col].n_unique()
    if cardinality > min(100, max(20, frame.height // 10)):
        return CheckResult(
            f"{name}.segment_cardinality.{segment_col}",
            Status.WARNING,
            "Segment cardinality may be too high for stable reporting",
            int(cardinality),
        )
    return CheckResult(
        f"{name}.segment_cardinality.{segment_col}",
        Status.OK,
        "Segment cardinality is acceptable",
        int(cardinality),
    )

import polars as pl

from credit_risk_validation import PDValidationSuite
from credit_risk_validation.config import ColumnConfig, ValidationOptions
from credit_risk_validation.metrics.segmentation import segment_analysis
from credit_risk_validation.status import Status


def test_segment_analysis_returns_rows(sample_frame: pl.DataFrame) -> None:
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=["segment"],
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run(validation_data=sample_frame)
    table = result.tables["segment_analysis"]
    assert table.height == 2
    assert set(table["status"].to_list()) == {Status.OK.value}


def test_small_segment_is_insufficient(sample_frame: pl.DataFrame) -> None:
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=["segment"],
        min_events=100,
        min_non_events=100,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run(validation_data=sample_frame)
    assert Status.INSUFFICIENT_DATA.value in result.tables["segment_analysis"]["status"].to_list()


def test_null_segment_is_a_real_group() -> None:
    frame = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "pd": [0.1, 0.8, 0.2, 0.7],
            "segment": [None, None, "A", "A"],
        }
    )

    table = segment_analysis(
        frame,
        columns=ColumnConfig(score=None, segments=["segment"]),
        validation=ValidationOptions(
            min_events=1,
            min_non_events=1,
            min_rows=1,
            min_segment_size=1,
        ),
    )

    assert table.select("segment", "count").to_dicts() == [
        {"segment": "A", "count": 2},
        {"segment": "MISSING", "count": 2},
    ]


def test_structured_groups_do_not_collide_on_delimiters_or_backslashes() -> None:
    groups = [
        ("A | B", "C"),
        ("A", "B | C"),
        ("A\\", "B"),
        ("A", "\\| B"),
    ]
    frame = pl.DataFrame(
        {
            "target": [0, 1] * len(groups),
            "pd": [0.1, 0.8] * len(groups),
            "first": [value for first, _ in groups for value in (first, first)],
            "second": [value for _, second in groups for value in (second, second)],
        }
    )

    table = segment_analysis(
        frame,
        columns=ColumnConfig(score=None, segments=["first", "second"]),
        validation=ValidationOptions(
            min_events=1,
            min_non_events=1,
            min_rows=1,
            min_segment_size=1,
        ),
    )

    assert table.height == len(groups)
    assert table["segment"].n_unique() == len(groups)
    assert "A \\| B | C" in table["segment"].to_list()
    assert "A\\\\ | B" in table["segment"].to_list()


def test_literal_missing_does_not_collide_with_null() -> None:
    frame = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "pd": [0.1, 0.8, 0.2, 0.7],
            "segment": [None, None, "MISSING", "MISSING"],
        }
    )

    table = segment_analysis(
        frame,
        columns=ColumnConfig(score=None, segments=["segment"]),
        validation=ValidationOptions(
            min_events=1,
            min_non_events=1,
            min_rows=1,
            min_segment_size=1,
        ),
    )

    assert set(table["segment"].to_list()) == {"MISSING", "\\MISSING"}


def test_min_segment_size_marks_group_as_insufficient() -> None:
    frame = pl.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "pd": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.2, 0.7, 0.1, 0.8],
            "segment": ["small"] * 4 + ["large"] * 6,
        }
    )

    table = segment_analysis(
        frame,
        columns=ColumnConfig(score=None, segments=["segment"]),
        validation=ValidationOptions(
            min_events=1,
            min_non_events=1,
            min_rows=1,
            min_segment_size=5,
        ),
    )

    statuses = dict(table.select("segment", "status").iter_rows())
    assert statuses == {
        "large": Status.OK.value,
        "small": Status.INSUFFICIENT_DATA.value,
    }

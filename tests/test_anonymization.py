import json
from pathlib import Path

import polars as pl

from credit_risk_validation.config import (
    ColumnConfig,
    ModelMetadata,
    PDValidationConfig,
    ReportConfig,
    ValidationOptions,
)
from credit_risk_validation.suite import PDValidationSuite

SECRETS = {
    "Secret Default Flag",
    "Secret Forecast PD",
    "Secret Score",
    "Secret Quarter",
    "Secret Weight",
    "Secret Customer ID",
    "Secret Portfolio",
    "SecretRetail",
    "SecretCorporate",
    "SecretQuarterOne",
    "SecretQuarterTwo",
    "Secret Model 2026",
    "Secret Risk Team",
    "Secret underwriting purpose",
    "Secret Committee Report",
}


def _secret_result(*, anonymize: bool = True):
    rows = 240
    frame = pl.DataFrame(
        {
            "Secret Default Flag": ([0, 0, 0, 1, 0, 1] * (rows // 6))[:rows],
            "Secret Forecast PD": ([0.02, 0.04, 0.08, 0.35, 0.12, 0.55] * (rows // 6))[:rows],
            "Secret Score": ([820, 780, 740, 620, 700, 560] * (rows // 6))[:rows],
            "Secret Quarter": (["SecretQuarterOne", "SecretQuarterTwo"] * (rows // 2))[:rows],
            "Secret Weight": [1.0] * rows,
            "Secret Customer ID": [f"customer-{index}" for index in range(rows)],
            "Secret Portfolio": (["SecretRetail", "SecretCorporate"] * (rows // 2))[:rows],
        }
    )
    config = PDValidationConfig(
        model=ModelMetadata(
            name="Secret Model 2026",
            owner="Secret Risk Team",
            purpose="Secret underwriting purpose",
        ),
        columns=ColumnConfig(
            target="Secret Default Flag",
            pd="Secret Forecast PD",
            score="Secret Score",
            period="Secret Quarter",
            weight="Secret Weight",
            id="Secret Customer ID",
            segments=["Secret Portfolio"],
        ),
        validation=ValidationOptions(
            min_events=5,
            min_non_events=5,
            min_rows=20,
            min_segment_size=20,
            score_direction="lower_is_riskier",
        ),
        report=ReportConfig(
            title="Secret Committee Report",
            anonymize=anonymize,
        ),
    )
    return PDValidationSuite.from_config(config).run_drift(
        reference_data=frame,
        current_data=frame,
    )


def _table_text(directory: Path, file_format: str) -> str:
    rows: dict[str, list[dict[str, object]]] = {}
    for path in sorted(directory.glob(f"*.{file_format}")):
        table = pl.read_csv(path) if file_format == "csv" else pl.read_parquet(path)
        rows[path.name] = table.to_dicts()
    return json.dumps(rows, default=str, sort_keys=True)


def test_anonymized_projection_covers_every_export_without_mutating_result(
    tmp_path: Path,
) -> None:
    result = _secret_result()
    original_config = result.config.model_dump(mode="json")
    original_tables = {name: table.clone() for name, table in result.tables.items()}
    original_fingerprints = {
        key: value for key, value in result.metadata.items() if key.endswith("sha256")
    }

    json_path = tmp_path / "result.json"
    html_path = tmp_path / "report.html"
    model_card_path = tmp_path / "model-card.md"
    csv_directory = tmp_path / "csv"
    parquet_directory = tmp_path / "parquet"
    result.to_json(json_path)
    result.to_html(html_path)
    result.to_model_card(model_card_path)
    result.to_tables(csv_directory, file_format="csv")
    result.to_tables(parquet_directory, file_format="parquet")

    exported_texts = [
        json_path.read_text(encoding="utf-8"),
        html_path.read_text(encoding="utf-8"),
        model_card_path.read_text(encoding="utf-8"),
        _table_text(csv_directory, "csv"),
        _table_text(parquet_directory, "parquet"),
    ]
    for exported in exported_texts:
        assert all(secret not in exported for secret in SECRETS)

    payload = json.loads(exported_texts[0])
    assert payload["config"]["columns"] == {
        "target": "target",
        "pd": "pd",
        "score": "score",
        "period": "period",
        "weight": "weight",
        "id": "id",
        "segments": ["segment_1"],
    }
    assert payload["config"]["model"]["name"] == "Anonymous PD Model"
    assert payload["config"]["model"]["owner"] == "REDACTED"
    assert payload["config"]["model"]["purpose"] == "REDACTED"
    assert payload["config"]["report"]["title"] == "Anonymized Validation Report"
    assert payload["metadata"] | original_fingerprints == payload["metadata"]
    assert payload["metadata"]["config_sha256"] == original_fingerprints["config_sha256"]
    assert "value_" in exported_texts[0]
    assert "psi_segment_1" in payload["metrics"]

    assert result.config.model_dump(mode="json") == original_config
    assert result.config.model.name == "Secret Model 2026"
    for name, table in original_tables.items():
        assert result.tables[name].equals(table)


def test_anonymized_aliases_are_deterministic_and_consistent() -> None:
    result = _secret_result()

    first = result.to_dict()
    second = result.to_dict()

    assert first == second
    segment_values = {row["segment"] for row in first["tables"]["segment_drift"]}
    analysis_values = {row["segment"] for row in first["tables"]["segment_analysis"]}
    assert segment_values == analysis_values
    assert segment_values == {"value_001", "value_004"}
    assert all(value.startswith("value_") for value in segment_values)
    assert all(len(value.removeprefix("value_")) == 3 for value in segment_values)


def test_anonymization_opt_out_preserves_original_values(tmp_path: Path) -> None:
    result = _secret_result(anonymize=False)

    result.to_json(tmp_path / "result.json")
    result.to_html(tmp_path / "report.html")
    result.to_model_card(tmp_path / "model-card.md")
    result.to_tables(tmp_path / "tables")

    payload = result.to_dict()
    assert payload["config"] == result.config.to_public_dict()
    assert payload["config"]["model"]["name"] == "Secret Model 2026"
    assert payload["config"]["columns"]["segments"] == ["Secret Portfolio"]
    assert "Secret Committee Report" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Secret Risk Team" in (tmp_path / "model-card.md").read_text(encoding="utf-8")
    assert "SecretRetail" in _table_text(tmp_path / "tables", "csv")

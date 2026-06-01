from credit_risk_validation.metrics.champion_challenger import compare_champion_challenger


def test_champion_challenger_table() -> None:
    table = compare_champion_challenger(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.7, 0.8],
        [0.05, 0.3, 0.6, 0.9],
    )
    assert {"metric", "champion", "challenger", "absolute_delta"}.issubset(set(table.columns))
    assert "Review required" in table["message"][0]

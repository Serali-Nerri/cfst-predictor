import json

from src.utils.model_utils import load_best_params


def test_load_best_params_returns_none_when_file_missing(tmp_path):
    result = load_best_params(str(tmp_path / "missing.json"), expected_context_hash="abc123")
    assert result is None


def test_load_best_params_returns_none_for_invalid_format(tmp_path):
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps({"best_rmse": 1.23}), encoding="utf-8")

    result = load_best_params(str(path), expected_context_hash="abc123")
    assert result is None


def test_load_best_params_returns_none_for_context_mismatch(tmp_path):
    path = tmp_path / "best_params.json"
    path.write_text(
        json.dumps(
            {
                "parameters": {"max_depth": 5},
                "context_hash": "wrong-context",
            }
        ),
        encoding="utf-8",
    )

    result = load_best_params(str(path), expected_context_hash="expected-context")
    assert result is None


def test_load_best_params_returns_parameters_on_context_match(tmp_path):
    path = tmp_path / "best_params.json"
    expected = {"max_depth": 5, "learning_rate": 0.05}
    path.write_text(
        json.dumps(
            {
                "parameters": expected,
                "context_hash": "expected-context",
            }
        ),
        encoding="utf-8",
    )

    result = load_best_params(str(path), expected_context_hash="expected-context")
    assert result == expected

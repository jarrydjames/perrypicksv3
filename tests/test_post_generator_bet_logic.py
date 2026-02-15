import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_modules():
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = []
    automation_pkg = types.ModuleType("src.automation")
    automation_pkg.__path__ = []

    sys.modules["src"] = src_pkg
    sys.modules["src.automation"] = automation_pkg

    _load_module("src.betting", "src/betting.py")
    _load_module("src.automation.prediction_formatter", "src/automation/prediction_formatter.py")
    helpers = _load_module("src.automation.post_generator_helpers", "src/automation/post_generator_helpers.py")
    post_generator = _load_module("src.automation.post_generator", "src/automation/post_generator.py")
    return helpers, post_generator


HELPERS, POST_GENERATOR = _bootstrap_modules()
PostGenerator = POST_GENERATOR.PostGenerator
_generate_best_bets = HELPERS._generate_best_bets


def _base_prediction():
    return {
        "status": "success",
        "home_name": "Heat",
        "away_name": "Knicks",
        "total": 226.0,
        "margin": 3.5,
        "total_sd": 8.0,
        "margin_sd": 7.0,
        "home_win_prob": 0.63,
        "odds_total_line": 221.5,
        "odds_total_over": -110,
        "odds_total_under": -110,
        "odds_spread_home_line": -1.0,
        "odds_spread_home": -110,
        "odds_spread_away": -110,
        "odds_home_ml": -120,
        "odds_away_ml": 110,
    }


def test_generate_best_bets_requires_edge_and_probability_thresholds_and_sorts():
    prediction = _base_prediction()
    bets = _generate_best_bets(prediction, "halftime", max_bets=5)

    assert bets, "Expected at least one recommended bet"

    for bet in bets:
        assert "edge_value" in bet
        assert "hit_probability" in bet
        assert "confidence_tier" in bet
        assert "variance" in bet

        if bet["type"] == "Total":
            assert bet["edge_value"] >= 2.0
            assert bet["hit_probability"] >= 0.56
        elif bet["type"] == "Spread":
            assert bet["edge_value"] >= 1.5
            assert bet["hit_probability"] >= 0.57
        elif bet["type"] == "Moneyline":
            assert bet["edge_value"] >= 0.03
            assert bet["hit_probability"] >= 0.58

    ordered = sorted(
        bets,
        key=lambda b: (b["edge_value"], b["hit_probability"], -b["variance"]),
        reverse=True,
    )
    assert bets == ordered


def test_post_generator_formats_probability_recommendations_and_summary():
    prediction = _base_prediction()
    post = PostGenerator().generate_halftime_post(prediction, platform="discord")

    assert "Hit Probability:" in post
    assert "Confidence:" in post
    assert "Summary:" in post
    assert "Confidence Distribution:" in post
    assert "Average Edge by Type:" in post

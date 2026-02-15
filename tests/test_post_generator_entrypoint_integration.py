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


def _bootstrap_post_generator():
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = []
    automation_pkg = types.ModuleType("src.automation")
    automation_pkg.__path__ = []

    sys.modules["src"] = src_pkg
    sys.modules["src.automation"] = automation_pkg

    _load_module("src.betting", "src/betting.py")
    _load_module("src.automation.prediction_formatter", "src/automation/prediction_formatter.py")
    _load_module("src.automation.post_generator_helpers", "src/automation/post_generator_helpers.py")
    post_generator = _load_module("src.automation.post_generator", "src/automation/post_generator.py")
    return post_generator.PostGenerator


PostGenerator = _bootstrap_post_generator()


def _prediction_payload(total: float, margin: float, home_win_prob: float):
    return {
        "status": "success",
        "trigger_type": "halftime",
        "home_name": "Heat",
        "away_name": "Knicks",
        "h1_home": 56,
        "h1_away": 51,
        "pred_final_home": (total + margin) / 2,
        "pred_final_away": (total - margin) / 2,
        "total": total,
        "margin": margin,
        "total_sd": 8.0,
        "margin_sd": 7.0,
        "home_win_prob": home_win_prob,
        "odds_total_line": 221.5,
        "odds_total_over": -110,
        "odds_total_under": -110,
        "odds_spread_home_line": -1.0,
        "odds_spread_home": -110,
        "odds_spread_away": -110,
        "odds_home_ml": -120,
        "odds_away_ml": 110,
    }


def test_generate_halftime_post_entrypoint_includes_ranked_recommendations_and_summary():
    payload = _prediction_payload(total=226.0, margin=3.5, home_win_prob=0.63)

    post = PostGenerator().generate_halftime_post(payload, platform="discord")

    assert "Best Bets (edge, then hit probability):" in post
    assert "Heat ML" in post or "Knicks ML" in post
    assert "Summary:" in post
    assert "Average Edge by Type:" in post
    assert "No bets passed edge + hit-probability thresholds" not in post


def test_generate_halftime_post_entrypoint_handles_no_qualified_bets():
    payload = _prediction_payload(total=221.6, margin=0.2, home_win_prob=0.51)

    post = PostGenerator().generate_halftime_post(payload, platform="twitter")

    assert "No bets passed edge + hit-probability thresholds." in post

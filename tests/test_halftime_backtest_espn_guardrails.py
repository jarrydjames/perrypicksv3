from __future__ import annotations

import ast
from pathlib import Path


SCRIPT_PATH = Path("scripts/halftime_backtest_espn.py")


def _source() -> str:
    return SCRIPT_PATH.read_text()


def test_has_robust_topk_param_builder() -> None:
    src = _source()
    assert "def _robust_topk_params" in src
    assert "np.median" in src


def test_main_runs_feature_gate_before_training_step() -> None:
    src = _source()
    gate_idx = src.index("Fail-fast feature health gate")
    train_idx = src.index("STEP 4: TRAINING PRODUCTION MODEL")
    assert gate_idx < train_idx


def test_cli_has_new_alignment_and_env_flags() -> None:
    src = _source()
    assert "--param-selection" in src
    assert "--param-topk" in src
    assert "--sigma-calib-frac" in src
    assert "--allow-feature-issues" in src


def test_script_parses() -> None:
    ast.parse(_source())

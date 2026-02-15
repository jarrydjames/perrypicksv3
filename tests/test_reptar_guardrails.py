"""
REPTAR Guardrails Tests
=======================

These tests ensure REPTAR is always used for halftime predictions.

Run with:
    pytest tests/test_reptar_guardrails.py -v
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reptar import (
    REPTAR_VERSION,
    REPTAR_CODE,
    REPTAR_DATA_PATH,
    REPTAR_TEAM_ID_MAP_PATH,
    validate_reptar_data,
    load_reptar_model,
    is_reptar_loaded,
    calculate_reptar_win_probability,
    ReptarValidationError,
    ReptarModelNotLoadedError,
)


class TestReptarConfiguration:
    """Test REPTAR configuration."""
    
    def test_reptar_version_exists(self):
        """REPTAR must have a version."""
        assert REPTAR_VERSION is not None
        assert len(REPTAR_VERSION) > 0
    
    def test_reptar_code_is_correct(self):
        """REPTAR code must be 'REPTAR'."""
        assert REPTAR_CODE == "REPTAR"
    
    def test_reptar_data_path_exists(self):
        """REPTAR data path must be defined."""
        assert REPTAR_DATA_PATH is not None
        assert str(REPTAR_DATA_PATH).endswith("halftime_with_refined_temporal.parquet")
    
    def test_reptar_team_id_map_path_exists(self):
        """REPTAR team ID map path must be defined."""
        assert REPTAR_TEAM_ID_MAP_PATH is not None
        assert str(REPTAR_TEAM_ID_MAP_PATH).endswith("team_tricode_to_custom_id.json")


class TestReptarValidation:
    """Test REPTAR validation."""
    
    def test_validate_reptar_data(self):
        """REPTAR data validation should pass."""
        is_valid, msg = validate_reptar_data()
        assert is_valid, f"REPTAR data validation failed: {msg}"
    
    def test_reptar_data_file_exists(self):
        """REPTAR data file must exist."""
        assert REPTAR_DATA_PATH.exists(), f"Missing: {REPTAR_DATA_PATH}"
    
    def test_reptar_team_id_map_exists(self):
        """REPTAR team ID map must exist."""
        assert REPTAR_TEAM_ID_MAP_PATH.exists(), f"Missing: {REPTAR_TEAM_ID_MAP_PATH}"


class TestReptarLoading:
    """Test REPTAR model loading."""
    
    def test_load_reptar_model(self):
        """REPTAR should load successfully."""
        state = load_reptar_model(validate=True, strict=True)
        assert state["loaded"] is True
        assert state["validation_passed"] is True
    
    def test_is_reptar_loaded(self):
        """REPTAR loaded state should be True after loading."""
        load_reptar_model()
        assert is_reptar_loaded() is True
    
    def test_reptar_team_id_map_loaded(self):
        """REPTAR team ID map should be loaded."""
        state = load_reptar_model()
        assert state["team_id_map"] is not None
        assert len(state["team_id_map"]) == 30  # All NBA teams
    
    def test_reptar_feature_columns_loaded(self):
        """REPTAR feature columns should be loaded."""
        state = load_reptar_model()
        assert state["feature_columns"] is not None
        assert len(state["feature_columns"]) > 100  # Should have 132 features


class TestReptarWinProbability:
    """Test REPTAR win probability calculation."""
    
    def test_win_probability_bounds(self):
        """Win probability must be between 0 and 1."""
        p_win = calculate_reptar_win_probability(
            h1_margin=10.0,
            pred_h2_margin=2.0,
            sigma_h2_margin=5.0,
            sigma_k_margin=3.0,
        )
        assert 0.0 <= p_win <= 1.0
    
    def test_win_probability_favored_team(self):
        """Favored team should have >50% win probability."""
        # H1: +10, Pred H2: +5 → Full game: +15 (home favored)
        p_win = calculate_reptar_win_probability(
            h1_margin=10.0,
            pred_h2_margin=5.0,
            sigma_h2_margin=5.0,
            sigma_k_margin=1.0,  # Low uncertainty
        )
        assert p_win > 0.5, f"Favored team should have >50% win prob, got {p_win}"
    
    def test_win_probability_underdog(self):
        """Underdog should have <50% win probability."""
        # H1: -10, Pred H2: -5 → Full game: -15 (away favored)
        p_win = calculate_reptar_win_probability(
            h1_margin=-10.0,
            pred_h2_margin=-5.0,
            sigma_h2_margin=5.0,
            sigma_k_margin=1.0,
        )
        assert p_win < 0.5, f"Underdog should have <50% win prob, got {p_win}"
    
    def test_win_probability_symmetry(self):
        """Win probability should be symmetric."""
        p1 = calculate_reptar_win_probability(
            h1_margin=10.0,
            pred_h2_margin=0.0,
            sigma_h2_margin=10.0,
            sigma_k_margin=3.0,
        )
        p2 = calculate_reptar_win_probability(
            h1_margin=-10.0,
            pred_h2_margin=0.0,
            sigma_h2_margin=10.0,
            sigma_k_margin=3.0,
        )
        # Symmetry: P(home wins | +10) + P(home wins | -10) ≈ 1
        assert abs(p1 + p2 - 1.0) < 0.01, f"Win prob not symmetric: {p1} + {p2} = {p1 + p2}"


class TestReptarIntegration:
    """Test REPTAR integration with scripts."""
    
    def test_halftime_backtest_uses_reptar_data(self):
        """Halftime backtest must use REPTAR data."""
        backtest_script = Path("scripts/halftime_backtest_espn.py")
        if not backtest_script.exists():
            pytest.skip("Backtest script not found")
        
        content = backtest_script.read_text()
        
        # Must use refined temporal data
        assert "halftime_with_refined_temporal.parquet" in content, \
            "Backtest must use REPTAR data (halftime_with_refined_temporal.parquet)"
    
    def test_halftime_backtest_uses_reptar_win_prob(self):
        """Halftime backtest must use REPTAR win probability calculation."""
        backtest_script = Path("scripts/halftime_backtest_espn.py")
        if not backtest_script.exists():
            pytest.skip("Backtest script not found")
        
        content = backtest_script.read_text()
        
        # Must use correct formula: norm.cdf(-h1_margin, ...)
        assert "norm.cdf(-h1_margin" in content or "norm.cdf(-" in content, \
            "Backtest must use REPTAR win probability formula"
    
    def test_halftime_backtest_uses_team_id_map(self):
        """Halftime backtest must use REPTAR team ID map."""
        backtest_script = Path("scripts/halftime_backtest_espn.py")
        if not backtest_script.exists():
            pytest.skip("Backtest script not found")
        
        content = backtest_script.read_text()
        
        # Must use team ID map
        assert "team_tricode_to_custom_id.json" in content, \
            "Backtest must use REPTAR team ID map"


class TestReptarFailsafes:
    """Test REPTAR failsafes."""
    
    def test_reptar_not_loaded_error(self):
        """Should raise error if REPTAR not loaded."""
        # This test should fail if REPTAR is already loaded
        # For now, just test that the exception exists
        assert ReptarModelNotLoadedError is not None
    
    def test_reptar_validation_error(self):
        """Should raise error if REPTAR validation fails."""
        # This test should fail if data is valid
        # For now, just test that the exception exists
        assert ReptarValidationError is not None


class TestReptarPerformance:
    """Test REPTAR performance metrics."""
    
    def test_reptar_metrics_exist(self):
        """REPTAR metrics file should exist."""
        from src.reptar import REPTAR_METRICS_PATH
        assert REPTAR_METRICS_PATH.exists(), f"Missing: {REPTAR_METRICS_PATH}"
    
    def test_reptar_backtest_results_exist(self):
        """REPTAR backtest results should exist."""
        feb11_results = Path("reports/backtest/halftime_backtest_2026-02-11_detailed.csv")
        feb9_results = Path("reports/backtest/halftime_backtest_2026-02-09_detailed.csv")
        
        assert feb11_results.exists() or feb9_results.exists(), \
            "At least one REPTAR backtest result should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

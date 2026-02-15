"""
REPTAR Integration Tests
========================

These tests ensure REPTAR is integrated correctly across all scripts.

Run with:
    pytest tests/test_reptar_integration.py -v
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestReptarFormulaInScripts:
    """Test that all scripts use the REPTAR win probability formula."""
    
    def test_halftime_backtest_espn_uses_reptar_formula(self):
        """halftime_backtest_espn.py must use REPTAR formula."""
        script_path = Path("scripts/halftime_backtest_espn.py")
        assert script_path.exists()
        
        content = script_path.read_text()
        
        # Must use correct formula with -h1_margin
        assert "norm.cdf(-h1_margin" in content or "norm.cdf(-" in content
    
    def test_halftime_backtest_production_uses_reptar_formula(self):
        """halftime_backtest_production.py must use REPTAR formula."""
        script_path = Path("scripts/halftime_backtest_production.py")
        if not script_path.exists():
            pytest.skip("Script not found")
        
        content = script_path.read_text()
        
        # Must use correct formula with -h1_margin
        assert "norm.cdf(-h1_margin" in content or "norm.cdf(-" in content
    
    def test_extract_oof_predictions_uses_reptar_formula(self):
        """extract_oof_predictions.py must use REPTAR formula."""
        script_path = Path("scripts/extract_oof_predictions.py")
        if not script_path.exists():
            pytest.skip("Script not found")
        
        content = script_path.read_text()
        
        # Must use correct formula with -h1_margin
        assert "norm.cdf(-h1_margin" in content or "norm.cdf(-" in content


class TestReptarDataInScripts:
    """Test that all scripts use REPTAR data."""
    
    def test_halftime_backtest_espn_uses_reptar_data(self):
        """halftime_backtest_espn.py must use REPTAR data."""
        script_path = Path("scripts/halftime_backtest_espn.py")
        assert script_path.exists()
        
        content = script_path.read_text()
        
        # Must use refined temporal features
        assert "halftime_with_refined_temporal.parquet" in content
    
    def test_halftime_backtest_espn_uses_team_id_map(self):
        """halftime_backtest_espn.py must use team ID map."""
        script_path = Path("scripts/halftime_backtest_espn.py")
        assert script_path.exists()
        
        content = script_path.read_text()
        
        # Must have team ID mapping logic
        has_mapping = "tri_to_id" in content or "team_id" in content
        assert has_mapping


class TestReptarIntegrationModule:
    """Test REPTAR integration module."""
    
    def test_reptar_integration_module_exists(self):
        """REPTAR integration module must exist."""
        module_path = Path("src/reptar_integration.py")
        assert module_path.exists()
    
    def test_reptar_integrator_class_exists(self):
        """ReptarIntegrator class must exist."""
        from src.reptar_integration import ReptarIntegrator
        assert ReptarIntegrator is not None
    
    def test_reptar_integrator_can_calculate_win_prob(self):
        """ReptarIntegrator can calculate win probability."""
        from src.reptar_integration import ReptarIntegrator
        
        integrator = ReptarIntegrator(auto_load=False)
        
        p_win = integrator.calculate_win_probability(
            h1_margin=10.0,
            pred_h2_margin=5.0,
            sigma_h2_margin=5.0,
        )
        
        assert 0.0 <= p_win <= 1.0
        assert p_win > 0.5
    
    def test_reptar_integrator_can_enrich_result(self):
        """ReptarIntegrator can enrich halftime result."""
        from src.reptar_integration import ReptarIntegrator
        
        integrator = ReptarIntegrator(auto_load=False)
        
        result = {
            'h1_home': 60,
            'h1_away': 50,
            'pred_2h_home': 55,
            'pred_2h_away': 50,
            'margin_sd': 10.0,
        }
        
        enriched = integrator.enrich_halftime_result(result)
        
        assert 'home_win_prob' in enriched
        assert 'away_win_prob' in enriched
        assert 'reptar_version' in enriched
        
        prob = enriched['home_win_prob']
        assert 0.0 <= prob <= 1.0


class TestPredictApiUsesReptar:
    """Test that predict_api.py uses REPTAR."""
    
    def test_predict_api_imports_reptar_integration(self):
        """predict_api.py must import REPTAR integration."""
        script_path = Path("src/predict_api.py")
        assert script_path.exists()
        
        content = script_path.read_text()
        
        # Must import REPTAR integration
        has_import = "from src.reptar_integration import" in content
        assert has_import
    
    def test_predict_api_enriches_halftime_with_reptar(self):
        """predict_api.py must enrich halftime with REPTAR."""
        script_path = Path("src/predict_api.py")
        assert script_path.exists()
        
        content = script_path.read_text()
        
        # Must call enrich_halftime_prediction
        assert "enrich_halftime_prediction" in content


class TestReptarCommentsInCode:
    """Test that REPTAR is properly documented in code."""
    
    def test_halftime_backtest_has_reptar_comment(self):
        """halftime_backtest_espn.py should have REPTAR comment."""
        script_path = Path("scripts/halftime_backtest_espn.py")
        content = script_path.read_text()
        
        # Should have REPTAR comment
        has_reptar = "REPTAR" in content or "reptar" in content.lower()
        assert has_reptar
    
    def test_predict_api_has_reptar_comment(self):
        """predict_api.py should have REPTAR comment."""
        script_path = Path("src/predict_api.py")
        content = script_path.read_text()
        
        # Should have REPTAR comment
        has_reptar = "REPTAR" in content or "reptar" in content.lower()
        assert has_reptar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

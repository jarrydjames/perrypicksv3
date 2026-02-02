"""Model monitor for tracking model health and calibration.

Monitors:
- Prediction bias over time
- Calibration drift
- Model staleness
- Feature importance changes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json


class ModelMonitor:
    """Monitor model health and calibration."""
    
    def __init__(self, storage_dir: str = 'data/monitoring'):
        """
        Initialize model monitor.
        
        Args:
            storage_dir: Directory to store monitoring data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.calibration_file = self.storage_dir / 'calibration_metrics.json'
        self.bias_file = self.storage_dir / 'bias_tracking.json'
        self.model_health_file = self.storage_dir / 'model_health.json'
        
        self.calibration_metrics = []
        self.bias_tracking = {}
        self.model_health = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load existing monitoring data."""
        if self.calibration_file.exists():
            with open(self.calibration_file, 'r') as f:
                self.calibration_metrics = json.load(f)
        
        if self.bias_file.exists():
            with open(self.bias_file, 'r') as f:
                self.bias_tracking = json.load(f)
        
        if self.model_health_file.exists():
            with open(self.model_health_file, 'r') as f:
                self.model_health = json.load(f)
        else:
            self.model_health = {
                'last_trained': None,
                'days_since_training': 0,
                'total_predictions_made': 0,
                'model_version': 'v1.0',
            }
    
    def record_calibration(
        self,
        timestamp: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        win_probs: Optional[np.ndarray] = None,
        actual_winners: Optional[np.ndarray] = None
    ):
        """
        Record calibration metrics.
        
        Args:
            timestamp: ISO format timestamp
            predictions: Predicted values
            actuals: Actual values
            win_probs: Predicted win probabilities
            actual_winners: Actual win indicators (1/0)
        """
        calibration_entry = {
            'timestamp': timestamp,
            'mae': float(np.mean(np.abs(predictions - actuals))),
            'rmse': float(np.sqrt(np.mean((predictions - actuals) ** 2))),
            'bias': float(np.mean(predictions - actuals)),
            'n_samples': int(len(predictions)),
        }
        
        # Win probability calibration
        if win_probs is not None and actual_winners is not None:
            calibration_entry = {
                'timestamp': timestamp,
                'bins': [],
                'reliability': {},
            }
            
            # Create calibration bins
            for bin_center in [0.1, 0.3, 0.5, 0.7, 0.9]:
                bin_low = bin_center - 0.1
                bin_high = bin_center + 0.1
                bin_mask = (win_probs >= bin_low) & (win_probs < bin_high)
                
                if np.sum(bin_mask) > 0:
                    actual_rate = np.mean(actual_winners[bin_mask])
                    calibration_entry['bins'].append({
                        'center': bin_center,
                        'low': bin_low,
                        'high': bin_high,
                        'predicted_prob': bin_center,
                        'actual_rate': float(actual_rate),
                        'n_games': int(np.sum(bin_mask)),
                    })
                    calibration_entry['reliability'][f'bin_{bin_center}'] = {
                        'expected': bin_center,
                        'actual': float(actual_rate),
                        'error': abs(bin_center - actual_rate),
                    }
            
            self.calibration_metrics.append(calibration_entry)
        
        # Bias tracking
        self.bias_tracking[timestamp] = {
            'bias': calibration_entry['bias'],
            'mae': calibration_entry['mae'],
        }
        
        self._save_data()
    
    def _save_data(self):
        """Save monitoring data."""
        with open(self.calibration_file, 'w') as f:
            json.dump(self.calibration_metrics, f, indent=2)
        
        with open(self.bias_file, 'w') as f:
            json.dump(self.bias_tracking, f, indent=2)
        
        with open(self.model_health_file, 'w') as f:
            json.dump(self.model_health, f, indent=2)
    
    def get_calibration_report(self) -> str:
        """
        Generate calibration report.
        
        Returns:
            Formatted report string
        """
        if not self.calibration_metrics:
            return "No calibration data available."
        
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL CALIBRATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Recent metrics
        recent = self.calibration_metrics[-1]
        lines.append("RECENT CALIBRATION:")
        lines.append(f"  Timestamp: {recent.get('timestamp', 'N/A')}")
        if 'mae' in recent:
            lines.append(f"  MAE: {recent['mae']:.2f}")
        if 'rmse' in recent:
            lines.append(f"  RMSE: {recent['rmse']:.2f}")
        if 'bias' in recent:
            lines.append(f"  Bias: {recent['bias']:+.2f}")
        lines.append("")
        
        # Win probability calibration
        if 'bins' in recent:
            lines.append("WIN PROBABILITY CALIBRATION:")
            for bin_data in recent['bins']:
                predicted = bin_data['predicted_prob']
                actual = bin_data['actual_rate']
                error = abs(predicted - actual)
                lines.append(f"  {predicted:.1f}: {actual:.1f} (error: {error:.2f})")
            lines.append("")
        
        # Bias tracking
        if self.bias_tracking:
            lines.append("BIAS TRACKING (Last 10 readings):")
            sorted_timestamps = sorted(self.bias_tracking.keys())[-10:]
            for ts in sorted_timestamps:
                bias = self.bias_tracking[ts]['bias']
                lines.append(f"  {ts[:10]}: {bias:+.2f}")
            lines.append("")
        
        # Model health
        lines.append("MODEL HEALTH:")
        lines.append(f"  Version: {self.model_health.get('model_version', 'N/A')}")
        lines.append(f"  Last Trained: {self.model_health.get('last_trained', 'N/A')}")
        lines.append(f"  Days Since Training: {self.model_health.get('days_since_training', 0)}")
        lines.append(f"  Total Predictions: {self.model_health.get('total_predictions_made', 0)}")
        
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def detect_drift(self, window_size: int = 7, threshold: float = 0.1) -> Dict[str, bool]:
        """
        Detect model drift.
        
        Args:
            window_size: Number of recent readings to compare
            threshold: Drift threshold
            
        Returns:
            Dictionary with drift detection results
        """
        if not self.bias_tracking:
            return {'drift_detected': False, 'reason': 'No data'}
        
        timestamps = sorted(self.bias_tracking.keys())
        if len(timestamps) < window_size * 2:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Compare recent window to previous window
        recent = timestamps[-window_size:]
        previous = timestamps[-(window_size * 2):-window_size]
        
        recent_bias = np.mean([self.bias_tracking[t]['bias'] for t in recent])
        previous_bias = np.mean([self.bias_tracking[t]['bias'] for t in previous])
        
        bias_drift = abs(recent_bias - previous_bias)
        recent_mae = np.mean([self.bias_tracking[t]['mae'] for t in recent])
        previous_mae = np.mean([self.bias_tracking[t]['mae'] for t in previous])
        mae_drift = abs(recent_mae - previous_mae)
        
        drift_detected = (bias_drift > threshold) or (mae_drift > threshold * 2)
        
        return {
            'drift_detected': drift_detected,
            'bias_drift': bias_drift,
            'mae_drift': mae_drift,
            'recent_bias': recent_bias,
            'previous_bias': previous_bias,
            'recent_mae': recent_mae,
            'previous_mae': previous_mae,
            'threshold': threshold,
        }
    
    def recommend_retrain(self, max_days: int = 30, mae_threshold: float = 12) -> bool:
        """
        Recommend if model should be retrained.
        
        Args:
            max_days: Maximum days since training
            mae_threshold: MAE threshold for retraining
            
        Returns:
            True if retraining recommended
        """
        # Check days since training
        days_since = self.model_health.get('days_since_training', 999)
        
        # Check recent MAE
        if self.bias_tracking:
            timestamps = sorted(self.bias_tracking.keys())
            if len(timestamps) >= 7:
                recent_mae = np.mean([self.bias_tracking[t]['mae'] for t in timestamps[-7:]])
                mae_too_high = recent_mae > mae_threshold
            else:
                mae_too_high = False
        else:
            mae_too_high = False
        
        # Check for drift
        drift_result = self.detect_drift()
        
        # Recommend retrain if any condition met
        recommend = (
            days_since > max_days or
            mae_too_high or
            drift_result['drift_detected']
        )
        
        reason = []
        if days_since > max_days:
            reason.append(f"Model is {days_since} days old (max: {max_days})")
        if mae_too_high:
            reason.append(f"Recent MAE too high ({recent_mae:.2f} > {mae_threshold})")
        if drift_result['drift_detected']:
            reason.append(f"Drift detected (bias: {drift_result['bias_drift']:.2f}, MAE: {drift_result['mae_drift']:.2f})")
        
        return {
            'recommend': recommend,
            'reasons': reason,
            'days_since_training': days_since,
            'recent_mae': recent_mae if self.bias_tracking else None,
            'drift_detected': drift_result['drift_detected'],
        }


if __name__ == '__main__':
    # Test monitor
    print("Testing ModelMonitor...")
    
    monitor = ModelMonitor('data/monitoring_test')
    
    # Simulate calibration data
    import random
    random.seed(42)
    
    for day in range(1, 11):
        timestamp = f"2026-01-{day:02d}T12:00:00"
        
        predictions = np.random.randn(100) * 5 + 220
        actuals = predictions + np.random.randn(100) * 2
        
        win_probs = np.random.rand(100)
        actual_winners = (np.random.rand(100) < win_probs).astype(int)
        
        monitor.record_calibration(timestamp, predictions, actuals, win_probs, actual_winners)
    
    # Generate report
    print("\n")
    print(monitor.get_calibration_report())
    
    # Detect drift
    drift = monitor.detect_drift()
    print(f"\nDrift detection:")
    for key, val in drift.items():
        print(f"  {key}: {val}")
    
    # Recommend retrain
    retrain = monitor.recommend_retrain()
    print(f"\nRetrain recommendation: {retrain['recommend']}")
    if retrain['reasons']:
        print("Reasons:")
        for reason in retrain['reasons']:
            print(f"  - {reason}")

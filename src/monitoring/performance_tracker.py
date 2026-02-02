"""Performance tracker for continuous model monitoring.

Tracks:
- Daily prediction accuracy
- Winner accuracy over time
- Total/margin prediction errors
- Calibration drift
- Model degradation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
from pathlib import Path
import json


class PerformanceTracker:
    """Track and analyze model performance over time."""
    
    def __init__(self, storage_dir: str = 'data/monitoring'):
        """
        Initialize performance tracker.
        
        Args:
            storage_dir: Directory to store performance data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.daily_file = self.storage_dir / 'daily_performance.csv'
        self.cumulative_file = self.storage_dir / 'cumulative_performance.json'
        self.alerts_file = self.storage_dir / 'performance_alerts.json'
        
        self.daily_records = []
        self.cumulative_stats = {}
        self.alerts = []
        
        self._load_data()
    
    def _load_data(self):
        """Load existing performance data."""
        # Load daily records
        if self.daily_file.exists():
            self.daily_records = pd.read_csv(self.daily_file).to_dict('records')
        else:
            self.daily_records = []
        
        # Load cumulative stats
        if self.cumulative_file.exists():
            with open(self.cumulative_file, 'r') as f:
                self.cumulative_stats = json.load(f)
        else:
            self.cumulative_stats = {
                'total_predictions': 0,
                'total_winner_correct': 0,
                'total_mae': 0.0,
                'total_rmse': 0.0,
                'model_versions': [],
            }
        
        # Load alerts
        if self.alerts_file.exists():
            with open(self.alerts_file, 'r') as f:
                self.alerts = json.load(f)
        else:
            self.alerts = []
    
    def record_daily_performance(
        self,
        date: str,
        predictions: List[Dict[str, any]],
        model_version: str
    ):
        """
        Record performance for a day.
        
        Args:
            date: Date string (YYYY-MM-DD)
            predictions: List of prediction dictionaries
            model_version: Model version used
        """
        if not predictions:
            return
        
        # Calculate metrics for the day
        total_correct = sum(p['winner_correct'] for p in predictions)
        total_predictions = len(predictions)
        
        total_errors = [p['total_error'] for p in predictions if 'total_error' in p]
        margin_errors = [p['margin_error'] for p in predictions if 'margin_error' in p]
        
        total_mae = np.mean(np.abs(total_errors)) if total_errors else 0
        total_rmse = np.sqrt(np.mean(np.array(total_errors) ** 2)) if total_errors else 0
        margin_mae = np.mean(np.abs(margin_errors)) if margin_errors else 0
        
        winner_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        record = {
            'date': date,
            'model_version': model_version,
            'predictions': total_predictions,
            'winner_correct': total_correct,
            'winner_accuracy': winner_accuracy,
            'total_mae': total_mae,
            'total_rmse': total_rmse,
            'margin_mae': margin_mae,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.daily_records.append(record)
        self._save_daily_records()
        
        # Update cumulative stats
        self.cumulative_stats['total_predictions'] += total_predictions
        self.cumulative_stats['total_winner_correct'] += total_correct
        self.cumulative_stats['total_mae'] = (
            self.cumulative_stats['total_mae'] * self.cumulative_stats['total_predictions'] + total_mae * total_predictions
        ) / (self.cumulative_stats['total_predictions'] + total_predictions) if self.cumulative_stats['total_predictions'] + total_predictions > 0 else 0
        
        if model_version not in self.cumulative_stats['model_versions']:
            self.cumulative_stats['model_versions'].append(model_version)
        
        self._save_cumulative_stats()
        
        # Check for alerts
        self._check_for_alerts(record)
        
        print(f"Recorded performance for {date}: {total_predictions} predictions, {winner_accuracy:.1%} winner accuracy")
    
    def _save_daily_records(self):
        """Save daily performance records."""
        df = pd.DataFrame(self.daily_records)
        df.to_csv(self.daily_file, index=False)
    
    def _save_cumulative_stats(self):
        """Save cumulative statistics."""
        with open(self.cumulative_file, 'w') as f:
            json.dump(self.cumulative_stats, f, indent=2)
    
    def _check_for_alerts(self, record: Dict[str, any]):
        """
        Check for performance alerts.
        
        Args:
            record: Daily performance record
        """
        alerts = []
        
        # Alert 1: Winner accuracy below threshold
        if record['winner_accuracy'] < 0.5:
            alerts.append({
                'type': 'low_winner_accuracy',
                'date': record['date'],
                'value': record['winner_accuracy'],
                'threshold': 0.5,
                'message': f"Winner accuracy ({record['winner_accuracy']:.1%}) below 50% threshold",
            })
        
        # Alert 2: Total MAE above threshold
        if record['total_mae'] > 15:
            alerts.append({
                'type': 'high_total_mae',
                'date': record['date'],
                'value': record['total_mae'],
                'threshold': 15,
                'message': f"Total MAE ({record['total_mae']:.2f}) above 15 point threshold",
            })
        
        # Alert 3: Performance degradation (compare to 7-day average)
        if len(self.daily_records) >= 7:
            recent_7_days = self.daily_records[-7:]
            avg_acc = np.mean([r['winner_accuracy'] for r in recent_7_days])
            if record['winner_accuracy'] < avg_acc - 0.1:
                alerts.append({
                    'type': 'performance_degradation',
                    'date': record['date'],
                    'value': record['winner_accuracy'],
                    'average': avg_acc,
                    'message': f"Winner accuracy dropped {avg_acc - record['winner_accuracy']:.1%} below 7-day average",
                })
        
        # Save new alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            self.alerts.append(alert)
        
        if alerts:
            self._save_alerts()
    
    def _save_alerts(self):
        """Save performance alerts."""
        with open(self.alerts_file, 'w') as f:
            json.dump(self.alerts, f, indent=2)
    
    def get_cumulative_stats(self) -> Dict[str, any]:
        """
        Get cumulative performance statistics.
        
        Returns:
            Dictionary with cumulative stats
        """
        return self.cumulative_stats
    
    def get_recent_performance(self, n_days: int = 7) -> pd.DataFrame:
        """
        Get performance for last N days.
        
        Args:
            n_days: Number of recent days to return
            
        Returns:
            DataFrame with recent performance
        """
        df = pd.DataFrame(self.daily_records)
        return df.tail(n_days)
    
    def get_performance_trend(self, metric: str = 'winner_accuracy', window: int = 7) -> Dict[str, float]:
        """
        Get performance trend for a metric.
        
        Args:
            metric: Metric to analyze
            window: Window size for trend calculation
            
        Returns:
            Dictionary with trend metrics
        """
        df = pd.DataFrame(self.daily_records)
        
        if len(df) < window:
            return {'trend': 0.0, 'current': 0.0, 'average': 0.0}
        
        recent = df.tail(window)
        current = recent[metric].iloc[-1]
        average = recent[metric].mean()
        trend = current - average
        
        return {
            'trend': trend,
            'current': current,
            'average': average,
            'window': window,
        }
    
    def generate_report(self) -> str:
        """
        Generate performance report.
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE MONITORING REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Cumulative stats
        lines.append("CUMULATIVE STATISTICS:")
        total_preds = self.cumulative_stats['total_predictions']
        total_correct = self.cumulative_stats['total_winner_correct']
        overall_acc = total_correct / total_preds if total_preds > 0 else 0
        
        lines.append(f"  Total Predictions: {total_preds}")
        lines.append(f"  Overall Winner Accuracy: {overall_acc:.1%}")
        lines.append(f"  Cumulative Total MAE: {self.cumulative_stats['total_mae']:.2f}")
        lines.append("")
        
        # Recent performance
        if len(self.daily_records) >= 7:
            lines.append("LAST 7 DAYS:")
            df = pd.DataFrame(self.daily_records)
            recent = df.tail(7)
            
            recent_acc = recent['winner_accuracy'].mean()
            recent_mae = recent['total_mae'].mean()
            recent_rmse = recent['total_rmse'].mean()
            
            lines.append(f"  Average Winner Accuracy: {recent_acc:.1%}")
            lines.append(f"  Average Total MAE: {recent_mae:.2f}")
            lines.append(f"  Average Total RMSE: {recent_rmse:.2f}")
            lines.append("")
        
        # Trend
        trend = self.get_performance_trend('winner_accuracy', 7)
        lines.append("WINNER ACCURACY TREND (7-day):")
        lines.append(f"  Current: {trend['current']:.1%}")
        lines.append(f"  Average: {trend['average']:.1%}")
        lines.append(f"  Trend: {trend['trend']:+.1%}")
        lines.append("")
        
        # Alerts
        if self.alerts:
            lines.append("ACTIVE ALERTS:")
            for alert in self.alerts[-5:]:  # Last 5 alerts
                lines.append(f"  [{alert['date']}] {alert['message']}")
            lines.append("")
        else:
            lines.append("No active alerts")
            lines.append("")
        
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def plot_performance_trend(self, save_path: Optional[str] = None):
        """
        Plot performance trend (requires matplotlib).
        
        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return
        
        df = pd.DataFrame(self.daily_records)
        
        if len(df) == 0:
            print("No data to plot.")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Winner accuracy
        axes[0].plot(df['date'], df['winner_accuracy'], marker='o', linewidth=2)
        axes[0].axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
        axes[0].set_ylabel('Winner Accuracy')
        axes[0].set_title('Winner Accuracy Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Total MAE
        axes[1].plot(df['date'], df['total_mae'], marker='s', linewidth=2, color='orange')
        axes[1].axhline(y=15, color='r', linestyle='--', label='Threshold')
        axes[1].set_ylabel('Total MAE')
        axes[1].set_xlabel('Date')
        axes[1].set_title('Total MAE Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


if __name__ == '__main__':
    # Test tracker
    print("Testing PerformanceTracker...")
    
    tracker = PerformanceTracker('data/monitoring_test')
    
    # Simulate 10 days of predictions
    import random
    random.seed(42)
    
    for day in range(1, 11):
        date_str = f"2026-01-{day:02d}"
        
        predictions = []
        for i in range(random.randint(5, 15)):
            predictions.append({
                'game_id': f'00{day}{i:04d}',
                'home': 'LAL' if random.random() > 0.5 else 'BOS',
                'away': 'BOS' if random.random() > 0.5 else 'LAL',
                'winner_correct': random.random() > 0.4,
                'total_error': random.gauss(0, 10),
                'margin_error': random.gauss(0, 8),
            })
        
        tracker.record_daily_performance(date_str, predictions, 'v1.0')
    
    # Generate report
    print("\n")
    print(tracker.generate_report())
    
    # Get recent performance
    recent = tracker.get_recent_performance(5)
    print(f"\nRecent 5 days performance:")
    print(recent[['date', 'predictions', 'winner_accuracy', 'total_mae']])
    
    # Get trend
    trend = tracker.get_performance_trend('winner_accuracy', 7)
    print(f"\nWinner accuracy trend: {trend['trend']:+.1%}")

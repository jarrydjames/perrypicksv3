"""
Block bootstrap for time-valid confidence intervals.

Implements block bootstrap to compute confidence intervals for paired loss
differials between models. Preserves time-series structure by
sampling contiguous blocks.

Reference: execution_specification_for_statistically_valid_nba_forecasting_system.md Section 5.3
"""

import numpy as np
from typing import Tuple, Optional


def block_bootstrap(
    loss_differentials: np.ndarray,
    block_size: int = 200,
    n_bootstraps: int = 1000,
    random_state: Optional[int] = None,
) -> dict:
    """
    Perform block bootstrap on loss differentials.
    
    Args:
        loss_differentials: Array of per-game loss differentials (L_new - L_baseline)
                            Shape: (n_games,)
        block_size: Size of contiguous blocks to sample (default: 200)
        n_bootstraps: Number of bootstrap samples (default: 1000)
        random_state: Random seed for reproducibility (optional)
    
    Returns:
        Dictionary with:
        - mean_diff: Mean of loss differentials
        - ci_lower: Lower bound of 95% CI
        - ci_upper: Upper bound of 95% CI
        - p_improvement: Probability that mean(diff) < 0
        - bootstrap_samples: Array of bootstrap sample means
        - block_size: Block size used
        - n_bootstraps: Number of bootstraps performed
    """
    n = len(loss_differentials)
    
    if n == 0:
        return {
            "mean_diff": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_improvement": np.nan,
            "bootstrap_samples": np.array([]),
            "block_size": block_size,
            "n_bootstraps": n_bootstraps,
            "error": "No loss differentials provided",
        }
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Compute number of blocks
    n_blocks = int(np.ceil(n / block_size))
    
    # Perform block bootstrap
    bootstrap_means = np.zeros(n_bootstraps)
    
    for i in range(n_bootstraps):
        # Sample block indices with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        
        # Expand block indices to sample indices
        sample_indices = []
        for block_idx in block_indices:
            start = block_idx * block_size
            end = min(start + block_size, n)
            sample_indices.extend(range(start, end))
        
        # Trim to original length
        sample_indices = sample_indices[:n]
        
        # Sample loss differentials
        sampled_losses = loss_differentials[sample_indices]
        
        # Compute mean of sampled losses
        bootstrap_means[i] = np.mean(sampled_losses)
    
    # Compute statistics
    mean_diff = np.mean(bootstrap_means)
    ci_lower = np.percentile(bootstrap_means, 2.5)  # 95% CI (2.5th percentile)
    ci_upper = np.percentile(bootstrap_means, 97.5)  # 97.5th percentile
    p_improvement = np.mean(bootstrap_means < 0)
    
    return {
        "mean_diff": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_improvement": float(p_improvement),
        "bootstrap_samples": bootstrap_means,
        "block_size": block_size,
        "n_bootstraps": n_bootstraps,
        "n_games": n,
    }


def block_bootstrap_summary(bootstrap_results: dict) -> str:
    """
    Generate human-readable summary of block bootstrap results.
    """
    lines = [
        "=" * 80,
        "BLOCK BOOTSTRAP RESULTS",
        "=" * 80,
        "",
        f"Mean Loss Differential: {bootstrap_results['mean_diff']:.4f}",
        f"95% Confidence Interval: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]",
        f"Probability of Improvement: {bootstrap_results['p_improvement']:.2%}",
        "",
        f"Bootstrap Parameters:",
        f"  Block Size: {bootstrap_results['block_size']}",
        f"  Number of Bootstraps: {bootstrap_results['n_bootstraps']}",
        f"  Number of Games: {bootstrap_results['n_games']}",
        "",
        "=" * 80,
    ]
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test block bootstrap
    np.random.seed(42)
    
    # Create sample loss differentials (100 games, 0.5 average improvement)
    n = 100
    loss_differentials = np.random.normal(loc=-0.5, scale=1.0, size=n)
    
    print("Testing block bootstrap with sample data...")
    print(f"Loss differentials: {loss_differentials[:10]}...")
    
    results = block_bootstrap(loss_differentials, block_size=20, n_bootstraps=1000)
    print(block_bootstrap_summary(results))

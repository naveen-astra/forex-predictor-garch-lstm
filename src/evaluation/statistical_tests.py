"""
Statistical Tests for Model Comparison

This module implements statistical tests for comparing forecast accuracy,
specifically designed for financial time series evaluation.

Key Functions:
    - diebold_mariano_test(): Compare forecast accuracy statistically
    - regime_analysis(): Evaluate performance across volatility regimes
    - directional_accuracy_test(): Test significance of directional predictions

Author: Research Team
Date: January 2026
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional
from pathlib import Path


def diebold_mariano_test(
    errors_model1: np.ndarray,
    errors_model2: np.ndarray,
    h: int = 1,
    power: int = 2
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests the null hypothesis that two forecasting methods have equal accuracy.
    
    Args:
        errors_model1: Forecast errors from model 1 (actual - predicted)
        errors_model2: Forecast errors from model 2 (actual - predicted)
        h: Forecast horizon (default=1 for one-step ahead)
        power: Power parameter for loss function (default=2 for squared errors)
        
    Returns:
        Tuple of (DM statistic, p-value)
        
    Interpretation:
        - DM > 0: Model 2 is more accurate than Model 1
        - DM < 0: Model 1 is more accurate than Model 2
        - p-value < 0.05: Reject null hypothesis (significant difference at 5%)
        - p-value >= 0.05: Fail to reject null (no significant difference)
        
    Reference:
        Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
        Journal of Business & Economic Statistics, 13(3), 253-263.
        
    Notes:
        - Assumes forecast errors are from same test set
        - Valid for one-step ahead forecasts (h=1)
        - Uses two-tailed test
        - Adjusts for small sample sizes
    """
    # Ensure inputs are numpy arrays
    errors_model1 = np.asarray(errors_model1).flatten()
    errors_model2 = np.asarray(errors_model2).flatten()
    
    # Verify same length
    if len(errors_model1) != len(errors_model2):
        raise ValueError(f"Error arrays must have same length. "
                        f"Got {len(errors_model1)} and {len(errors_model2)}")
    
    n = len(errors_model1)
    
    # Calculate loss differential
    # loss_diff > 0 means model2 has larger loss (model1 is better)
    loss_model1 = np.abs(errors_model1) ** power
    loss_model2 = np.abs(errors_model2) ** power
    loss_diff = loss_model1 - loss_model2
    
    # Mean of loss differential
    mean_diff = np.mean(loss_diff)
    
    # Calculate variance of loss differential
    # Account for autocorrelation in forecast errors
    variance = np.var(loss_diff, ddof=1)
    
    # For h-step ahead forecasts, adjust for autocorrelation
    # Here we use h=1 (one-step ahead), so no adjustment needed
    if h > 1:
        # Harvey-Leybourne-Newbold (1997) adjustment
        variance = variance * (1 + 2 * sum([(h - k) / h for k in range(1, h)]))
    
    # Standard error
    std_error = np.sqrt(variance / n)
    
    # DM statistic
    if std_error == 0:
        # If no variance, cannot compute test
        return 0.0, 1.0
    
    dm_stat = mean_diff / std_error
    
    # Two-tailed p-value (using standard normal approximation)
    # For large samples, DM ~ N(0,1) under null hypothesis
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    # Small sample adjustment (Harvey, Leybourne, Newbold, 1997)
    # For n < 100, use t-distribution instead of normal
    if n < 100:
        dm_stat_adjusted = dm_stat * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_adjusted), n - 1))
        dm_stat = dm_stat_adjusted
    
    return dm_stat, p_value


def interpret_dm_test(dm_stat: float, p_value: float, 
                     model1_name: str, model2_name: str,
                     alpha: float = 0.05) -> str:
    """
    Interpret Diebold-Mariano test results.
    
    Args:
        dm_stat: DM statistic
        p_value: p-value from test
        model1_name: Name of first model
        model2_name: Name of second model
        alpha: Significance level (default=0.05)
        
    Returns:
        Human-readable interpretation string
    """
    interpretation = []
    
    # Statistical significance
    if p_value < alpha:
        interpretation.append(f"✓ Statistically significant at {alpha*100:.0f}% level (p = {p_value:.4f})")
    else:
        interpretation.append(f"✗ Not statistically significant at {alpha*100:.0f}% level (p = {p_value:.4f})")
    
    # Direction of difference
    if dm_stat > 0:
        interpretation.append(f"→ {model2_name} is more accurate than {model1_name}")
        if p_value < alpha:
            interpretation.append(f"   Conclusion: {model2_name} significantly outperforms {model1_name}")
        else:
            interpretation.append(f"   Conclusion: Difference not statistically significant")
    elif dm_stat < 0:
        interpretation.append(f"→ {model1_name} is more accurate than {model2_name}")
        if p_value < alpha:
            interpretation.append(f"   Conclusion: {model1_name} significantly outperforms {model2_name}")
        else:
            interpretation.append(f"   Conclusion: Difference not statistically significant")
    else:
        interpretation.append(f"→ Models have equal accuracy")
    
    # Effect size
    abs_dm = abs(dm_stat)
    if abs_dm < 1.0:
        interpretation.append("   Effect size: Small")
    elif abs_dm < 2.0:
        interpretation.append("   Effect size: Moderate")
    else:
        interpretation.append("   Effect size: Large")
    
    return "\n".join(interpretation)


def regime_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    volatility: np.ndarray,
    model_name: str = "Model",
    n_regimes: int = 3
) -> pd.DataFrame:
    """
    Analyze model performance across volatility regimes.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        volatility: GARCH conditional volatility
        model_name: Name of model for results
        n_regimes: Number of volatility regimes (default=3: low/medium/high)
        
    Returns:
        DataFrame with performance metrics by regime
        
    Notes:
        - Regimes defined by volatility percentiles
        - Low: 0-33rd percentile
        - Medium: 33rd-67th percentile  
        - High: 67th-100th percentile
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Ensure same length
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    volatility = np.asarray(volatility).flatten()
    
    min_len = min(len(y_true), len(y_pred), len(volatility))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    volatility = volatility[:min_len]
    
    # Define regime boundaries
    if n_regimes == 2:
        thresholds = [np.percentile(volatility, 50)]
        regime_names = ['Low Volatility', 'High Volatility']
    elif n_regimes == 3:
        thresholds = [np.percentile(volatility, 33.33), np.percentile(volatility, 66.67)]
        regime_names = ['Low Volatility', 'Medium Volatility', 'High Volatility']
    else:
        raise ValueError("n_regimes must be 2 or 3")
    
    # Assign regimes
    regimes = np.zeros(len(volatility), dtype=int)
    for i, threshold in enumerate(thresholds):
        regimes[volatility > threshold] = i + 1
    
    # Calculate metrics for each regime
    results = []
    for regime_idx, regime_name in enumerate(regime_names):
        mask = regimes == regime_idx
        if np.sum(mask) == 0:
            continue
            
        y_true_regime = y_true[mask]
        y_pred_regime = y_pred[mask]
        
        # Calculate metrics
        mse = mean_squared_error(y_true_regime, y_pred_regime)
        mae = mean_absolute_error(y_true_regime, y_pred_regime)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        actual_direction = np.sign(y_true_regime[1:])
        pred_direction = np.sign(y_pred_regime[1:])
        dir_acc = np.mean(actual_direction == pred_direction) * 100
        
        results.append({
            'Regime': regime_name,
            'Model': model_name,
            'N_Observations': np.sum(mask),
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Directional_Accuracy': dir_acc,
            'Volatility_Range': f"{volatility[mask].min():.6f} - {volatility[mask].max():.6f}"
        })
    
    return pd.DataFrame(results)


def directional_accuracy_test(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Test if directional accuracy is significantly better than random (50%).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple of (directional accuracy %, p-value)
        
    Notes:
        - Uses binomial test
        - Null hypothesis: accuracy = 50% (random guess)
        - Alternative: accuracy > 50% (better than random)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Calculate directional movements
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Count correct predictions
    correct = np.sum(actual_direction == pred_direction)
    total = len(actual_direction)
    
    accuracy = (correct / total) * 100
    
    # Binomial test: H0: p = 0.5, H1: p > 0.5
    p_value = stats.binom_test(correct, total, 0.5, alternative='greater')
    
    return accuracy, p_value


def calculate_confidence_interval(
    errors: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for forecast errors.
    
    Args:
        errors: Forecast errors
        confidence: Confidence level (default=0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    errors = np.asarray(errors).flatten()
    mean = np.mean(errors)
    std_error = stats.sem(errors)
    
    # Use t-distribution for small samples
    df = len(errors) - 1
    t_value = stats.t.ppf((1 + confidence) / 2, df)
    
    margin_of_error = t_value * std_error
    lower = mean - margin_of_error
    upper = mean + margin_of_error
    
    return mean, lower, upper


def compare_all_models(
    y_true: np.ndarray,
    y_pred_garch: np.ndarray,
    y_pred_lstm: np.ndarray,
    y_pred_hybrid: np.ndarray
) -> pd.DataFrame:
    """
    Comprehensive pairwise comparison of all models using DM tests.
    
    Args:
        y_true: True values
        y_pred_garch: GARCH predictions
        y_pred_lstm: LSTM predictions
        y_pred_hybrid: Hybrid predictions
        
    Returns:
        DataFrame with all pairwise DM test results
    """
    # Calculate errors
    errors_garch = y_true - y_pred_garch
    errors_lstm = y_true - y_pred_lstm
    errors_hybrid = y_true - y_pred_hybrid
    
    comparisons = []
    
    # Hybrid vs GARCH
    dm_stat, p_value = diebold_mariano_test(errors_hybrid, errors_garch)
    comparisons.append({
        'Comparison': 'Hybrid vs GARCH',
        'DM_Statistic': dm_stat,
        'P_Value': p_value,
        'Significant_at_5%': 'Yes' if p_value < 0.05 else 'No',
        'Winner': 'Hybrid' if dm_stat < 0 else 'GARCH' if dm_stat > 0 else 'Tie'
    })
    
    # Hybrid vs LSTM
    dm_stat, p_value = diebold_mariano_test(errors_hybrid, errors_lstm)
    comparisons.append({
        'Comparison': 'Hybrid vs LSTM',
        'DM_Statistic': dm_stat,
        'P_Value': p_value,
        'Significant_at_5%': 'Yes' if p_value < 0.05 else 'No',
        'Winner': 'Hybrid' if dm_stat < 0 else 'LSTM' if dm_stat > 0 else 'Tie'
    })
    
    # LSTM vs GARCH
    dm_stat, p_value = diebold_mariano_test(errors_lstm, errors_garch)
    comparisons.append({
        'Comparison': 'LSTM vs GARCH',
        'DM_Statistic': dm_stat,
        'P_Value': p_value,
        'Significant_at_5%': 'Yes' if p_value < 0.05 else 'No',
        'Winner': 'LSTM' if dm_stat < 0 else 'GARCH' if dm_stat > 0 else 'Tie'
    })
    
    return pd.DataFrame(comparisons)


if __name__ == "__main__":
    print("Statistical Tests Module for Model Comparison")
    print("=" * 70)
    print("Available functions:")
    print("  - diebold_mariano_test(): Compare forecast accuracy")
    print("  - interpret_dm_test(): Human-readable interpretation")
    print("  - regime_analysis(): Performance across volatility regimes")
    print("  - directional_accuracy_test(): Test directional predictions")
    print("  - compare_all_models(): Pairwise DM tests for all models")
    print()
    print("Example usage:")
    print("  from src.evaluation.statistical_tests import diebold_mariano_test")
    print("  dm_stat, p_value = diebold_mariano_test(errors1, errors2)")

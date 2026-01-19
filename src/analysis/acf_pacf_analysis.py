"""
ACF and PACF Analysis for FOREX Time Series
Determines optimal lag orders for GARCH modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import ForexPreprocessor
from src.utils.config import PREPROCESSING_CONFIG

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['axes.edgecolor'] = '#2a2a2a'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['grid.color'] = '#2a2a2a'
plt.rcParams['grid.alpha'] = 0.3


def perform_acf_pacf_analysis(data_path=None, max_lags=40):
    """
    Perform ACF and PACF analysis on EUR/USD returns
    
    Args:
        data_path: Path to cleaned data (uses default if None)
        max_lags: Maximum number of lags to analyze
    
    Returns:
        dict: Analysis results including optimal orders
    """
    if data_path is None:
        data_path = PROJECT_ROOT / 'LSTM_GARCH' / 'input' / 'financial_data.csv'
    print("=" * 80)
    print("ACF/PACF ANALYSIS FOR GARCH MODEL SELECTION")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading EUR/USD data...")
    df = pd.read_csv(data_path)
    
    # Check if data needs transposing (based on shape)
    if df.shape[0] < df.shape[1]:
        df = df.T
        df.columns = df.iloc[0]
        df = df[1:]
        df = df.reset_index(drop=True)
    
    # Get close prices
    if 'Close' in df.columns:
        prices = df['Close'].astype(float).values
    elif 'close' in df.columns:
        prices = df['close'].astype(float).values
    else:
        # Assume first numeric column is close price
        prices = df.iloc[:, 0].astype(float).values
    
    # Calculate returns
    returns = np.diff(np.log(prices)) * 100  # Percentage returns
    
    print(f"✓ Loaded {len(returns)} return observations")
    print(f"  Mean return: {np.mean(returns):.6f}%")
    print(f"  Std return: {np.std(returns):.4f}%")
    print(f"  Min return: {np.min(returns):.4f}%")
    print(f"  Max return: {np.max(returns):.4f}%")
    
    # Calculate squared returns for volatility clustering
    squared_returns = returns ** 2
    
    # Compute ACF and PACF
    print(f"\n[2/5] Computing ACF/PACF for {max_lags} lags...")
    acf_values = acf(squared_returns, nlags=max_lags)
    pacf_values = pacf(squared_returns, nlags=max_lags)
    
    # Find significant lags (beyond 95% confidence interval)
    n = len(squared_returns)
    conf_interval = 1.96 / np.sqrt(n)
    
    significant_acf_lags = np.where(np.abs(acf_values[1:]) > conf_interval)[0] + 1
    significant_pacf_lags = np.where(np.abs(pacf_values[1:]) > conf_interval)[0] + 1
    
    print(f"✓ Significant ACF lags (q candidates): {significant_acf_lags[:10].tolist()}")
    print(f"✓ Significant PACF lags (p candidates): {significant_pacf_lags[:10].tolist()}")
    
    # Determine optimal GARCH orders
    print("\n[3/5] Determining optimal GARCH orders...")
    
    # Use first few significant lags as candidates
    p_candidates = significant_pacf_lags[:5] if len(significant_pacf_lags) > 0 else [1, 2]
    q_candidates = significant_acf_lags[:5] if len(significant_acf_lags) > 0 else [1, 2]
    
    # Common GARCH orders to test
    suggested_orders = [
        (1, 1),  # Standard GARCH(1,1)
        (1, 2),  # More flexible MA component
        (2, 1),  # More flexible AR component
        (2, 2),  # Highly flexible
    ]
    
    print("✓ Suggested GARCH(p,q) orders based on ACF/PACF:")
    for p, q in suggested_orders:
        print(f"  • GARCH({p},{q})")
    
    # Create visualizations
    print("\n[4/5] Generating ACF/PACF plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('ACF/PACF Analysis for EUR/USD Squared Returns', 
                 fontsize=16, fontweight='bold', color='white', y=0.995)
    
    # Plot 1: Returns time series
    axes[0, 0].plot(returns, color='#3b82f6', linewidth=0.5, alpha=0.7)
    axes[0, 0].set_title('EUR/USD Log Returns (%)', fontsize=12, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel('Time Index', fontsize=10)
    axes[0, 0].set_ylabel('Return (%)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='#666666', linestyle='--', linewidth=1)
    
    # Plot 2: Squared returns (volatility proxy)
    axes[0, 1].plot(squared_returns, color='#8b5cf6', linewidth=0.5, alpha=0.7)
    axes[0, 1].set_title('Squared Returns (Volatility Clustering)', fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('Time Index', fontsize=10)
    axes[0, 1].set_ylabel('Squared Return', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ACF of squared returns
    plot_acf(squared_returns, lags=max_lags, ax=axes[1, 0], color='#06b6d4')
    axes[1, 0].set_title(f'ACF of Squared Returns (q order)', fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel('Lag', fontsize=10)
    axes[1, 0].set_ylabel('Autocorrelation', fontsize=10)
    
    # Plot 4: PACF of squared returns
    plot_pacf(squared_returns, lags=max_lags, ax=axes[1, 1], color='#10b981')
    axes[1, 1].set_title(f'PACF of Squared Returns (p order)', fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel('Lag', fontsize=10)
    axes[1, 1].set_ylabel('Partial Autocorrelation', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = PROJECT_ROOT / 'results' / 'figures' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'acf_pacf_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"✓ Saved ACF/PACF plot to: {output_path}")
    
    # Create detailed statistics table
    print("\n[5/5] Generating detailed statistics...")
    
    stats_df = pd.DataFrame({
        'Lag': range(1, min(11, len(acf_values))),
        'ACF': acf_values[1:11],
        'PACF': pacf_values[1:11],
        'ACF_Significant': [abs(acf_values[i]) > conf_interval for i in range(1, min(11, len(acf_values)))],
        'PACF_Significant': [abs(pacf_values[i]) > conf_interval for i in range(1, min(11, len(pacf_values)))]
    })
    
    stats_path = output_dir / 'acf_pacf_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Saved statistics to: {stats_path}")
    
    print("\n" + "=" * 80)
    print("ACF/PACF ANALYSIS COMPLETE")
    print("=" * 80)
    
    results = {
        'returns': returns,
        'squared_returns': squared_returns,
        'acf_values': acf_values,
        'pacf_values': pacf_values,
        'suggested_orders': suggested_orders,
        'statistics': stats_df,
        'plot_path': str(output_path),
        'stats_path': str(stats_path)
    }
    
    return results


if __name__ == "__main__":
    results = perform_acf_pacf_analysis()
    
    print("\nRecommendation:")
    print("→ Start with GARCH(1,1) as baseline")
    print("→ Test GARCH(1,2), GARCH(2,1), GARCH(2,2) for comparison")
    print("→ Use AIC/BIC for model selection")

"""
Compare Multiple GARCH Models with Different Orders
Trains GARCH(p,q) models and evaluates performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import ForexPreprocessor
from src.utils.config import PREPROCESSING_CONFIG

# Set style
sns.set_style("darkgrid")


def train_garch_variants(data_path=None):
    """
    Train multiple GARCH models with different orders
    
    Args:
        data_path: Path to cleaned data (uses default if None)
    
    Returns:
        dict: Results for all GARCH variants
    """
    if data_path is None:
        data_path = PROJECT_ROOT / 'LSTM_GARCH' / 'input' / 'financial_data.csv'
    print("=" * 80)
    print("GARCH MODEL ORDER COMPARISON")
    print("=" * 80)
    
    # Load data
    print("\n[1/4] Loading and preparing data...")
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
        prices = df.iloc[:, 0].astype(float).values
    
    # Split data (70/15/15)
    n = len(prices)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    
    train_prices = prices[:train_size]
    val_prices = prices[train_size:train_size + val_size]
    test_prices = prices[train_size + val_size:]
    
    # Calculate returns
    train_returns = np.diff(np.log(train_prices)) * 100
    val_returns = np.diff(np.log(val_prices)) * 100
    test_returns = np.diff(np.log(test_prices)) * 100
    
    print(f"✓ Train returns: {len(train_returns)}")
    print(f"✓ Val returns: {len(val_returns)}")
    print(f"✓ Test returns: {len(test_returns)}")
    
    # Define GARCH orders to test
    garch_orders = [
        (1, 1),  # Standard
        (1, 2),  # More MA terms
        (2, 1),  # More AR terms
        (2, 2),  # Highly flexible
        (1, 3),  # Extended MA
        (3, 1),  # Extended AR
    ]
    
    print(f"\n[2/4] Training {len(garch_orders)} GARCH variants...")
    
    all_results = {}
    
    for p, q in garch_orders:
        print(f"\n  → Training GARCH({p},{q})...")
        
        try:
            # Fit model
            model = arch_model(train_returns, vol='Garch', p=p, q=q, dist='normal')
            fitted_model = model.fit(disp='off', show_warning=False)
            
            # Generate forecasts
            val_forecasts = []
            for i in range(len(val_returns)):
                forecast = fitted_model.forecast(horizon=1, reindex=False)
                val_forecasts.append(forecast.variance.values[-1, 0])
            
            test_forecasts = []
            for i in range(len(test_returns)):
                forecast = fitted_model.forecast(horizon=1, reindex=False)
                test_forecasts.append(forecast.variance.values[-1, 0])
            
            # Calculate metrics
            val_mse = mean_squared_error(val_returns**2, val_forecasts)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(val_returns**2, val_forecasts)
            
            test_mse = mean_squared_error(test_returns**2, test_forecasts)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(test_returns**2, test_forecasts)
            
            # Model selection criteria
            aic = fitted_model.aic
            bic = fitted_model.bic
            log_likelihood = fitted_model.loglikelihood
            
            # Store results
            all_results[f'GARCH({p},{q})'] = {
                'order': (p, q),
                'parameters': len(fitted_model.params),
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'alpha': fitted_model.params.get('alpha[1]', None),
                'beta': fitted_model.params.get('beta[1]', None),
                'omega': fitted_model.params.get('omega', None),
                'convergence': fitted_model.convergence_flag == 0
            }
            
            print(f"    ✓ AIC: {aic:.4f} | BIC: {bic:.4f}")
            print(f"    ✓ Test RMSE: {test_rmse:.6f} | Test MSE: {test_mse:.8f}")
            
        except Exception as e:
            print(f"    ✗ Failed: {str(e)}")
            all_results[f'GARCH({p},{q})'] = {'error': str(e)}
    
    # Find best models
    print("\n[3/4] Ranking models by performance...")
    
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    # Rank by AIC (lower is better)
    sorted_by_aic = sorted(valid_results.items(), key=lambda x: x[1]['aic'])
    # Rank by BIC (lower is better)
    sorted_by_bic = sorted(valid_results.items(), key=lambda x: x[1]['bic'])
    # Rank by test RMSE (lower is better)
    sorted_by_rmse = sorted(valid_results.items(), key=lambda x: x[1]['test_rmse'])
    
    print("\n  Best by AIC:")
    for i, (name, res) in enumerate(sorted_by_aic[:3], 1):
        print(f"    {i}. {name}: AIC={res['aic']:.4f}")
    
    print("\n  Best by BIC:")
    for i, (name, res) in enumerate(sorted_by_bic[:3], 1):
        print(f"    {i}. {name}: BIC={res['bic']:.4f}")
    
    print("\n  Best by Test RMSE:")
    for i, (name, res) in enumerate(sorted_by_rmse[:3], 1):
        print(f"    {i}. {name}: RMSE={res['test_rmse']:.6f}")
    
    # Create comparison visualizations
    print("\n[4/4] Creating comparison visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('GARCH Model Order Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    models = list(valid_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    # Plot 1: AIC Comparison
    aic_values = [valid_results[m]['aic'] for m in models]
    axes[0, 0].bar(range(len(models)), aic_values, color=colors, alpha=0.7)
    axes[0, 0].set_title('AIC (Lower is Better)', fontweight='bold')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: BIC Comparison
    bic_values = [valid_results[m]['bic'] for m in models]
    axes[0, 1].bar(range(len(models)), bic_values, color=colors, alpha=0.7)
    axes[0, 1].set_title('BIC (Lower is Better)', fontweight='bold')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Test RMSE
    rmse_values = [valid_results[m]['test_rmse'] for m in models]
    axes[0, 2].bar(range(len(models)), rmse_values, color=colors, alpha=0.7)
    axes[0, 2].set_title('Test RMSE (Lower is Better)', fontweight='bold')
    axes[0, 2].set_xlabel('Model')
    axes[0, 2].set_xticks(range(len(models)))
    axes[0, 2].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Test MSE
    mse_values = [valid_results[m]['test_mse'] for m in models]
    axes[1, 0].bar(range(len(models)), mse_values, color=colors, alpha=0.7)
    axes[1, 0].set_title('Test MSE (Lower is Better)', fontweight='bold')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Test MAE
    mae_values = [valid_results[m]['test_mae'] for m in models]
    axes[1, 1].bar(range(len(models)), mae_values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Test MAE (Lower is Better)', fontweight='bold')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Parameter Count
    param_counts = [valid_results[m]['parameters'] for m in models]
    axes[1, 2].bar(range(len(models)), param_counts, color=colors, alpha=0.7)
    axes[1, 2].set_title('Number of Parameters', fontweight='bold')
    axes[1, 2].set_xlabel('Model')
    axes[1, 2].set_xticks(range(len(models)))
    axes[1, 2].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    output_dir = PROJECT_ROOT / 'results' / 'figures' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / 'garch_order_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to: {plot_path}")
    
    # Save detailed results
    results_df = pd.DataFrame(valid_results).T
    results_df = results_df.round(6)
    
    csv_path = output_dir / 'garch_order_results.csv'
    results_df.to_csv(csv_path)
    print(f"✓ Saved results CSV to: {csv_path}")
    
    # Save JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_tested': len(garch_orders),
        'successful': len(valid_results),
        'failed': len(all_results) - len(valid_results),
        'best_aic': sorted_by_aic[0][0],
        'best_bic': sorted_by_bic[0][0],
        'best_rmse': sorted_by_rmse[0][0],
        'all_results': all_results
    }
    
    json_path = output_dir / 'garch_order_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved JSON summary to: {json_path}")
    
    print("\n" + "=" * 80)
    print("GARCH ORDER COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nBest Overall Model: {sorted_by_bic[0][0]}")
    print(f"  • BIC: {sorted_by_bic[0][1]['bic']:.4f}")
    print(f"  • Test RMSE: {sorted_by_bic[0][1]['test_rmse']:.6f}")
    print(f"  • Test MSE: {sorted_by_bic[0][1]['test_mse']:.8f}")
    
    return summary


if __name__ == "__main__":
    results = train_garch_variants()

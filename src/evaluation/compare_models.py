"""
Model Comparison Script: 7-Model Comprehensive Comparison

Compares all models:
1. ARIMA - Classical time series
2. GARCH - Volatility modeling
3. LSTM - Deep learning baseline
4. Hybrid GARCH-LSTM - Volatility + DL
5. ARIMA-LSTM - Linear + Non-linear
6. ARIMA-GARCH - Mean + Variance
7. ARIMA-GARCH-LSTM - Complete hybrid

Author: Naveen Babu
Date: January 19, 2026
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.utils.config import PREDICTIONS_DIR, FIGURES_DIR

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_latest_results(model_name: str) -> dict:
    """
    Load the latest prediction results for a given model.
    
    Args:
        model_name: Model directory prefix (e.g., 'arima', 'garch', 'hybrid')
        
    Returns:
        Dictionary with metrics for train/val/test
    """
    # Find latest results directory
    model_dirs = sorted(PREDICTIONS_DIR.glob(f"{model_name}_*"))
    
    if not model_dirs:
        print(f"⚠️ No results found for {model_name}")
        return None
    
    latest_dir = model_dirs[-1]
    metrics_file = latest_dir / "metrics_summary.json"
    
    if not metrics_file.exists():
        print(f"⚠️ Metrics file not found: {metrics_file}")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print(f"✅ Loaded {model_name} from: {latest_dir.name}")
    return metrics


def create_comparison_table(all_metrics: dict) -> pd.DataFrame:
    """
    Create comparison table across all models.
    
    Args:
        all_metrics: Dictionary of {model_name: metrics_dict}
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for model_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        
        for subset in ['train', 'val', 'test']:
            if subset in metrics:
                subset_metrics = metrics[subset]
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Subset': subset.upper(),
                    'RMSE': subset_metrics.get('rmse', np.nan),
                    'MAE': subset_metrics.get('mae', np.nan),
                    'R²': subset_metrics.get('r2', np.nan),
                    'Directional Accuracy (%)': subset_metrics.get('directional_accuracy', np.nan),
                    'Samples': subset_metrics.get('n_samples', 0)
                })
    
    return pd.DataFrame(comparison_data)


def plot_comparison_bar_charts(comparison_df: pd.DataFrame, output_dir: Path):
    """
    Create bar charts comparing models across metrics.
    
    Args:
        comparison_df: DataFrame with comparison data
        output_dir: Directory to save plots
    """
    metrics_to_plot = ['RMSE', 'MAE', 'R²', 'Directional Accuracy (%)']
    subsets = ['TEST']  # Focus on test set for main comparison
    
    test_data = comparison_df[comparison_df['Subset'] == 'TEST'].copy()
    
    if test_data.empty:
        print("⚠️ No test data available for plotting")
        return
    
    # Calculate number of models
    n_models = len(test_data)
    colors = plt.cm.tab10(range(n_models))  # Dynamic colors for all models
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Sort by metric (lower is better for RMSE/MAE, higher for R²/Dir.Acc)
        ascending = True if metric in ['RMSE', 'MAE'] else False
        plot_data = test_data.sort_values(metric, ascending=ascending)
        
        # Bar plot
        bars = ax.bar(range(len(plot_data)), plot_data[metric], color=colors)
        
        # Customize
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data['Model'], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison (Test Set)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            label_format = '{:.6f}' if metric in ['RMSE', 'MAE'] else '{:.4f}' if metric == 'R²' else '{:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label_format.format(height),
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "model_comparison_test.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {output_path}")
    plt.show()


def plot_all_subsets_comparison(comparison_df: pd.DataFrame, output_dir: Path):
    """
    Create grouped bar charts showing train/val/test for each model.
    
    Args:
        comparison_df: DataFrame with comparison data
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics_to_plot = ['RMSE', 'MAE', 'R²', 'Directional Accuracy (%)']
    models = comparison_df['Model'].unique()
    subsets = ['TRAIN', 'VAL', 'TEST']
    
    x = np.arange(len(models))
    width = 0.25
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for i, subset in enumerate(subsets):
            subset_data = comparison_df[comparison_df['Subset'] == subset]
            values = [subset_data[subset_data['Model'] == model][metric].values[0] 
                     if len(subset_data[subset_data['Model'] == model]) > 0 else 0
                     for model in models]
            
            ax.bar(x + i*width, values, width, label=subset, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Across Train/Val/Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=0, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "model_comparison_all_subsets.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved all-subsets comparison: {output_path}")
    plt.show()


def main():
    """Main execution function."""
    print("="*80)
    print(" " * 25 + "MODEL COMPARISON")
    print("="*80)
    print(" "*20 + "7-MODEL COMPREHENSIVE COMPARISON")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load results from all 7 models
    print("\n[INFO] Loading model results...")
    
    all_metrics = {
        'ARIMA': load_latest_results('arima_predictions'),
        'GARCH': load_latest_results('garch_predictions'),
        'LSTM': load_latest_results('lstm_predictions'),
        'Hybrid GARCH-LSTM': load_latest_results('hybrid_predictions'),
        'ARIMA-LSTM': load_latest_results('arima_lstm_hybrid'),
        'ARIMA-GARCH': load_latest_results('arima_garch_hybrid'),
        'ARIMA-GARCH-LSTM': load_latest_results('arima_garch_lstm_hybrid'),
    }
    
    # Filter out None values
    all_metrics = {k: v for k, v in all_metrics.items() if v is not None}
    
    if not all_metrics:
        print("\n❌ No model results found. Please train models first.")
        return
    
    print(f"\n✅ Loaded {len(all_metrics)} model(s)")
    
    # Create comparison table
    print("\n" + "="*80)
    print("CREATING COMPARISON TABLE")
    print("="*80)
    
    comparison_df = create_comparison_table(all_metrics)
    
    # Display comparison table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Format for display
    display_df = comparison_df.copy()
    for col in ['RMSE', 'MAE']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if not pd.isna(x) else "N/A")
    display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    display_df['Directional Accuracy (%)'] = display_df['Directional Accuracy (%)'].apply(
        lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
    )
    display_df['Samples'] = display_df['Samples'].apply(lambda x: f"{int(x):,}" if x > 0 else "N/A")
    
    print(display_df.to_string(index=False))
    
    # Save comparison table
    output_dir = FIGURES_DIR / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_csv = output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n✅ Saved comparison table: {comparison_csv}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    plot_comparison_bar_charts(comparison_df, output_dir)
    plot_all_subsets_comparison(comparison_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    test_data = comparison_df[comparison_df['Subset'] == 'TEST']
    if not test_data.empty:
        best_rmse = test_data.loc[test_data['RMSE'].idxmin()]
        best_dir_acc = test_data.loc[test_data['Directional Accuracy (%)'].idxmax()]
        
        print(f"\n🏆 Best Test RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.6f})")
        print(f"🏆 Best Directional Accuracy: {best_dir_acc['Model']} ({best_dir_acc['Directional Accuracy (%)']:.2f}%)")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

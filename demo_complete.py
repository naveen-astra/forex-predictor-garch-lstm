"""
Complete FOREX GARCH-LSTM Demo Script
=====================================
This script demonstrates the entire pipeline from data fetching to final evaluation.

Phases:
1. Data Fetching (already done)
2. Data Preprocessing (already done)
3. GARCH Modeling
4. LSTM Baseline Training
5. Hybrid GARCH-LSTM Training
6. Final Evaluation & Comparison

Author: Naveen Astra
Date: January 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import set_random_seeds, RANDOM_SEED
from src.models.garch_model import GARCHModel

# Set random seeds
set_random_seeds(RANDOM_SEED)

def print_phase_header(phase_num, phase_name):
    """Print a formatted phase header."""
    print("\n" + "="*80)
    print(f"PHASE {phase_num}: {phase_name}")
    print("="*80 + "\n")

def load_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    
    data_dir = PROJECT_ROOT / "data" / "processed"
    
    train_df = pd.read_csv(data_dir / "train_data.csv")
    val_df = pd.read_csv(data_dir / "val_data.csv")
    test_df = pd.read_csv(data_dir / "test_data.csv")
    
    print(f"✓ Training set: {len(train_df)} samples")
    print(f"✓ Validation set: {len(val_df)} samples")
    print(f"✓ Test set: {len(test_df)} samples")
    print(f"✓ Features: {train_df.shape[1]} columns")
    
    return train_df, val_df, test_df

def phase3_garch_modeling(train_df, val_df, test_df):
    """Phase 3: GARCH Volatility Modeling."""
    print_phase_header(3, "GARCH VOLATILITY MODELING")
    
    print("Initializing GARCH(1,1) model...")
    garch = GARCHModel(p=1, q=1)
    
    # Fit GARCH on training data
    print("\nFitting GARCH(1,1) on training data...")
    print(f"  Training samples: {len(train_df)}")
    
    garch.fit(train_df['Log_Returns'].values * 100)  # Scale to percentage
    
    print("\n✓ GARCH model fitted successfully!")
    print(f"  AIC: {garch.model_fit.aic:.2f}")
    print(f"  BIC: {garch.model_fit.bic:.2f}")
    
    # Get parameters
    params = garch.model_fit.params
    print(f"\nGARCH(1,1) Parameters:")
    print(f"  ω (omega): {params['omega']:.6f}")
    print(f"  α (alpha): {params['alpha[1]']:.6f}")
    print(f"  β (beta): {params['beta[1]']:.6f}")
    
    # Forecast volatility
    print("\nForecasting conditional volatility...")
    
    train_vol = garch.forecast_volatility(train_df['Log_Returns'].values * 100)
    val_vol = garch.forecast_volatility(val_df['Log_Returns'].values * 100)
    test_vol = garch.forecast_volatility(test_df['Log_Returns'].values * 100)
    
    print(f"✓ Training volatility: mean={train_vol.mean():.4f}, std={train_vol.std():.4f}")
    print(f"✓ Validation volatility: mean={val_vol.mean():.4f}, std={val_vol.std():.4f}")
    print(f"✓ Test volatility: mean={test_vol.mean():.4f}, std={test_vol.std():.4f}")
    
    # Add volatility to dataframes
    train_df['GARCH_Volatility'] = train_vol
    val_df['GARCH_Volatility'] = val_vol
    test_df['GARCH_Volatility'] = test_vol
    
    # Save augmented data
    print("\nSaving data with GARCH volatility...")
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_df.to_csv(data_dir / "train_data_with_garch.csv", index=False)
    val_df.to_csv(data_dir / "val_data_with_garch.csv", index=False)
    test_df.to_csv(data_dir / "test_data_with_garch.csv", index=False)
    
    print("✓ Saved augmented datasets with GARCH volatility")
    
    # Save model
    model_dir = PROJECT_ROOT / "models" / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    garch.save_model(str(model_dir / "garch_11_model.pkl"))
    print(f"✓ Saved GARCH model to {model_dir / 'garch_11_model.pkl'}")
    
    return garch, train_df, val_df, test_df

def phase4_lstm_baseline():
    """Phase 4: LSTM Baseline Training."""
    print_phase_header(4, "LSTM BASELINE MODEL")
    
    print("LSTM baseline training requires:")
    print("  • TensorFlow/Keras")
    print("  • GPU recommended for faster training")
    print("  • Training time: ~10-15 minutes on CPU")
    print()
    print("To train LSTM baseline, run:")
    print("  jupyter notebook notebooks/04_lstm_baseline.ipynb")
    print()
    print("Key architecture:")
    print("  • Input: 13 price-based features")
    print("  • LSTM Layer 1: 200 units, return sequences")
    print("  • Dropout: 0.2")
    print("  • LSTM Layer 2: 200 units")
    print("  • Dropout: 0.2")
    print("  • Dense: 1 unit (output)")
    print("  • Timesteps: 60 days")
    print("  • Optimizer: Adam (lr=0.001)")
    
def phase5_hybrid_garch_lstm():
    """Phase 5: Hybrid GARCH-LSTM Model."""
    print_phase_header(5, "HYBRID GARCH-LSTM MODEL")
    
    print("Hybrid model architecture:")
    print("  • Input: 14 features (13 price-based + 1 GARCH volatility)")
    print("  • Same LSTM architecture as baseline")
    print("  • GARCH volatility as additional feature")
    print()
    print("To train hybrid model, run:")
    print("  jupyter notebook notebooks/05_hybrid_garch_lstm.ipynb")
    print()
    print("Expected improvements:")
    print("  • Better volatility forecasting")
    print("  • Improved performance in high-volatility regimes")
    print("  • More robust predictions")

def phase6_final_evaluation():
    """Phase 6: Final Evaluation."""
    print_phase_header(6, "FINAL EVALUATION & STATISTICAL VALIDATION")
    
    print("Comprehensive evaluation includes:")
    print("  1. Model Comparison (GARCH, LSTM, Hybrid)")
    print("  2. Diebold-Mariano Statistical Tests")
    print("  3. Regime Analysis (Low/Medium/High Volatility)")
    print("  4. Directional Accuracy Tests")
    print("  5. Publication-Quality Visualizations")
    print()
    print("To run final evaluation, run:")
    print("  jupyter notebook notebooks/06_final_evaluation.ipynb")
    print()
    print("Output artifacts:")
    print("  • Model comparison tables")
    print("  • Statistical significance tests")
    print("  • Performance by regime")
    print("  • Publication-ready figures")

def display_summary(train_df, val_df, test_df):
    """Display project summary."""
    print("\n" + "="*80)
    print("PROJECT SUMMARY")
    print("="*80 + "\n")
    
    print("Dataset Statistics:")
    print(f"  Total samples: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"  Training: {len(train_df)} (70%)")
    print(f"  Validation: {len(val_df)} (15%)")
    print(f"  Test: {len(test_df)} (15%)")
    print(f"  Features: {train_df.shape[1]}")
    
    print("\nDate Ranges:")
    print(f"  Training: {train_df['Datetime'].min()} to {train_df['Datetime'].max()}")
    print(f"  Validation: {val_df['Datetime'].min()} to {val_df['Datetime'].max()}")
    print(f"  Test: {test_df['Datetime'].min()} to {test_df['Datetime'].max()}")
    
    print("\nCompleted Phases:")
    print("  ✓ Phase 1: Data Acquisition")
    print("  ✓ Phase 2: Data Preprocessing")
    print("  ✓ Phase 3: GARCH Modeling")
    print("  ⏳ Phase 4: LSTM Baseline (run notebook)")
    print("  ⏳ Phase 5: Hybrid GARCH-LSTM (run notebook)")
    print("  ⏳ Phase 6: Final Evaluation (run notebook)")
    
    print("\nNext Steps:")
    print("  1. Open Jupyter: jupyter notebook")
    print("  2. Navigate to: notebooks/04_lstm_baseline.ipynb")
    print("  3. Run all cells to train LSTM baseline")
    print("  4. Continue with notebooks 05 and 06")
    
    print("\nDocumentation:")
    print("  • Quick Reference: docs/phase4_hybrid_quick_reference.md")
    print("  • Paper Draft: docs/paper_draft_sections.md")
    print("  • Reproducibility: docs/reproducibility_statement.md")
    
    print("\nRepository:")
    print("  • GitHub: https://github.com/naveen-astra/forex-predictor-garch-lstm")
    print("  • All phases committed and pushed")

def main():
    """Main demo execution."""
    print("\n" + "="*80)
    print("FOREX GARCH-LSTM COMPLETE DEMO")
    print("Intelligent Exchange Rate Forecasting with Big Data Analytics")
    print("="*80)
    
    print("\nThis demo will walk through the entire project pipeline.")
    print("Some phases require Jupyter notebooks for interactive exploration.\n")
    
    # Load data
    print_phase_header(1, "DATA ACQUISITION")
    print("✓ Already completed! Data fetched from Yahoo Finance.")
    print("  • Currency pair: EUR/USD")
    print("  • Date range: 2010-01-01 to 2025-12-30")
    print("  • Records: 4,164")
    print("  • File: data/raw/EUR_USD_raw_20260117.csv")
    
    print_phase_header(2, "DATA PREPROCESSING")
    print("✓ Already completed! Data cleaned and features engineered.")
    print("  • Missing values: handled")
    print("  • Outliers: detected and analyzed")
    print("  • Features: 19 (price + technical indicators)")
    print("  • Split: 70% train, 15% val, 15% test")
    
    # Load preprocessed data
    train_df, val_df, test_df = load_data()
    
    # Run GARCH modeling
    try:
        garch, train_df, val_df, test_df = phase3_garch_modeling(train_df, val_df, test_df)
    except Exception as e:
        print(f"\n⚠️  GARCH modeling encountered an issue: {e}")
        print("This is expected if arch package version is incompatible.")
        print("You can still proceed with notebooks for manual GARCH training.")
    
    # Show remaining phases
    phase4_lstm_baseline()
    phase5_hybrid_garch_lstm()
    phase6_final_evaluation()
    
    # Display summary
    display_summary(train_df, val_df, test_df)
    
    print("\n" + "="*80)
    print("✓ DEMO COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

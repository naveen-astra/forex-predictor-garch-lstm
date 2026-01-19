"""
Configuration file for FOREX GARCH-LSTM forecasting project.
This file contains all hyperparameters, paths, and random seeds for reproducibility.

Author: Research Team
Date: January 2026
Project: Intelligent FOREX Exchange Rate Forecasting using Hybrid GARCH-LSTM and Big Data Analytics
"""

import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  SAVED_MODELS_DIR, CHECKPOINTS_DIR,
                  FIGURES_DIR, TABLES_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# REPRODUCIBILITY SETTINGS
# ============================================================================
RANDOM_SEED = 42  # Master seed for all random operations

def set_random_seeds(seed=RANDOM_SEED):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
        
    Note:
        Call this function at the start of every script/notebook to ensure
        reproducible results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # For hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow deterministic operations (may slow down training)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"✓ Random seeds set to {seed} for reproducibility")

# ============================================================================
# DATA ACQUISITION SETTINGS
# ============================================================================
DATA_CONFIG = {
    # Primary currency pair
    'currency_pair': 'EURUSD=X',  # Yahoo Finance ticker for EUR/USD
    'currency_pair_name': 'EUR_USD',
    
    # Date range (extensive historical data for robust training)
    'start_date': '2010-01-01',  # 15+ years of data
    'end_date': '2025-12-31',
    
    # Data frequency
    'frequency': '1d',  # Daily data (can be changed to '1h' for intraday)
    
    # Data sources (in order of preference)
    'data_sources': ['yfinance', 'alpha_vantage'],
    
    # Alpha Vantage API settings (requires API key)
    'alpha_vantage_api_key': 'YOUR_API_KEY_HERE',  # Replace with actual key
    'alpha_vantage_function': 'FX_DAILY',
    'alpha_vantage_from_currency': 'EUR',
    'alpha_vantage_to_currency': 'USD',
    
    # Features to fetch
    'price_columns': ['Open', 'High', 'Low', 'Close'],
    'volume_column': 'Volume',  # May not be available for FOREX
}

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================
PREPROCESSING_CONFIG = {
    # Missing values
    'missing_value_strategy': 'forward_fill',  # Options: 'forward_fill', 'interpolate', 'drop'
    'max_missing_consecutive': 5,  # Drop if more than 5 consecutive NaNs
    
    # Outlier detection (using IQR method)
    'outlier_detection': True,
    'outlier_iqr_multiplier': 3.0,  # Standard is 1.5, we use 3.0 for financial data
    
    # Log transformation
    'apply_log_returns': True,
    'log_return_columns': ['Close'],
    
    # Feature engineering windows
    'rolling_volatility_windows': [10, 30, 60],  # days
    'technical_indicator_periods': {
        'rsi': 14,
        'sma': [14, 50, 200],
        'ema': [14, 26],
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
    },
    
    # Train/validation/test split
    'train_ratio': 0.70,
    'validation_ratio': 0.15,
    'test_ratio': 0.15,  # Most recent data for testing
}

# ============================================================================
# GARCH MODEL SETTINGS
# ============================================================================
GARCH_CONFIG = {
    # Model parameters - Using optimal order from ACF/PACF analysis
    'p': 2,  # GARCH lag order (optimal: BIC = 14257.22)
    'q': 1,  # ARCH lag order (optimal from ACF/PACF)
    'target_variable': 'Log_Returns',
    
    # Variants to test
    'model_variants': ['GARCH', 'EGARCH', 'GJR-GARCH'],  # For robustness checks
    
    # Prediction settings
    'rolling_window': True,
    'fit_once': True,  # Fit model once for efficiency
    'forecast_horizon': 10,  # 10-day ahead forecast
    
    # Output features
    'output_features': [
        'GARCH_volatility',
        'GARCH_conditional_variance',
    ],
}

# ============================================================================
# LSTM MODEL SETTINGS
# ============================================================================
LSTM_CONFIG = {
    # Architecture
    'lstm_units': [200, 200],  # Two-layer LSTM with 200 units each
    'dropout_rate': 0.2,
    'recurrent_dropout': 0.2,
    'activation': 'tanh',
    'return_sequences': [True, False],  # First layer returns sequences
    
    # Input shape
    'n_timesteps': 4,  # Number of time steps to look back
    'n_features': None,  # Calculated dynamically based on feature engineering
    'lag_order': 30,  # Create 30-day lagged features
    
    # Output
    'output_units': 1,  # Predicting single value (volatility)
    'output_activation': 'linear',
    
    # Training parameters
    'optimizer': 'adam',
    'learning_rate': 0.01,
    'loss_function': 'mse',
    'metrics': ['rmse', 'mae'],
    
    'batch_size': 700,
    'epochs': 60,
    'validation_split': 0.3,
    'shuffle': False,  # Don't shuffle time series data
    
    # Callbacks
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'model_checkpoint_monitor': 'val_loss',
    'model_checkpoint_mode': 'min',
}

# ============================================================================
# HYBRID MODEL SETTINGS
# ============================================================================
HYBRID_CONFIG = {
    'combine_method': 'concatenate',  # How to combine GARCH and other features
    'garch_weight': 1.0,  # Weight for GARCH features (if using weighted combination)
    'feature_scaling': 'minmax',  # Options: 'minmax', 'standard', None
    'scaling_range': (0, 1),
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
EVALUATION_CONFIG = {
    # Metrics to compute
    'metrics': [
        'rmse',  # Root Mean Squared Error
        'mae',   # Mean Absolute Error
        'mape',  # Mean Absolute Percentage Error
        'mse',   # Mean Squared Error
        'r2',    # R-squared
        'directional_accuracy',  # % correct direction predictions
    ],
    
    # Statistical tests
    'run_statistical_tests': True,
    'diebold_mariano_test': True,
    'model_confidence_set': True,
    
    # Baseline models for comparison
    'baseline_models': [
        'naive',  # Random walk
        'arima',
        'garch_only',
        'lstm_only',
    ],
    
    # Visualization settings
    'plot_predictions': True,
    'plot_residuals': True,
    'plot_loss_curves': True,
    'save_plots': True,
    'plot_format': 'png',
    'plot_dpi': 300,
}

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
EXPERIMENT_CONFIG = {
    'experiment_name': 'forex_garch_lstm',
    'experiment_version': 'v1.0',
    'experiment_description': 'Hybrid GARCH-LSTM for EUR/USD forecasting',
    
    # Logging
    'log_level': 'INFO',
    'log_file': PROJECT_ROOT / 'experiment.log',
    
    # Versioning
    'save_experiment_config': True,
    'config_save_path': RESULTS_DIR / 'experiment_config.json',
}

# ============================================================================
# BIG DATA SETTINGS (For Future Scaling)
# ============================================================================
SPARK_CONFIG = {
    'enable_spark': False,  # Set to True when scaling to Spark
    'spark_master': 'local[*]',
    'spark_app_name': 'FOREX_GARCH_LSTM',
    'spark_memory': '4g',
    'spark_executor_cores': 4,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_config_summary():
    """Print a summary of current configuration."""
    print("="*80)
    print("FOREX GARCH-LSTM PROJECT CONFIGURATION")
    print("="*80)
    print(f"Currency Pair: {DATA_CONFIG['currency_pair_name']}")
    print(f"Date Range: {DATA_CONFIG['start_date']} to {DATA_CONFIG['end_date']}")
    print(f"GARCH Model: GARCH({GARCH_CONFIG['p']}, {GARCH_CONFIG['q']})")
    print(f"LSTM Architecture: {LSTM_CONFIG['lstm_units']}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("="*80)

def validate_paths():
    """Validate that all necessary directories exist."""
    required_dirs = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR,
        FIGURES_DIR, TABLES_DIR, PREDICTIONS_DIR
    ]
    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Required directory not found: {directory}")
    print("✓ All required directories exist")

if __name__ == "__main__":
    # When run directly, display configuration summary
    get_config_summary()
    validate_paths()
    set_random_seeds()

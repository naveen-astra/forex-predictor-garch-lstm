"""
Hybrid GARCH-LSTM Model for FOREX Return Forecasting

This module implements the core research contribution: integrating GARCH conditional
volatility estimates as input features to an LSTM model. This hybrid approach combines:
- Statistical rigor of GARCH (volatility modeling)
- Pattern recognition of LSTM (non-linear learning)

Research Question:
    Does augmenting LSTM with GARCH volatility improve forecasting performance
    compared to LSTM-only or GARCH-only baselines?

Hypothesis:
    GARCH volatility provides LSTM with explicit information about:
    - Volatility clustering
    - Mean reversion dynamics
    - Conditional heteroskedasticity
    
    This should improve predictions, especially during high-volatility periods.

Implementation:
    - Uses LSTMForexModel as base class
    - Adds GARCH conditional volatility as 14th feature
    - Maintains same architecture for fair comparison
    - Same training protocol as baseline LSTM

Author: Research Team
Date: January 2026
License: MIT
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from src.models.lstm_model import LSTMForexModel
from src.models.garch_model import GARCHModel


class HybridGARCHLSTM:
    """
    Hybrid GARCH-LSTM model for FOREX forecasting.
    
    This class orchestrates the hybrid approach:
        1. Load GARCH volatility estimates (from Phase 2)
        2. Merge with price-based features
        3. Train LSTM with augmented feature set
        4. Compare performance with baselines
    
    The LSTM architecture remains identical to the baseline for fair comparison.
    The only difference is the addition of GARCH volatility as an input feature.
    
    Attributes:
        lstm_model: Underlying LSTM model
        garch_volatility_train: GARCH volatility for training
        garch_volatility_val: GARCH volatility for validation
        garch_volatility_test: GARCH volatility for test
        feature_columns: List of all features (including GARCH)
    """
    
    def __init__(self,
                 n_timesteps: int = 4,
                 lstm_units: List[int] = [200, 200],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.01,
                 verbose: int = 1):
        """
        Initialize Hybrid GARCH-LSTM model.
        
        Args:
            n_timesteps: Number of past time steps (same as baseline)
            lstm_units: LSTM layer sizes (same as baseline)
            dropout_rate: Dropout rate (same as baseline)
            learning_rate: Learning rate (same as baseline)
            verbose: Verbosity level
        """
        self.lstm_model = LSTMForexModel(
            n_timesteps=n_timesteps,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            verbose=verbose
        )
        
        self.garch_volatility_train = None
        self.garch_volatility_val = None
        self.garch_volatility_test = None
        self.feature_columns = None
        
    def load_garch_volatility(self,
                            train_path: Path,
                            val_path: Path,
                            test_path: Path,
                            volatility_column: str = 'GARCH_Volatility') -> None:
        """
        Load GARCH conditional volatility from Phase 2 outputs.
        
        Args:
            train_path: Path to training data with GARCH volatility
            val_path: Path to validation data with GARCH volatility
            test_path: Path to test data with GARCH volatility
            volatility_column: Name of GARCH volatility column
            
        Notes:
            - These files were generated in Phase 2 (GARCH modeling)
            - GARCH volatility was computed without data leakage
            - Training volatility uses only training data
            - Val/test volatility uses fixed GARCH parameters
        """
        print("Loading GARCH conditional volatility from Phase 2...")
        print("=" * 70)
        
        train_data = pd.read_csv(train_path, index_col=0, parse_dates=True)
        val_data = pd.read_csv(val_path, index_col=0, parse_dates=True)
        test_data = pd.read_csv(test_path, index_col=0, parse_dates=True)
        
        # Verify GARCH volatility column exists
        if volatility_column not in train_data.columns:
            raise ValueError(f"GARCH volatility column '{volatility_column}' not found in data.")
        
        self.garch_volatility_train = train_data[volatility_column]
        self.garch_volatility_val = val_data[volatility_column]
        self.garch_volatility_test = test_data[volatility_column]
        
        print(f"✓ Loaded GARCH volatility:")
        print(f"  Train: {len(self.garch_volatility_train)} observations")
        print(f"  Val:   {len(self.garch_volatility_val)} observations")
        print(f"  Test:  {len(self.garch_volatility_test)} observations")
        print()
        print(f"GARCH Volatility Statistics (Training):")
        print(self.garch_volatility_train.describe())
        
        return train_data, val_data, test_data
    
    def prepare_hybrid_features(self,
                               train_data: pd.DataFrame,
                               val_data: pd.DataFrame,
                               test_data: pd.DataFrame,
                               base_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Combine base features with GARCH volatility.
        
        Args:
            train_data: Training DataFrame with GARCH volatility
            val_data: Validation DataFrame with GARCH volatility
            test_data: Test DataFrame with GARCH volatility
            base_features: List of base feature columns (from Phase 3)
            
        Returns:
            Tuple of (train, val, test) DataFrames with all features
            
        Notes:
            - GARCH volatility is added as the last feature
            - Feature ordering: [base_features, 'GARCH_Volatility']
            - This allows direct comparison with LSTM baseline
        """
        print("Preparing hybrid feature set...")
        print("=" * 70)
        
        # Add GARCH volatility to feature list
        hybrid_features = base_features + ['GARCH_Volatility']
        self.feature_columns = hybrid_features
        
        # Extract hybrid features
        train_hybrid = train_data[hybrid_features].copy()
        val_hybrid = val_data[hybrid_features].copy()
        test_hybrid = test_data[hybrid_features].copy()
        
        # Handle any NaN values
        train_hybrid = train_hybrid.fillna(method='ffill').fillna(method='bfill').dropna()
        val_hybrid = val_hybrid.fillna(method='ffill').fillna(method='bfill').dropna()
        test_hybrid = test_hybrid.fillna(method='ffill').fillna(method='bfill').dropna()
        
        print(f"✓ Hybrid feature set created:")
        print(f"  Base features: {len(base_features)}")
        print(f"  GARCH volatility: 1")
        print(f"  Total features: {len(hybrid_features)}")
        print()
        print("Feature list:")
        for i, feat in enumerate(hybrid_features, 1):
            marker = " [GARCH]" if feat == 'GARCH_Volatility' else ""
            print(f"  {i:2d}. {feat}{marker}")
        
        return train_hybrid, val_hybrid, test_hybrid
    
    def train_hybrid_model(self,
                          train_data: pd.DataFrame,
                          val_data: pd.DataFrame,
                          test_data: pd.DataFrame,
                          target_column: str = 'Log_Returns',
                          epochs: int = 100,
                          batch_size: int = 32,
                          early_stopping_patience: int = 10,
                          checkpoint_path: Optional[Path] = None) -> Dict:
        """
        Train hybrid GARCH-LSTM model.
        
        Args:
            train_data: Training DataFrame with GARCH volatility
            val_data: Validation DataFrame with GARCH volatility
            test_data: Test DataFrame with GARCH volatility
            target_column: Name of target variable
            epochs: Maximum training epochs
            batch_size: Batch size
            early_stopping_patience: Early stopping patience
            checkpoint_path: Path to save best model
            
        Returns:
            Training history dictionary
            
        Notes:
            - Uses same training protocol as LSTM baseline
            - Only difference: additional GARCH volatility feature
            - This ensures fair comparison
        """
        print("\n" + "=" * 70)
        print("TRAINING HYBRID GARCH-LSTM MODEL")
        print("=" * 70)
        
        # Prepare sequences
        X_train, y_train, X_val, y_val, X_test, y_test = self.lstm_model.prepare_data(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            feature_columns=self.feature_columns,
            target_column=target_column
        )
        
        # Build model
        self.lstm_model.build_model(n_features=len(self.feature_columns))
        
        # Train model
        history = self.lstm_model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=checkpoint_path
        )
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        return history
    
    def evaluate_hybrid(self) -> Dict[str, float]:
        """
        Evaluate hybrid model on test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not hasattr(self, 'X_test'):
            raise ValueError("Model must be trained first.")
        
        print("\n" + "=" * 70)
        print("EVALUATING HYBRID GARCH-LSTM MODEL")
        print("=" * 70)
        
        metrics = self.lstm_model.evaluate(self.X_test, self.y_test)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using hybrid model."""
        return self.lstm_model.predict(X)
    
    def save_model(self, model_path: Path, scaler_path: Path) -> None:
        """Save trained hybrid model."""
        self.lstm_model.save_model(model_path, scaler_path)
    
    @classmethod
    def load_model(cls, model_path: Path, scaler_path: Path):
        """Load trained hybrid model."""
        instance = cls()
        instance.lstm_model = LSTMForexModel.load_model(model_path, scaler_path)
        instance.feature_columns = instance.lstm_model.feature_columns
        return instance


def compare_models(garch_metrics: Dict[str, float],
                  lstm_metrics: Dict[str, float],
                  hybrid_metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Compare GARCH-only, LSTM-only, and Hybrid GARCH-LSTM models.
    
    Args:
        garch_metrics: Metrics from GARCH model (Phase 2)
        lstm_metrics: Metrics from LSTM baseline (Phase 3)
        hybrid_metrics: Metrics from Hybrid model (Phase 4)
        
    Returns:
        Comparison DataFrame with performance improvements
        
    Notes:
        - Lower is better for MSE, MAE, RMSE
        - Higher is better for Directional Accuracy
    """
    comparison = pd.DataFrame({
        'GARCH-only': garch_metrics,
        'LSTM-only': lstm_metrics,
        'Hybrid GARCH-LSTM': hybrid_metrics
    }).T
    
    # Calculate improvements over baselines
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison)
    
    print("\n" + "=" * 70)
    print("PERFORMANCE IMPROVEMENTS")
    print("=" * 70)
    
    # Hybrid vs LSTM-only
    for metric in ['MSE', 'MAE', 'RMSE']:
        lstm_val = lstm_metrics[metric]
        hybrid_val = hybrid_metrics[metric]
        improvement = ((lstm_val - hybrid_val) / lstm_val) * 100
        print(f"{metric:25s}: {improvement:+.2f}% (Hybrid vs LSTM-only)")
    
    # Directional accuracy
    lstm_acc = lstm_metrics['Directional_Accuracy']
    hybrid_acc = hybrid_metrics['Directional_Accuracy']
    acc_improvement = hybrid_acc - lstm_acc
    print(f"{'Directional Accuracy':25s}: {acc_improvement:+.2f} percentage points")
    
    # Overall assessment
    print("\n" + "=" * 70)
    print("STATISTICAL INTERPRETATION:")
    print("=" * 70)
    
    rmse_improvement = ((lstm_metrics['RMSE'] - hybrid_metrics['RMSE']) / lstm_metrics['RMSE']) * 100
    
    if rmse_improvement > 5:
        print("✓ SUBSTANTIAL IMPROVEMENT: Hybrid model significantly outperforms LSTM-only")
    elif rmse_improvement > 2:
        print("✓ MODERATE IMPROVEMENT: Hybrid model shows meaningful gains")
    elif rmse_improvement > 0:
        print("⚠ MARGINAL IMPROVEMENT: Small gains, may not be statistically significant")
    else:
        print("✗ NO IMPROVEMENT: LSTM-only performs as well or better")
        print("  Possible reasons: GARCH info already captured by rolling volatility")
    
    return comparison


if __name__ == "__main__":
    print("Hybrid GARCH-LSTM Model Module")
    print("=" * 70)
    print("This module implements the core research contribution:")
    print("  - Combines GARCH volatility with LSTM pattern recognition")
    print("  - Fair comparison with GARCH-only and LSTM-only baselines")
    print("  - Journal-ready implementation and evaluation")
    print()
    print("Use notebooks/05_hybrid_garch_lstm.ipynb for complete analysis.")

"""
ARIMA Baseline Model for FOREX Volatility Forecasting

This module implements ARIMA (AutoRegressive Integrated Moving Average) as a 
classical baseline for time series forecasting. ARIMA is a traditional statistical
model widely used in econometrics and serves as a benchmark for comparing more 
sophisticated models like GARCH and LSTM.

Key Features:
- Parameter identification using ACF/PACF or auto_arima
- Model training on log returns
- Multi-step ahead forecasting
- Comprehensive evaluation metrics
- Model persistence (save/load)

Author: Naveen Babu
Date: January 18, 2026
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import pickle
import json
from datetime import datetime

# Statistical modeling
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Auto ARIMA for parameter selection
try:
    from pmdarima import auto_arima
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False
    print("[WARNING] pmdarima not available. Manual parameter selection will be used.")

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project imports
from src.utils.config import (
    SAVED_MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR,
    FIGURES_DIR, RANDOM_SEED
)

# Set random seed
np.random.seed(RANDOM_SEED)


class ARIMABaselineModel:
    """
    ARIMA Baseline Model for time series forecasting.
    
    ARIMA(p,d,q) where:
    - p: Order of autoregressive (AR) component
    - d: Degree of differencing (I - Integrated)
    - q: Order of moving average (MA) component
    
    Attributes:
        order: Tuple (p, d, q) for ARIMA order
        model: Fitted statsmodels ARIMA model
        predictions: Dictionary storing predictions
        metrics: Dictionary storing evaluation metrics
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 0, 1)):
        """
        Initialize ARIMA model.
        
        Args:
            order: ARIMA order (p, d, q). Default (1, 0, 1) for AR(1) + MA(1)
        """
        self.order = order
        self.model = None
        self.predictions = {}
        self.metrics = {}
        self.training_data = None
        self.fitted = False
        
        print(f"[INFO] ARIMA model initialized with order {order}")
    
    @staticmethod
    def check_stationarity(series: pd.Series, alpha: float = 0.05) -> Dict:
        """
        Check stationarity using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series data
            alpha: Significance level (default: 0.05)
            
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*60)
        print("STATIONARITY TEST (Augmented Dickey-Fuller)")
        print("="*60)
        
        # Remove NaN values
        series_clean = series.dropna()
        
        # Perform ADF test
        result = adfuller(series_clean, autolag='AIC')
        
        test_results = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < alpha
        }
        
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"P-value: {result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.6f}")
        
        if test_results['is_stationary']:
            print(f"\n✅ Series IS stationary (p-value < {alpha})")
            print("   → No differencing required (d=0)")
        else:
            print(f"\n❌ Series is NOT stationary (p-value >= {alpha})")
            print("   → Differencing may be required (d≥1)")
        
        return test_results
    
    @staticmethod
    def identify_order_manual(series: pd.Series, max_lags: int = 40) -> Tuple[int, int, int]:
        """
        Identify ARIMA order using ACF and PACF analysis.
        
        Args:
            series: Time series data
            max_lags: Maximum number of lags to consider
            
        Returns:
            Suggested ARIMA order (p, d, q)
        """
        print("\n" + "="*60)
        print("ARIMA ORDER IDENTIFICATION (Manual ACF/PACF)")
        print("="*60)
        
        series_clean = series.dropna()
        
        # Check stationarity
        stationarity = ARIMABaselineModel.check_stationarity(series_clean)
        d = 0 if stationarity['is_stationary'] else 1
        
        # If not stationary, difference the series
        if d > 0:
            series_diff = series_clean.diff().dropna()
            print(f"\n[INFO] Applying differencing (d={d})")
            ARIMABaselineModel.check_stationarity(series_diff)
        else:
            series_diff = series_clean
        
        # Compute ACF and PACF
        acf_values = acf(series_diff, nlags=max_lags)
        pacf_values = pacf(series_diff, nlags=max_lags)
        
        # Identify significant lags (simple heuristic)
        # AR order (p): Based on PACF cutoff
        # MA order (q): Based on ACF cutoff
        
        # Find where PACF cuts off (becomes insignificant)
        significance = 1.96 / np.sqrt(len(series_diff))
        
        p = 0
        for i in range(1, min(max_lags, len(pacf_values))):
            if abs(pacf_values[i]) > significance:
                p = i
            else:
                break
        
        q = 0
        for i in range(1, min(max_lags, len(acf_values))):
            if abs(acf_values[i]) > significance:
                q = i
            else:
                break
        
        # Conservative approach: limit to reasonable orders
        p = min(p, 5)
        q = min(q, 5)
        
        suggested_order = (p, d, q)
        
        print(f"\n[INFO] ACF/PACF Analysis:")
        print(f"  Suggested p (AR order): {p}")
        print(f"  Suggested d (Differencing): {d}")
        print(f"  Suggested q (MA order): {q}")
        print(f"  → ARIMA{suggested_order}")
        
        return suggested_order
    
    @staticmethod
    def identify_order_auto(series: pd.Series, 
                           max_p: int = 5, 
                           max_d: int = 2, 
                           max_q: int = 5) -> Tuple[int, int, int]:
        """
        Identify ARIMA order using auto_arima (if available).
        
        Args:
            series: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            
        Returns:
            Optimal ARIMA order (p, d, q)
        """
        if not AUTO_ARIMA_AVAILABLE:
            print("[WARNING] pmdarima not available. Using manual method.")
            return ARIMABaselineModel.identify_order_manual(series)
        
        print("\n" + "="*60)
        print("ARIMA ORDER IDENTIFICATION (Auto ARIMA)")
        print("="*60)
        
        series_clean = series.dropna()
        
        print(f"[INFO] Searching ARIMA space:")
        print(f"  p: 0 to {max_p}")
        print(f"  d: 0 to {max_d}")
        print(f"  q: 0 to {max_q}")
        print(f"  Criterion: AIC (Akaike Information Criterion)")
        
        # Run auto_arima
        model = auto_arima(
            series_clean,
            start_p=0, max_p=max_p,
            start_d=0, max_d=max_d,
            start_q=0, max_q=max_q,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
            random_state=RANDOM_SEED
        )
        
        optimal_order = model.order
        
        print(f"\n[OK] Optimal order found: ARIMA{optimal_order}")
        print(f"  AIC: {model.aic():.2f}")
        print(f"  BIC: {model.bic():.2f}")
        
        return optimal_order
    
    def fit(self, train_data: pd.DataFrame, target_col: str = 'Log_Returns') -> 'ARIMABaselineModel':
        """
        Fit ARIMA model on training data.
        
        Args:
            train_data: Training DataFrame with datetime index
            target_col: Column name for target variable
            
        Returns:
            Self for method chaining
        """
        print("\n" + "="*60)
        print("TRAINING ARIMA MODEL")
        print("="*60)
        
        # Extract target series
        if 'Datetime' in train_data.columns:
            train_data = train_data.set_index('Datetime')
        
        y_train = train_data[target_col].dropna()
        self.training_data = y_train
        
        print(f"Training samples: {len(y_train):,}")
        print(f"Target variable: {target_col}")
        print(f"ARIMA order: {self.order}")
        
        # Fit ARIMA model
        print("\n[INFO] Fitting ARIMA model...")
        self.model = ARIMA(y_train, order=self.order)
        self.model_fit = self.model.fit()
        self.fitted = True
        
        print("[OK] Model fitted successfully")
        
        # Print model summary
        print("\n" + "-"*60)
        print("MODEL SUMMARY")
        print("-"*60)
        print(self.model_fit.summary())
        
        return self
    
    def predict(self, 
                val_data: Optional[pd.DataFrame] = None,
                test_data: Optional[pd.DataFrame] = None,
                target_col: str = 'Log_Returns') -> Dict[str, np.ndarray]:
        """
        Generate predictions on validation and test sets.
        
        Args:
            val_data: Validation DataFrame
            test_data: Test DataFrame
            target_col: Column name for target variable
            
        Returns:
            Dictionary with predictions for each subset
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        print("\n" + "="*60)
        print("GENERATING PREDICTIONS")
        print("="*60)
        
        predictions = {}
        
        # In-sample predictions (training)
        print("\n[TRAIN] In-sample predictions...")
        train_pred = self.model_fit.fittedvalues
        predictions['train'] = {
            'y_true': self.training_data.values,
            'y_pred': train_pred.values
        }
        print(f"  Predictions: {len(train_pred):,}")
        
        # Out-of-sample predictions (validation)
        if val_data is not None:
            print("\n[VAL] Out-of-sample predictions...")
            
            if 'Datetime' in val_data.columns:
                val_data = val_data.set_index('Datetime')
            
            y_val = val_data[target_col].dropna()
            
            # Forecast validation period
            n_val = len(y_val)
            val_pred = self.model_fit.forecast(steps=n_val)
            
            predictions['val'] = {
                'y_true': y_val.values,
                'y_pred': val_pred.values
            }
            print(f"  Predictions: {len(val_pred):,}")
        
        # Out-of-sample predictions (test)
        if test_data is not None:
            print("\n[TEST] Out-of-sample predictions...")
            
            if 'Datetime' in test_data.columns:
                test_data = test_data.set_index('Datetime')
            
            y_test = test_data[target_col].dropna()
            
            # Forecast test period (continuing from validation end)
            n_test = len(y_test)
            if val_data is not None:
                n_val = len(val_data)
                test_pred = self.model_fit.forecast(steps=n_val + n_test)[n_val:]
            else:
                test_pred = self.model_fit.forecast(steps=n_test)
            
            predictions['test'] = {
                'y_true': y_test.values,
                'y_pred': test_pred.values
            }
            print(f"  Predictions: {len(test_pred):,}")
        
        self.predictions = predictions
        return predictions
    
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on all subsets.
        
        Returns:
            Dictionary with metrics for each subset
        """
        if not self.predictions:
            raise ValueError("Must generate predictions before evaluation")
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        metrics = {}
        
        for subset_name, pred_dict in self.predictions.items():
            y_true = pred_dict['y_true']
            y_pred = pred_dict['y_pred']
            
            # Ensure same length (handle edge cases)
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Directional accuracy
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean((y_true_diff * y_pred_diff) > 0) * 100
            
            metrics[subset_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'n_samples': len(y_true)
            }
            
            print(f"\n[{subset_name.upper()}]")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
            print(f"  Samples: {len(y_true):,}")
        
        self.metrics = metrics
        return metrics
    
    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save predictions and metrics to disk.
        
        Args:
            output_dir: Output directory (default: PREDICTIONS_DIR/arima_predictions_<timestamp>)
            
        Returns:
            Path to output directory
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = PREDICTIONS_DIR / f"arima_predictions_{timestamp}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        print(f"Output directory: {output_dir}")
        
        # Save predictions for each subset
        for subset_name, pred_dict in self.predictions.items():
            pred_df = pd.DataFrame({
                'y_true': pred_dict['y_true'],
                'y_pred': pred_dict['y_pred'],
                'residual': pred_dict['y_true'] - pred_dict['y_pred']
            })
            
            pred_path = output_dir / f"{subset_name}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)
            print(f"  [OK] {subset_name}_predictions.csv")
        
        # Save metrics
        metrics_path = output_dir / "metrics_summary.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy types to native Python types
            metrics_serializable = {
                subset: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metrics.items()}
                for subset, metrics in self.metrics.items()
            }
            json.dump(metrics_serializable, f, indent=2)
        print(f"  [OK] metrics_summary.json")
        
        # Save model configuration
        config = {
            'model_type': 'ARIMA',
            'order': self.order,
            'n_training_samples': len(self.training_data) if self.training_data is not None else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  [OK] model_config.json")
        
        print(f"\n[OK] All results saved to: {output_dir}")
        
        return output_dir
    
    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """
        Save fitted ARIMA model to disk.
        
        Args:
            filepath: Path to save model (default: SAVED_MODELS_DIR/arima_model.pkl)
            
        Returns:
            Path to saved model
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")
        
        if filepath is None:
            filepath = SAVED_MODELS_DIR / "arima_model.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model_fit, f)
        
        print(f"[OK] Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: Path, order: Optional[Tuple[int, int, int]] = None):
        """
        Load fitted ARIMA model from disk.
        
        Args:
            filepath: Path to saved model
            order: ARIMA order (if known)
            
        Returns:
            ARIMABaselineModel instance
        """
        with open(filepath, 'rb') as f:
            model_fit = pickle.load(f)
        
        # Extract order from loaded model if not provided
        if order is None:
            # Try to extract from model_fit
            try:
                order = model_fit.model.order
            except:
                order = (1, 0, 1)  # Default fallback
        
        instance = cls(order=order)
        instance.model_fit = model_fit
        instance.fitted = True
        
        print(f"[OK] Model loaded from: {filepath}")
        print(f"  Order: ARIMA{order}")
        
        return instance


def main():
    """Main execution function for ARIMA baseline."""
    print("\n" + "="*70)
    print(" " * 20 + "ARIMA BASELINE MODEL")
    print("="*70)
    print("Classical Time Series Forecasting for FOREX Log Returns")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load preprocessed data
    print("\n[INFO] Loading preprocessed data...")
    
    train_data = pd.read_csv(PROCESSED_DATA_DIR / "train_data.csv")
    val_data = pd.read_csv(PROCESSED_DATA_DIR / "val_data.csv")
    test_data = pd.read_csv(PROCESSED_DATA_DIR / "test_data.csv")
    
    print(f"  Train: {len(train_data):,} records")
    print(f"  Val: {len(val_data):,} records")
    print(f"  Test: {len(test_data):,} records")
    
    # Identify optimal ARIMA order
    print("\n" + "="*70)
    print("STEP 1: PARAMETER IDENTIFICATION")
    print("="*70)
    
    # Use auto_arima if available, otherwise manual
    if AUTO_ARIMA_AVAILABLE:
        optimal_order = ARIMABaselineModel.identify_order_auto(
            train_data['Log_Returns'],
            max_p=5, max_d=2, max_q=5
        )
    else:
        optimal_order = ARIMABaselineModel.identify_order_manual(
            train_data['Log_Returns'],
            max_lags=40
        )
    
    # Initialize and train model
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    model = ARIMABaselineModel(order=optimal_order)
    model.fit(train_data, target_col='Log_Returns')
    
    # Generate predictions
    print("\n" + "="*70)
    print("STEP 3: PREDICTION")
    print("="*70)
    
    predictions = model.predict(
        val_data=val_data,
        test_data=test_data,
        target_col='Log_Returns'
    )
    
    # Evaluate model
    print("\n" + "="*70)
    print("STEP 4: EVALUATION")
    print("="*70)
    
    metrics = model.evaluate()
    
    # Save results
    print("\n" + "="*70)
    print("STEP 5: SAVE RESULTS")
    print("="*70)
    
    output_dir = model.save_results()
    model_path = model.save_model()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: ARIMA{optimal_order}")
    print(f"Results: {output_dir}")
    print(f"Model saved: {model_path}")
    
    print("\n" + "="*70)
    print("ARIMA BASELINE COMPLETE")
    print("="*70)
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()

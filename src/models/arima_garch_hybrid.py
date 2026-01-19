"""
ARIMA-GARCH Hybrid Model for FOREX Forecasting

This hybrid combines:
- ARIMA: Models mean (expected return)
- GARCH: Models variance (volatility)

Architecture:
1. ARIMA models the conditional mean (trend/autocorrelation)
2. GARCH models the conditional variance of ARIMA residuals
3. Final forecast = ARIMA mean ± GARCH volatility bands

This is a classical econometric approach (ARIMA-GARCH or ARMA-GARCH).

Author: Naveen Babu
Date: January 19, 2026
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("⚠️ pmdarima not available. Using fixed ARIMA order.")

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project imports
from src.utils.config import (
    SAVED_MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR,
    RANDOM_SEED, GARCH_CONFIG
)

np.random.seed(RANDOM_SEED)


class ARIMAGARCHHybrid:
    """
    ARIMA-GARCH Hybrid Model.
    
    Classical econometric approach:
    - ARIMA: Conditional mean E[y_t | past]
    - GARCH: Conditional variance Var[y_t | past]
    
    Final output:
    - Point forecast from ARIMA
    - Volatility estimate from GARCH
    - Prediction intervals using both
    """
    
    def __init__(self, arima_order=(1, 0, 1), garch_params=None):
        """
        Initialize ARIMA-GARCH hybrid.
        
        Args:
            arima_order: (p, d, q) for ARIMA
            garch_params: GARCH configuration (p, q)
        """
        self.arima_order = arima_order
        self.garch_params = garch_params or GARCH_CONFIG
        
        self.arima_model = None
        self.arima_fitted = None
        self.garch_model = None
        self.garch_fitted = None
        
        self.predictions = {}
        self.metrics = {}
        
        print(f"[INFO] ARIMA-GARCH Hybrid initialized")
        print(f"  ARIMA order: {arima_order}")
        print(f"  GARCH order: ({self.garch_params['p']}, {self.garch_params['q']})")
    
    def fit_arima(self, train_data: pd.Series):
        """Fit ARIMA on mean."""
        print("\n[ARIMA] Fitting conditional mean model...")
        
        self.arima_model = ARIMA(train_data, order=self.arima_order)
        self.arima_fitted = self.arima_model.fit()
        
        print(f"[OK] ARIMA{self.arima_order} fitted")
        print(f"  AIC: {self.arima_fitted.aic:.2f}")
        print(f"  BIC: {self.arima_fitted.bic:.2f}")
        
        return self.arima_fitted
    
    def fit_garch_on_residuals(self, residuals: pd.Series):
        """Fit GARCH on ARIMA residuals to model conditional variance."""
        print("\n[GARCH] Fitting conditional variance model on residuals...")
        
        # Scale residuals to percentage (GARCH works better with scaled data)
        scaled_residuals = residuals * 100
        
        # GARCH model
        self.garch_model = arch_model(
            scaled_residuals,
            vol='Garch',
            p=self.garch_params['p'],
            q=self.garch_params['q'],
            dist='normal',
            rescale=False
        )
        
        self.garch_fitted = self.garch_model.fit(disp='off')
        
        print(f"[OK] GARCH({self.garch_params['p']},{self.garch_params['q']}) fitted")
        print(f"  Log-Likelihood: {self.garch_fitted.loglikelihood:.2f}")
        print(f"  AIC: {self.garch_fitted.aic:.2f}")
        
        # Diagnostic summary
        print(f"\n  GARCH Parameters:")
        print(f"    omega: {self.garch_fitted.params['omega']:.6f}")
        print(f"    alpha[1]: {self.garch_fitted.params['alpha[1]']:.6f}")
        print(f"    beta[1]: {self.garch_fitted.params['beta[1]']:.6f}")
        
        persistence = self.garch_fitted.params['alpha[1]'] + self.garch_fitted.params['beta[1]']
        print(f"    Persistence (α+β): {persistence:.4f}")
        
        return self.garch_fitted
    
    def forecast_with_volatility(self, data: pd.DataFrame, subset_name: str = 'test'):
        """
        Generate ARIMA forecast with GARCH volatility bounds.
        
        Returns:
            - Mean forecast (ARIMA)
            - Volatility forecast (GARCH)
            - Upper/lower prediction bands
        """
        print(f"\n[HYBRID] Forecasting {subset_name} with volatility bands...")
        
        target_col = 'Log_Returns'
        y_true = data[target_col].values
        n_steps = len(data)
        
        # Step 1: ARIMA mean forecast
        arima_forecast = self.arima_fitted.forecast(steps=n_steps)
        
        # Step 2: GARCH volatility forecast
        # Note: GARCH forecast requires iterative approach for multi-step
        garch_vol_forecast = self.garch_fitted.forecast(horizon=n_steps, reindex=False)
        garch_variance = garch_vol_forecast.variance.values[-1, :]  # Last row = forecast
        garch_volatility = np.sqrt(garch_variance) / 100  # Convert back to decimal
        
        # Step 3: Prediction intervals (95% confidence)
        # Assuming normal distribution: mean ± 1.96 * std
        upper_band = arima_forecast.values + 1.96 * garch_volatility
        lower_band = arima_forecast.values - 1.96 * garch_volatility
        
        # Store results
        self.predictions[subset_name] = {
            'y_true': y_true,
            'y_pred': arima_forecast.values,  # Point forecast
            'volatility': garch_volatility,
            'upper_band': upper_band,
            'lower_band': lower_band
        }
        
        # Check coverage (how many actual values fall within bands)
        within_bands = np.sum((y_true >= lower_band) & (y_true <= upper_band))
        coverage = (within_bands / len(y_true)) * 100
        
        print(f"[OK] {subset_name} forecast generated")
        print(f"  Prediction interval coverage: {coverage:.2f}% (target: ~95%)")
        
        return arima_forecast.values, garch_volatility
    
    def evaluate(self):
        """Evaluate predictions."""
        print("\n" + "="*70)
        print("EVALUATION METRICS")
        print("="*70)
        
        for subset_name, pred_dict in self.predictions.items():
            y_true = pred_dict['y_true']
            y_pred = pred_dict['y_pred']
            
            # Standard metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Directional accuracy
            y_true_direction = np.diff(y_true)
            y_pred_direction = np.diff(y_pred)
            directional_accuracy = np.mean((y_true_direction * y_pred_direction) > 0) * 100
            
            # Average volatility
            avg_volatility = np.mean(pred_dict['volatility'])
            
            self.metrics[subset_name] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'directional_accuracy': float(directional_accuracy),
                'avg_volatility': float(avg_volatility),
                'n_samples': len(y_true)
            }
            
            print(f"\n[{subset_name.upper()}]")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
            print(f"  Avg Volatility: {avg_volatility:.6f}")
        
        return self.metrics
    
    def save_results(self, output_dir=None):
        """Save results."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = PREDICTIONS_DIR / f"arima_garch_hybrid_{timestamp}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[INFO] Saving results to: {output_dir}")
        
        # Save predictions
        for subset_name, pred_dict in self.predictions.items():
            pred_df = pd.DataFrame({
                'y_true': pred_dict['y_true'],
                'y_pred': pred_dict['y_pred'],
                'volatility': pred_dict['volatility'],
                'upper_band': pred_dict['upper_band'],
                'lower_band': pred_dict['lower_band'],
                'residual': pred_dict['y_true'] - pred_dict['y_pred']
            })
            
            pred_path = output_dir / f"{subset_name}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)
            print(f"  ✓ {subset_name}_predictions.csv")
        
        # Save metrics
        metrics_path = output_dir / "metrics_summary.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"  ✓ metrics_summary.json")
        
        # Save config
        config = {
            'model_type': 'ARIMA-GARCH Hybrid',
            'arima_order': self.arima_order,
            'garch_order': (self.garch_params['p'], self.garch_params['q']),
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ model_config.json")
        
        return output_dir
    
    def save_models(self):
        """Save trained models."""
        # Save ARIMA
        arima_path = SAVED_MODELS_DIR / "arima_garch_hybrid_arima.pkl"
        with open(arima_path, 'wb') as f:
            pickle.dump(self.arima_fitted, f)
        print(f"[OK] ARIMA saved: {arima_path}")
        
        # Save GARCH
        garch_path = SAVED_MODELS_DIR / "arima_garch_hybrid_garch.pkl"
        with open(garch_path, 'wb') as f:
            pickle.dump(self.garch_fitted, f)
        print(f"[OK] GARCH saved: {garch_path}")


def main():
    """Main execution."""
    print("="*80)
    print(" "*24 + "ARIMA-GARCH HYBRID MODEL")
    print("="*80)
    
    # Load data
    print("\n[INFO] Loading preprocessed data...")
    train_data = pd.read_csv(PROCESSED_DATA_DIR / "train_data.csv")
    val_data = pd.read_csv(PROCESSED_DATA_DIR / "val_data.csv")
    test_data = pd.read_csv(PROCESSED_DATA_DIR / "test_data.csv")
    
    print(f"  Train: {len(train_data):,} samples")
    print(f"  Val: {len(val_data):,} samples")
    print(f"  Test: {len(test_data):,} samples")
    
    # Initialize hybrid
    hybrid = ARIMAGARCHHybrid(arima_order=(1, 0, 1))
    
    # Step 1: Fit ARIMA
    train_target = train_data['Log_Returns'].dropna()
    hybrid.fit_arima(train_target)
    
    # Step 2: Get residuals and fit GARCH
    residuals = hybrid.arima_fitted.resid
    hybrid.fit_garch_on_residuals(residuals)
    
    # Step 3: Forecast with volatility
    hybrid.forecast_with_volatility(val_data, 'val')
    hybrid.forecast_with_volatility(test_data, 'test')
    
    # Train predictions
    train_pred = hybrid.arima_fitted.fittedvalues
    train_vol = hybrid.garch_fitted.conditional_volatility / 100
    hybrid.predictions['train'] = {
        'y_true': train_target.values,
        'y_pred': train_pred.values,
        'volatility': train_vol.values,
        'upper_band': train_pred.values + 1.96 * train_vol.values,
        'lower_band': train_pred.values - 1.96 * train_vol.values
    }
    
    # Step 4: Evaluate
    metrics = hybrid.evaluate()
    
    # Step 5: Save
    output_dir = hybrid.save_results()
    hybrid.save_models()
    
    print("\n" + "="*80)
    print("ARIMA-GARCH HYBRID TRAINING COMPLETE")
    print("="*80)
    print(f"Results: {output_dir}")
    
    return hybrid, metrics


if __name__ == "__main__":
    model, metrics = main()

"""
Combined ARIMA-GARCH-LSTM Hybrid Model

This is the most sophisticated hybrid combining all three approaches:
- ARIMA: Linear mean patterns (trend, seasonality, autocorrelation)
- GARCH: Volatility modeling (conditional heteroscedasticity)
- LSTM: Non-linear residual correction (complex patterns)

Architecture:
1. ARIMA models conditional mean
2. GARCH models conditional variance on ARIMA residuals
3. LSTM learns to predict remaining patterns in standardized residuals
4. Final prediction = ARIMA mean + LSTM correction, with GARCH volatility bounds

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

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project imports
from src.utils.config import (
    SAVED_MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR,
    RANDOM_SEED, GARCH_CONFIG, LSTM_CONFIG
)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class ARIMAGARCHLSTMHybrid:
    """
    Complete ARIMA-GARCH-LSTM Hybrid.
    
    Three-stage modeling:
    1. ARIMA: Captures linear patterns (AR, MA components)
    2. GARCH: Models volatility in residuals
    3. LSTM: Learns non-linear patterns in standardized residuals
    
    Final output:
    - Mean forecast = ARIMA + LSTM correction
    - Volatility bands from GARCH
    """
    
    def __init__(self, arima_order=(1, 0, 1), garch_params=None, lstm_config=None):
        """
        Initialize complete hybrid.
        
        Args:
            arima_order: (p, d, q) for ARIMA
            garch_params: GARCH configuration
            lstm_config: LSTM hyperparameters
        """
        self.arima_order = arima_order
        self.garch_params = garch_params or GARCH_CONFIG
        self.lstm_config = lstm_config or LSTM_CONFIG
        
        self.arima_model = None
        self.arima_fitted = None
        self.garch_model = None
        self.garch_fitted = None
        self.lstm_model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        self.predictions = {}
        self.metrics = {}
        
        print(f"[INFO] ARIMA-GARCH-LSTM Complete Hybrid initialized")
        print(f"  ARIMA order: {arima_order}")
        print(f"  GARCH order: ({self.garch_params['p']}, {self.garch_params['q']})")
        print(f"  LSTM units: {self.lstm_config.get('lstm_units', self.lstm_config.get('units', [200, 200]))}")
        print(f"  LSTM layers: {len(self.lstm_config.get('lstm_units', [200, 200]))}")
    
    def fit_arima(self, train_data: pd.Series):
        """Stage 1: Fit ARIMA."""
        print("\n[STAGE 1/3] Fitting ARIMA for conditional mean...")
        
        self.arima_model = ARIMA(train_data, order=self.arima_order)
        self.arima_fitted = self.arima_model.fit()
        
        print(f"[OK] ARIMA{self.arima_order} fitted")
        print(f"  AIC: {self.arima_fitted.aic:.2f}")
        print(f"  BIC: {self.arima_fitted.bic:.2f}")
        
        return self.arima_fitted
    
    def fit_garch_on_residuals(self, residuals: pd.Series):
        """Stage 2: Fit GARCH on ARIMA residuals."""
        print("\n[STAGE 2/3] Fitting GARCH for conditional variance...")
        
        scaled_residuals = residuals * 100
        
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
        
        # Get conditional volatility
        conditional_vol = self.garch_fitted.conditional_volatility / 100
        
        # Standardize residuals by volatility
        standardized_residuals = residuals / conditional_vol
        
        return standardized_residuals, conditional_vol
    
    def create_sequences(self, data: np.ndarray, n_timesteps: int):
        """Create sequences for LSTM."""
        X, y = [], []
        for i in range(n_timesteps, len(data)):
            X.append(data[i-n_timesteps:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_lstm(self, input_shape):
        """Build LSTM architecture."""
        model = Sequential(name='ARIMA_GARCH_LSTM_Hybrid')
        
        # Get LSTM units - handle both formats
        lstm_units = self.lstm_config.get('lstm_units', self.lstm_config.get('units', [200, 200]))
        if isinstance(lstm_units, list):
            first_units = lstm_units[0]
            second_units = lstm_units[1] if len(lstm_units) > 1 else lstm_units[0]
            n_layers = len(lstm_units)
        else:
            first_units = second_units = lstm_units
            n_layers = self.lstm_config.get('n_layers', 2)
        
        dropout_rate = self.lstm_config.get('dropout_rate', self.lstm_config.get('dropout', 0.2))
        
        # First LSTM layer
        model.add(LSTM(
            units=first_units,
            return_sequences=(n_layers > 1),
            input_shape=input_shape,
            name='lstm_1'
        ))
        model.add(Dropout(dropout_rate, name='dropout_1'))
        
        # Second LSTM layer
        if n_layers > 1:
            model.add(LSTM(
                units=second_units,
                return_sequences=False,
                name='lstm_2'
            ))
            model.add(Dropout(dropout_rate, name='dropout_2'))
        
        # Output
        model.add(Dense(1, name='output'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit_lstm_on_standardized_residuals(
        self, 
        train_std_residuals: np.ndarray,
        val_std_residuals: np.ndarray
    ):
        """Stage 3: Train LSTM on standardized residuals."""
        print("\n[STAGE 3/3] Training LSTM on standardized residuals...")
        
        n_timesteps = self.lstm_config['n_timesteps']
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_std_residuals, n_timesteps)
        X_val, y_val = self.create_sequences(val_std_residuals, n_timesteps)
        
        print(f"  Train sequences: {X_train.shape}")
        print(f"  Val sequences: {X_val.shape}")
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
        
        # Build model
        self.lstm_model = self.build_lstm(input_shape=(n_timesteps, 1))
        
        print("\nLSTM Architecture:")
        self.lstm_model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
        
        # Train
        print("\nTraining LSTM...")
        history = self.lstm_model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=self.lstm_config['epochs'],
            batch_size=self.lstm_config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print(f"[OK] LSTM trained")
        
        return history
    
    def predict_hybrid(
        self,
        data: pd.DataFrame,
        subset_name: str,
        arima_forecast: np.ndarray,
        standardized_residuals: np.ndarray,
        garch_volatility: np.ndarray
    ):
        """Generate complete hybrid predictions."""
        print(f"\n[HYBRID] Generating {subset_name} predictions...")
        
        target_col = 'Log_Returns'
        y_true = data[target_col].values
        n_timesteps = self.lstm_config['n_timesteps']
        
        # LSTM correction on standardized residuals
        if len(standardized_residuals) > n_timesteps:
            X_test, _ = self.create_sequences(standardized_residuals, n_timesteps)
            X_test_scaled = self.scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            
            lstm_correction = self.lstm_model.predict(X_test_reshaped, verbose=0).flatten()
            
            # Align dimensions
            arima_aligned = arima_forecast[n_timesteps:]
            garch_vol_aligned = garch_volatility[n_timesteps:]
            y_true_aligned = y_true[n_timesteps:]
            
            # Final prediction = ARIMA + LSTM correction * GARCH volatility
            y_pred = arima_aligned + lstm_correction * garch_vol_aligned
        else:
            # Fallback: Just use ARIMA
            arima_aligned = arima_forecast
            garch_vol_aligned = garch_volatility
            y_true_aligned = y_true
            y_pred = arima_aligned
            lstm_correction = np.zeros_like(arima_aligned)
        
        # Prediction intervals
        upper_band = y_pred + 1.96 * garch_vol_aligned
        lower_band = y_pred - 1.96 * garch_vol_aligned
        
        self.predictions[subset_name] = {
            'y_true': y_true_aligned,
            'y_pred': y_pred,
            'arima_component': arima_aligned,
            'lstm_correction': lstm_correction * garch_vol_aligned,
            'volatility': garch_vol_aligned,
            'upper_band': upper_band,
            'lower_band': lower_band
        }
        
        print(f"[OK] {subset_name} predictions generated ({len(y_pred)} samples)")
        
        return y_pred
    
    def evaluate(self):
        """Evaluate all predictions."""
        print("\n" + "="*70)
        print("EVALUATION METRICS - COMPLETE HYBRID")
        print("="*70)
        
        for subset_name, pred_dict in self.predictions.items():
            y_true = pred_dict['y_true']
            y_pred = pred_dict['y_pred']
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Directional accuracy
            y_true_direction = np.diff(y_true)
            y_pred_direction = np.diff(y_pred)
            directional_accuracy = np.mean((y_true_direction * y_pred_direction) > 0) * 100
            
            self.metrics[subset_name] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'directional_accuracy': float(directional_accuracy),
                'n_samples': len(y_true)
            }
            
            print(f"\n[{subset_name.upper()}]")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        return self.metrics
    
    def save_results(self, output_dir=None):
        """Save all results."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = PREDICTIONS_DIR / f"arima_garch_lstm_hybrid_{timestamp}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[INFO] Saving results to: {output_dir}")
        
        # Save predictions with component breakdown
        for subset_name, pred_dict in self.predictions.items():
            pred_df = pd.DataFrame({
                'y_true': pred_dict['y_true'],
                'y_pred': pred_dict['y_pred'],
                'arima_component': pred_dict['arima_component'],
                'lstm_correction': pred_dict['lstm_correction'],
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
            'model_type': 'ARIMA-GARCH-LSTM Complete Hybrid',
            'arima_order': self.arima_order,
            'garch_order': (self.garch_params['p'], self.garch_params['q']),
            'lstm_config': self.lstm_config,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ model_config.json")
        
        return output_dir
    
    def save_models(self):
        """Save all trained models."""
        # ARIMA
        arima_path = SAVED_MODELS_DIR / "arima_garch_lstm_hybrid_arima.pkl"
        with open(arima_path, 'wb') as f:
            pickle.dump(self.arima_fitted, f)
        print(f"[OK] ARIMA saved: {arima_path}")
        
        # GARCH
        garch_path = SAVED_MODELS_DIR / "arima_garch_lstm_hybrid_garch.pkl"
        with open(garch_path, 'wb') as f:
            pickle.dump(self.garch_fitted, f)
        print(f"[OK] GARCH saved: {garch_path}")
        
        # LSTM
        lstm_path = SAVED_MODELS_DIR / "arima_garch_lstm_hybrid_lstm.h5"
        self.lstm_model.save(lstm_path)
        print(f"[OK] LSTM saved: {lstm_path}")
        
        # Scaler
        scaler_path = SAVED_MODELS_DIR / "arima_garch_lstm_hybrid_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"[OK] Scaler saved: {scaler_path}")


def main():
    """Main training pipeline."""
    print("="*80)
    print(" "*18 + "ARIMA-GARCH-LSTM COMPLETE HYBRID MODEL")
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
    hybrid = ARIMAGARCHLSTMHybrid(arima_order=(1, 0, 1))
    
    # Stage 1: ARIMA
    train_target = train_data['Log_Returns'].dropna()
    hybrid.fit_arima(train_target)
    
    # Stage 2: GARCH
    train_residuals = hybrid.arima_fitted.resid
    train_std_residuals, train_garch_vol = hybrid.fit_garch_on_residuals(train_residuals)
    
    # Get validation standardized residuals
    val_target = val_data['Log_Returns'].values
    val_arima_forecast = hybrid.arima_fitted.forecast(steps=len(val_data))
    val_residuals = val_target - val_arima_forecast.values
    
    # GARCH volatility for validation (use last n observations to forecast)
    val_garch_forecast = hybrid.garch_fitted.forecast(horizon=len(val_data), reindex=False)
    val_garch_vol = np.sqrt(val_garch_forecast.variance.values[-1, :]) / 100
    val_std_residuals = val_residuals / val_garch_vol
    
    # Stage 3: LSTM
    history = hybrid.fit_lstm_on_standardized_residuals(
        train_std_residuals.values,
        val_std_residuals
    )
    
    # Predictions
    # Validation
    hybrid.predict_hybrid(
        val_data, 'val',
        val_arima_forecast.values,
        val_std_residuals,
        val_garch_vol
    )
    
    # Test
    test_target = test_data['Log_Returns'].values
    test_arima_forecast = hybrid.arima_fitted.forecast(steps=len(test_data))
    test_residuals = test_target - test_arima_forecast.values
    test_garch_forecast = hybrid.garch_fitted.forecast(horizon=len(test_data), reindex=False)
    test_garch_vol = np.sqrt(test_garch_forecast.variance.values[-1, :]) / 100
    test_std_residuals = test_residuals / test_garch_vol
    
    hybrid.predict_hybrid(
        test_data, 'test',
        test_arima_forecast.values,
        test_std_residuals,
        test_garch_vol
    )
    
    # Train (fitted values)
    train_fitted = hybrid.arima_fitted.fittedvalues.values
    train_true = train_target.values
    train_resid_fitted = train_true - train_fitted
    train_garch_fitted_vol = hybrid.garch_fitted.conditional_volatility.values / 100
    train_std_fitted = train_resid_fitted / train_garch_fitted_vol
    
    hybrid.predict_hybrid(
        train_data, 'train',
        train_fitted,
        train_std_fitted,
        train_garch_fitted_vol
    )
    
    # Evaluate
    metrics = hybrid.evaluate()
    
    # Save
    output_dir = hybrid.save_results()
    hybrid.save_models()
    
    print("\n" + "="*80)
    print("COMPLETE HYBRID TRAINING FINISHED")
    print("="*80)
    print(f"Results: {output_dir}")
    
    return hybrid, metrics


if __name__ == "__main__":
    model, metrics = main()

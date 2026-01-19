"""
ARIMA-LSTM Hybrid Model for FOREX Forecasting

This hybrid combines:
- ARIMA: Captures linear autocorrelation patterns
- LSTM: Learns non-linear relationships and complex patterns

Architecture:
1. ARIMA generates residuals (what it can't explain)
2. LSTM learns to predict these residuals
3. Final prediction = ARIMA forecast + LSTM residual correction

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
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("⚠️ pmdarima not available. Using fixed ARIMA order.")

# Deep learning
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Project imports
from src.utils.config import (
    SAVED_MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR,
    RANDOM_SEED, LSTM_CONFIG
)

# Set seeds
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class ARIMALSTMHybrid:
    """
    ARIMA-LSTM Hybrid Model.
    
    Step 1: Fit ARIMA on training data
    Step 2: Extract ARIMA residuals (errors)
    Step 3: Train LSTM to predict these residuals
    Step 4: Final prediction = ARIMA forecast + LSTM residual prediction
    """
    
    def __init__(self, arima_order=(1, 0, 1), lstm_config=None):
        """
        Initialize hybrid model.
        
        Args:
            arima_order: (p, d, q) for ARIMA
            lstm_config: LSTM hyperparameters
        """
        self.arima_order = arima_order
        self.lstm_config = lstm_config or LSTM_CONFIG
        
        self.arima_model = None
        self.arima_fitted = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        
        self.predictions = {}
        self.metrics = {}
        
        print(f"[INFO] ARIMA-LSTM Hybrid initialized")
        print(f"  ARIMA order: {arima_order}")
        print(f"  LSTM units: {self.lstm_config['lstm_units']}")
    
    def fit_arima(self, train_data: pd.Series):
        """Fit ARIMA model on training data."""
        print("\n[ARIMA] Fitting on training data...")
        
        self.arima_model = ARIMA(train_data, order=self.arima_order)
        self.arima_fitted = self.arima_model.fit()
        
        print(f"[OK] ARIMA{self.arima_order} fitted")
        print(f"  AIC: {self.arima_fitted.aic:.2f}")
        print(f"  BIC: {self.arima_fitted.bic:.2f}")
        
        return self.arima_fitted
    
    def get_arima_residuals(self, data: pd.Series, fitted_values: pd.Series = None):
        """Calculate ARIMA residuals (what ARIMA cannot explain)."""
        if fitted_values is None:
            fitted_values = self.arima_fitted.fittedvalues
        
        residuals = data - fitted_values
        return residuals
    
    def create_lstm_sequences(self, residuals: np.ndarray, n_timesteps: int = 4):
        """Create sequences for LSTM from residuals."""
        X, y = [], []
        
        for i in range(n_timesteps, len(residuals)):
            X.append(residuals[i-n_timesteps:i])
            y.append(residuals[i])
        
        return np.array(X), np.array(y)
    
    def build_lstm(self, input_shape):
        """Build LSTM architecture."""
        model = Sequential([
            LSTM(self.lstm_config['lstm_units'][0], 
                 return_sequences=True if len(self.lstm_config['lstm_units']) > 1 else False,
                 input_shape=input_shape),
            Dropout(self.lstm_config['dropout_rate']),
        ])
        
        if len(self.lstm_config['lstm_units']) > 1:
            model.add(LSTM(self.lstm_config['lstm_units'][1]))
            model.add(Dropout(self.lstm_config['dropout_rate']))
        
        model.add(Dense(1))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lstm_config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit_lstm_on_residuals(self, train_residuals: np.ndarray, val_residuals: np.ndarray):
        """Train LSTM to predict ARIMA residuals."""
        print("\n[LSTM] Training on ARIMA residuals...")
        
        n_timesteps = self.lstm_config['n_timesteps']
        
        # Create sequences
        X_train, y_train = self.create_lstm_sequences(train_residuals, n_timesteps)
        X_val, y_val = self.create_lstm_sequences(val_residuals, n_timesteps)
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], n_timesteps, 1)
        X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], n_timesteps, 1)
        
        # Build LSTM
        self.lstm_model = self.build_lstm(input_shape=(n_timesteps, 1))
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
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
            min_lr=0.0001,
            verbose=1
        )
        
        # Train
        history = self.lstm_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.lstm_config['epochs'],
            batch_size=self.lstm_config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("[OK] LSTM trained on residuals")
        
        return history
    
    def predict_hybrid(self, data: pd.DataFrame, subset_name: str = 'test'):
        """
        Generate hybrid predictions.
        
        Final prediction = ARIMA forecast + LSTM residual correction
        """
        print(f"\n[HYBRID] Generating {subset_name} predictions...")
        
        target_col = 'Log_Returns'
        y_true = data[target_col].values
        
        # Step 1: ARIMA forecast
        n_steps = len(data)
        arima_forecast = self.arima_fitted.forecast(steps=n_steps)
        
        # Step 2: Calculate ARIMA residuals for LSTM
        # For test set, we need to use actual values to calculate residuals progressively
        arima_residuals = y_true - arima_forecast.values
        
        # Step 3: LSTM predicts residuals
        n_timesteps = self.lstm_config['n_timesteps']
        
        # Use available residuals to predict next residual
        lstm_residual_preds = []
        
        # Start with training residuals for warm-up
        train_residuals = self.get_arima_residuals(
            data[target_col].iloc[:n_timesteps].values,
            self.arima_fitted.fittedvalues[:n_timesteps].values
        )
        
        residual_history = list(train_residuals[-n_timesteps:])
        
        for i in range(len(arima_residuals)):
            # Create sequence
            X_seq = np.array(residual_history[-n_timesteps:]).reshape(1, n_timesteps, 1)
            X_seq_scaled = self.scaler.transform(X_seq.reshape(-1, 1)).reshape(1, n_timesteps, 1)
            
            # Predict residual
            lstm_pred = self.lstm_model.predict(X_seq_scaled, verbose=0)[0, 0]
            lstm_residual_preds.append(lstm_pred)
            
            # Update history with actual residual
            if i < len(arima_residuals):
                residual_history.append(arima_residuals[i])
        
        lstm_residual_preds = np.array(lstm_residual_preds)
        
        # Step 4: Combine ARIMA + LSTM
        hybrid_predictions = arima_forecast.values + lstm_residual_preds
        
        # Store results
        self.predictions[subset_name] = {
            'y_true': y_true,
            'y_pred': hybrid_predictions,
            'arima_forecast': arima_forecast.values,
            'lstm_residual': lstm_residual_preds
        }
        
        print(f"[OK] {subset_name} predictions generated")
        
        return hybrid_predictions
    
    def evaluate(self):
        """Calculate metrics for all subsets."""
        print("\n" + "="*70)
        print("EVALUATION METRICS")
        print("="*70)
        
        for subset_name, pred_dict in self.predictions.items():
            y_true = pred_dict['y_true']
            y_pred = pred_dict['y_pred']
            
            # Metrics
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
        """Save predictions and metrics."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = PREDICTIONS_DIR / f"arima_lstm_hybrid_{timestamp}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[INFO] Saving results to: {output_dir}")
        
        # Save predictions
        for subset_name, pred_dict in self.predictions.items():
            pred_df = pd.DataFrame({
                'y_true': pred_dict['y_true'],
                'y_pred': pred_dict['y_pred'],
                'arima_component': pred_dict['arima_forecast'],
                'lstm_component': pred_dict['lstm_residual'],
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
            'model_type': 'ARIMA-LSTM Hybrid',
            'arima_order': self.arima_order,
            'lstm_config': self.lstm_config,
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
        arima_path = SAVED_MODELS_DIR / "arima_lstm_hybrid_arima.pkl"
        with open(arima_path, 'wb') as f:
            pickle.dump(self.arima_fitted, f)
        print(f"[OK] ARIMA saved: {arima_path}")
        
        # Save LSTM
        lstm_path = SAVED_MODELS_DIR / "arima_lstm_hybrid_lstm.h5"
        self.lstm_model.save(lstm_path)
        print(f"[OK] LSTM saved: {lstm_path}")
        
        # Save scaler
        scaler_path = SAVED_MODELS_DIR / "arima_lstm_hybrid_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"[OK] Scaler saved: {scaler_path}")


def main():
    """Main execution."""
    print("="*80)
    print(" "*25 + "ARIMA-LSTM HYBRID MODEL")
    print("="*80)
    
    # Load data
    print("\n[INFO] Loading preprocessed data...")
    train_data = pd.read_csv(PROCESSED_DATA_DIR / "train_data.csv")
    val_data = pd.read_csv(PROCESSED_DATA_DIR / "val_data.csv")
    test_data = pd.read_csv(PROCESSED_DATA_DIR / "test_data.csv")
    
    print(f"  Train: {len(train_data):,} samples")
    print(f"  Val: {len(val_data):,} samples")
    print(f"  Test: {len(test_data):,} samples")
    
    # Initialize hybrid model
    hybrid = ARIMALSTMHybrid(arima_order=(1, 0, 1))
    
    # Step 1: Fit ARIMA
    train_target = train_data['Log_Returns'].dropna()
    hybrid.fit_arima(train_target)
    
    # Step 2: Get ARIMA residuals
    train_residuals = hybrid.get_arima_residuals(train_target)
    
    # For validation, forecast and calculate residuals
    val_target = val_data['Log_Returns'].dropna()
    val_forecast = hybrid.arima_fitted.forecast(steps=len(val_target))
    val_residuals = val_target.values - val_forecast.values
    
    # Step 3: Train LSTM on residuals
    hybrid.fit_lstm_on_residuals(train_residuals.values, val_residuals)
    
    # Step 4: Generate hybrid predictions
    hybrid.predict_hybrid(val_data, 'val')
    hybrid.predict_hybrid(test_data, 'test')
    
    # Also generate train predictions for completeness
    train_pred = hybrid.arima_fitted.fittedvalues
    hybrid.predictions['train'] = {
        'y_true': train_target.values,
        'y_pred': train_pred.values,
        'arima_forecast': train_pred.values,
        'lstm_residual': np.zeros(len(train_pred))
    }
    
    # Step 5: Evaluate
    metrics = hybrid.evaluate()
    
    # Step 6: Save results
    output_dir = hybrid.save_results()
    hybrid.save_models()
    
    print("\n" + "="*80)
    print("ARIMA-LSTM HYBRID TRAINING COMPLETE")
    print("="*80)
    print(f"Results: {output_dir}")
    
    return hybrid, metrics


if __name__ == "__main__":
    model, metrics = main()

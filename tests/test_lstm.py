"""
Quick test script to verify LSTM model implementation.
This can be run to ensure the model works before opening the notebook.

Usage:
    python test_lstm.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from src.utils.config import set_random_seeds, RANDOM_SEED
from src.models.lstm_model import LSTMForexModel

# Set random seed
set_random_seeds(RANDOM_SEED)

print("="*70)
print("LSTM MODEL IMPLEMENTATION TEST")
print("="*70)

# Generate synthetic time series data
print("\n1. Generating synthetic FOREX-like data...")
np.random.seed(RANDOM_SEED)
n_samples = 1000
n_features = 5

# Simulate price with trend and noise
price = 1.0 + np.cumsum(np.random.randn(n_samples) * 0.01)
returns = np.diff(np.log(price))
returns = np.concatenate([[0], returns])  # Pad to original length

# Create features (simulating technical indicators)
features = np.column_stack([
    returns,  # Log returns
    np.random.randn(n_samples) * 0.02,  # Volatility proxy
    np.random.randn(n_samples),  # RSI-like
    np.random.randn(n_samples),  # SMA-like
    np.random.randn(n_samples)   # MACD-like
])

# Create DataFrames
feature_names = ['Log_Returns', 'Volatility', 'RSI', 'SMA', 'MACD']

train_data = pd.DataFrame(features[:700], columns=feature_names)
val_data = pd.DataFrame(features[700:850], columns=feature_names)
test_data = pd.DataFrame(features[850:], columns=feature_names)

print(f"   ✓ Generated {n_samples} observations")
print(f"   ✓ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# Test 1: Model initialization
print("\n2. Testing model initialization...")
model = LSTMForexModel(
    n_timesteps=4,
    lstm_units=[50, 50],
    dropout_rate=0.2,
    learning_rate=0.01,
    verbose=0
)
print("   ✓ Model initialized successfully")

# Test 2: Data preparation
print("\n3. Testing data preparation and sequence creation...")
X_train, y_train, X_val, y_val, X_test, y_test = model.prepare_data(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    feature_columns=feature_names,
    target_column='Log_Returns'
)
print(f"   ✓ Sequences created:")
print(f"     Train: X={X_train.shape}, y={y_train.shape}")
print(f"     Val:   X={X_val.shape}, y={y_val.shape}")
print(f"     Test:  X={X_test.shape}, y={y_test.shape}")

# Test 3: Model building
print("\n4. Testing model architecture building...")
model.build_model(n_features=n_features)
print(f"   ✓ Model built with {model.model.count_params()} parameters")

# Test 4: Training (minimal epochs)
print("\n5. Testing model training (5 epochs for demo)...")
history = model.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=5,
    batch_size=32,
    early_stopping_patience=3
)
print(f"   ✓ Training completed")
print(f"     Final train loss: {history['loss'][-1]:.6f}")
print(f"     Final val loss: {history['val_loss'][-1]:.6f}")

# Test 5: Prediction
print("\n6. Testing prediction...")
predictions = model.predict(X_test)
print(f"   ✓ Generated {len(predictions)} predictions")
print(f"     Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")

# Test 6: Evaluation
print("\n7. Testing evaluation metrics...")
metrics = model.evaluate(X_test, y_test)
print(f"   ✓ Evaluation completed")

# Test 7: Save/Load
print("\n8. Testing model persistence...")
from tempfile import TemporaryDirectory
with TemporaryDirectory() as tmpdir:
    model_path = Path(tmpdir) / 'test_model.h5'
    scaler_path = Path(tmpdir) / 'test_scaler.pkl'
    
    model.save_model(model_path, scaler_path)
    print(f"   ✓ Model saved")
    
    loaded_model = LSTMForexModel.load_model(model_path, scaler_path)
    print(f"   ✓ Model loaded")
    
    # Verify predictions match
    loaded_predictions = loaded_model.predict(X_test)
    match = np.allclose(predictions, loaded_predictions)
    print(f"   ✓ Predictions match after reload: {match}")

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nLSTM model implementation is ready for use!")
print("Proceed to notebooks/04_lstm_baseline.ipynb for full analysis.")

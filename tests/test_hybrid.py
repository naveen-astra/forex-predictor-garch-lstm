"""
Test Script for Hybrid GARCH-LSTM Model

Validates Phase 4 implementation:
    - GARCH volatility loading
    - Feature integration
    - Hybrid model training
    - Comparative evaluation

Usage:
    python tests/test_hybrid.py

Expected Runtime: ~2-3 minutes

Author: Research Team
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from pathlib import Path
from src.models.hybrid_garch_lstm import HybridGARCHLSTM, compare_models


def test_hybrid_initialization():
    """Test 1: Hybrid model initialization"""
    print("\n" + "="*70)
    print("TEST 1: Hybrid Model Initialization")
    print("="*70)
    
    try:
        model = HybridGARCHLSTM(
            n_timesteps=4,
            lstm_units=[200, 200],
            dropout_rate=0.2,
            learning_rate=0.01
        )
        print("✓ Hybrid model initialized successfully")
        print(f"  - LSTM timesteps: {model.lstm_model.n_timesteps}")
        print(f"  - LSTM layers: {len(model.lstm_model.lstm_units)}")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_garch_volatility_loading():
    """Test 2: Load GARCH volatility from Phase 2"""
    print("\n" + "="*70)
    print("TEST 2: Load GARCH Volatility")
    print("="*70)
    
    try:
        # Define paths
        output_dir = Path('../output')
        train_path = output_dir / 'train_data_with_garch.csv'
        val_path = output_dir / 'val_data_with_garch.csv'
        test_path = output_dir / 'test_data_with_garch.csv'
        
        # Check if files exist
        if not train_path.exists():
            print(f"✗ File not found: {train_path}")
            print("  Run Phase 2 (GARCH modeling) first to generate these files.")
            return False
        
        # Load data
        model = HybridGARCHLSTM()
        train_data, val_data, test_data = model.load_garch_volatility(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path
        )
        
        print("✓ GARCH volatility loaded successfully")
        print(f"  - Train volatility shape: {model.garch_volatility_train.shape}")
        print(f"  - Val volatility shape: {model.garch_volatility_val.shape}")
        print(f"  - Test volatility shape: {model.garch_volatility_test.shape}")
        
        return True, train_data, val_data, test_data
        
    except Exception as e:
        print(f"✗ GARCH loading failed: {e}")
        return False, None, None, None


def test_hybrid_feature_preparation(train_data, val_data, test_data):
    """Test 3: Prepare hybrid feature set"""
    print("\n" + "="*70)
    print("TEST 3: Hybrid Feature Preparation")
    print("="*70)
    
    try:
        # Define base features
        base_features = [
            'Open', 'High', 'Low', 'Close',
            'Log_Returns', 'Log_Returns_Lag1', 'Daily_Return',
            'MA_7', 'MA_14', 'MA_30',
            'Rolling_Std_7', 'Rolling_Std_14', 'Rolling_Std_30'
        ]
        
        model = HybridGARCHLSTM()
        model.garch_volatility_train = train_data['GARCH_Volatility']
        model.garch_volatility_val = val_data['GARCH_Volatility']
        model.garch_volatility_test = test_data['GARCH_Volatility']
        
        train_hybrid, val_hybrid, test_hybrid = model.prepare_hybrid_features(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            base_features=base_features
        )
        
        print("✓ Hybrid features prepared successfully")
        print(f"  - Train shape: {train_hybrid.shape}")
        print(f"  - Val shape: {val_hybrid.shape}")
        print(f"  - Test shape: {test_hybrid.shape}")
        print(f"  - Total features: {len(model.feature_columns)}")
        print(f"  - GARCH volatility included: {'GARCH_Volatility' in model.feature_columns}")
        
        return True, train_hybrid, val_hybrid, test_hybrid
        
    except Exception as e:
        print(f"✗ Feature preparation failed: {e}")
        return False, None, None, None


def test_hybrid_training(train_hybrid, val_hybrid, test_hybrid):
    """Test 4: Train hybrid model (limited epochs for testing)"""
    print("\n" + "="*70)
    print("TEST 4: Hybrid Model Training (5 epochs for testing)")
    print("="*70)
    
    try:
        model = HybridGARCHLSTM(
            n_timesteps=4,
            lstm_units=[50, 50],  # Smaller for faster testing
            dropout_rate=0.2,
            learning_rate=0.01,
            verbose=0
        )
        
        # Set feature columns
        model.feature_columns = train_hybrid.columns.tolist()
        
        # Train with minimal epochs
        history = model.train_hybrid_model(
            train_data=train_hybrid,
            val_data=val_hybrid,
            test_data=test_hybrid,
            target_column='Log_Returns',
            epochs=5,  # Just 5 epochs for testing
            batch_size=32,
            early_stopping_patience=10
        )
        
        print("✓ Training completed successfully")
        print(f"  - Epochs run: {len(history['loss'])}")
        print(f"  - Final train loss: {history['loss'][-1]:.6f}")
        print(f"  - Final val loss: {history['val_loss'][-1]:.6f}")
        
        return True, model
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False, None


def test_hybrid_evaluation(model):
    """Test 5: Evaluate hybrid model"""
    print("\n" + "="*70)
    print("TEST 5: Hybrid Model Evaluation")
    print("="*70)
    
    try:
        metrics = model.evaluate_hybrid()
        
        print("✓ Evaluation completed successfully")
        print(f"\nTest Set Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric:25s}: {value:.6f}")
        
        return True, metrics
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False, None


def test_model_comparison():
    """Test 6: Model comparison functionality"""
    print("\n" + "="*70)
    print("TEST 6: Model Comparison")
    print("="*70)
    
    try:
        # Mock metrics for testing
        garch_metrics = {
            'MSE': 0.0001,
            'MAE': 0.008,
            'RMSE': 0.01,
            'Directional_Accuracy': 52.0
        }
        
        lstm_metrics = {
            'MSE': 0.00008,
            'MAE': 0.007,
            'RMSE': 0.009,
            'Directional_Accuracy': 53.5
        }
        
        hybrid_metrics = {
            'MSE': 0.00007,
            'MAE': 0.0065,
            'RMSE': 0.0084,
            'Directional_Accuracy': 54.2
        }
        
        comparison_df = compare_models(garch_metrics, lstm_metrics, hybrid_metrics)
        
        print("✓ Model comparison completed successfully")
        print(f"  - Comparison table shape: {comparison_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        return False


def test_save_load_model(model):
    """Test 7: Save and load hybrid model"""
    print("\n" + "="*70)
    print("TEST 7: Save and Load Model")
    print("="*70)
    
    try:
        output_dir = Path('../output')
        output_dir.mkdir(exist_ok=True)
        
        model_path = output_dir / 'test_hybrid_model.keras'
        scaler_path = output_dir / 'test_hybrid_scaler.pkl'
        
        # Save model
        model.save_model(model_path, scaler_path)
        print(f"✓ Model saved to {model_path}")
        
        # Load model
        loaded_model = HybridGARCHLSTM.load_model(model_path, scaler_path)
        print(f"✓ Model loaded successfully")
        
        # Clean up test files
        model_path.unlink()
        scaler_path.unlink()
        print("✓ Test files cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Save/Load failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("HYBRID GARCH-LSTM MODEL TEST SUITE")
    print("Phase 4: Testing Hybrid Model Implementation")
    print("="*70)
    
    results = []
    
    # Test 1: Initialization
    results.append(test_hybrid_initialization())
    
    # Test 2: Load GARCH volatility
    garch_result = test_garch_volatility_loading()
    if isinstance(garch_result, tuple):
        success, train_data, val_data, test_data = garch_result
        results.append(success)
    else:
        results.append(garch_result)
        train_data = val_data = test_data = None
    
    # Test 3: Feature preparation
    if train_data is not None:
        feature_result = test_hybrid_feature_preparation(train_data, val_data, test_data)
        if isinstance(feature_result, tuple):
            success, train_hybrid, val_hybrid, test_hybrid = feature_result
            results.append(success)
        else:
            results.append(feature_result)
            train_hybrid = val_hybrid = test_hybrid = None
    else:
        print("\n⚠ Skipping feature preparation test (no data)")
        results.append(None)
        train_hybrid = val_hybrid = test_hybrid = None
    
    # Test 4: Training
    if train_hybrid is not None:
        train_result = test_hybrid_training(train_hybrid, val_hybrid, test_hybrid)
        if isinstance(train_result, tuple):
            success, model = train_result
            results.append(success)
        else:
            results.append(train_result)
            model = None
    else:
        print("\n⚠ Skipping training test (no features)")
        results.append(None)
        model = None
    
    # Test 5: Evaluation
    if model is not None:
        eval_result = test_hybrid_evaluation(model)
        if isinstance(eval_result, tuple):
            success, metrics = eval_result
            results.append(success)
        else:
            results.append(eval_result)
    else:
        print("\n⚠ Skipping evaluation test (no model)")
        results.append(None)
    
    # Test 6: Comparison
    results.append(test_model_comparison())
    
    # Test 7: Save/Load
    if model is not None:
        results.append(test_save_load_model(model))
    else:
        print("\n⚠ Skipping save/load test (no model)")
        results.append(None)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum([1 for r in results if r is True])
    failed = sum([1 for r in results if r is False])
    skipped = sum([1 for r in results if r is None])
    total = len(results)
    
    print(f"Total tests:   {total}")
    print(f"Passed:        {passed} ✓")
    print(f"Failed:        {failed} ✗")
    print(f"Skipped:       {skipped} ⚠")
    
    if failed == 0 and passed > 0:
        print("\n✓ All available tests passed!")
        print("\nNext steps:")
        print("  1. Run full training: jupyter notebook notebooks/05_hybrid_garch_lstm.ipynb")
        print("  2. Compare with baselines")
        print("  3. Generate visualizations")
    elif skipped > 0:
        print("\n⚠ Some tests were skipped.")
        print("  Make sure you have completed Phase 2 (GARCH modeling) first.")
        print("  The required files are:")
        print("    - output/train_data_with_garch.csv")
        print("    - output/val_data_with_garch.csv")
        print("    - output/test_data_with_garch.csv")
    else:
        print("\n✗ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    run_all_tests()

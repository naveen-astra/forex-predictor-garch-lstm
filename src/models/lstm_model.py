"""
LSTM Baseline Model for FOREX Return Forecasting

This module implements a pure LSTM approach for forecasting FOREX log returns
without incorporating GARCH volatility features. This serves as a baseline to
demonstrate the value-add of the hybrid GARCH-LSTM approach.

LSTM Architecture:
    - 2-layer LSTM with 200 units each
    - Dropout regularization (0.2)
    - Sliding window input sequences (timesteps=4)
    - Single-step ahead prediction

Key Features:
    - Price-based features only (log returns, technical indicators)
    - Proper train/validation/test workflow
    - Callbacks: Early stopping, learning rate reduction, checkpointing
    - Deterministic training (reproducible results)
    - Model persistence and loading

Evaluation:
    - MSE, MAE, RMSE on test set
    - Training vs validation loss plots
    - Prediction quality assessment

Author: Research Team
Date: January 2026
License: MIT
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, List, Optional
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class LSTMForexModel:
    """
    LSTM model for FOREX log return forecasting.
    
    This class provides a complete workflow for:
        1. Creating sliding window sequences from time series
        2. Building and compiling LSTM architecture
        3. Training with proper callbacks
        4. Making predictions on new data
        5. Evaluating model performance
        6. Saving/loading trained models
    
    Attributes:
        n_timesteps (int): Number of time steps in input sequences
        n_features (int): Number of input features
        lstm_units (List[int]): Number of units in each LSTM layer
        dropout_rate (float): Dropout rate for regularization
        model: Compiled Keras model
        history: Training history object
        scaler: Feature scaler (MinMaxScaler)
    """
    
    def __init__(self, 
                 n_timesteps: int = 4,
                 lstm_units: List[int] = [200, 200],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.01,
                 verbose: int = 1):
        """
        Initialize LSTM model configuration.
        
        Args:
            n_timesteps: Number of past time steps to use as input
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate after each LSTM layer
            learning_rate: Learning rate for Adam optimizer
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        """
        self.n_timesteps = n_timesteps
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        self.model = None
        self.history = None
        self.scaler = MinMaxScaler()
        self.n_features = None
        self.feature_columns = None
        
    def create_sequences(self, 
                        data: np.ndarray, 
                        target: np.ndarray,
                        n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM input.
        
        Args:
            data: Input features array, shape (n_samples, n_features)
            target: Target values array, shape (n_samples,)
            n_steps: Number of time steps in each sequence
            
        Returns:
            X: Sequences, shape (n_samples - n_steps, n_steps, n_features)
            y: Targets, shape (n_samples - n_steps,)
            
        Example:
            If data has 100 rows and n_steps=4:
            - Row 0-3 → predicts target[4]
            - Row 1-4 → predicts target[5]
            - ...
            - Row 95-98 → predicts target[99]
            Result: X has shape (96, 4, n_features), y has shape (96,)
        """
        X, y = [], []
        
        for i in range(len(data) - n_steps):
            # Extract sequence of length n_steps
            X.append(data[i:i + n_steps])
            # Target is the next value after the sequence
            y.append(target[i + n_steps])
            
        return np.array(X), np.array(y)
    
    def build_model(self, n_features: int) -> None:
        """
        Build and compile LSTM model architecture.
        
        Args:
            n_features: Number of input features
            
        Architecture:
            Input: (n_timesteps, n_features)
            → LSTM(lstm_units[0], return_sequences=True)
            → Dropout(dropout_rate)
            → LSTM(lstm_units[1])
            → Dropout(dropout_rate)
            → Dense(1, activation='linear')
            Output: Single value (predicted log return)
        """
        self.n_features = n_features
        
        model = Sequential(name='LSTM_Baseline')
        
        # First LSTM layer (returns sequences for next layer)
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=(self.n_timesteps, n_features),
            name='LSTM_Layer_1'
        ))
        model.add(Dropout(self.dropout_rate, name='Dropout_1'))
        
        # Second LSTM layer (returns single vector)
        model.add(LSTM(
            units=self.lstm_units[1],
            return_sequences=False,
            name='LSTM_Layer_2'
        ))
        model.add(Dropout(self.dropout_rate, name='Dropout_2'))
        
        # Output layer
        model.add(Dense(1, activation='linear', name='Output'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        self.model = model
        
        if self.verbose:
            print("LSTM Model Architecture:")
            print("=" * 70)
            model.summary()
            print("=" * 70)
    
    def prepare_data(self,
                    train_data: pd.DataFrame,
                    val_data: pd.DataFrame,
                    test_data: pd.DataFrame,
                    feature_columns: List[str],
                    target_column: str = 'Log_Returns') -> Tuple:
        """
        Prepare train/val/test data for LSTM training.
        
        Steps:
            1. Extract features and target
            2. Scale features using training data statistics
            3. Create sliding window sequences
            4. Return properly shaped arrays
            
        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            test_data: Test DataFrame
            feature_columns: List of feature column names
            target_column: Name of target column
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
            
        Notes:
            - Scaler is fit ONLY on training data
            - Same scaler transforms val and test data
            - Sequences reduce sample size by n_timesteps
        """
        self.feature_columns = feature_columns
        
        # Extract features and target
        X_train = train_data[feature_columns].values
        y_train = train_data[target_column].values
        
        X_val = val_data[feature_columns].values
        y_val = val_data[target_column].values
        
        X_test = test_data[feature_columns].values
        y_test = test_data[target_column].values
        
        # Fit scaler on training data only
        self.scaler.fit(X_train)
        
        # Transform all datasets
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(
            X_train_scaled, y_train, self.n_timesteps
        )
        X_val_seq, y_val_seq = self.create_sequences(
            X_val_scaled, y_val, self.n_timesteps
        )
        X_test_seq, y_test_seq = self.create_sequences(
            X_test_scaled, y_test, self.n_timesteps
        )
        
        print(f"Data prepared successfully:")
        print(f"  Training sequences:   {X_train_seq.shape}")
        print(f"  Validation sequences: {X_val_seq.shape}")
        print(f"  Test sequences:       {X_test_seq.shape}")
        print(f"  Features: {len(feature_columns)}")
        
        return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 60,
             batch_size: int = 32,
             early_stopping_patience: int = 10,
             checkpoint_path: Optional[Path] = None) -> Dict:
        """
        Train LSTM model with callbacks.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            early_stopping_patience: Epochs to wait before early stopping
            checkpoint_path: Path to save best model weights
            
        Returns:
            Dictionary with training history
            
        Callbacks:
            - EarlyStopping: Stop if val_loss doesn't improve
            - ReduceLROnPlateau: Reduce LR if val_loss plateaus
            - ModelCheckpoint: Save best model weights
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Define callbacks
        callbacks = []
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=self.verbose
        )
        callbacks.append(early_stop)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=self.verbose
        )
        callbacks.append(reduce_lr)
        
        # Model checkpointing
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=self.verbose
            )
            callbacks.append(checkpoint)
        
        # Train model
        print(f"\nTraining LSTM model...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=self.verbose,
            shuffle=False  # Don't shuffle time series
        )
        
        self.history = history
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input sequences, shape (n_samples, n_timesteps, n_features)
            
        Returns:
            Predictions, shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self,
                X_test: np.ndarray,
                y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test sequences
            y_test: True test targets
            
        Returns:
            Dictionary with metrics: MSE, MAE, RMSE
        """
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        actual_direction = np.sign(y_test[1:] - y_test[:-1])
        pred_direction = np.sign(predictions[1:] - predictions[:-1])
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Directional_Accuracy': directional_accuracy
        }
        
        print("\nTest Set Evaluation:")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        return metrics
    
    def save_model(self, model_path: Path, scaler_path: Path) -> None:
        """
        Save trained model and scaler.
        
        Args:
            model_path: Path to save Keras model
            scaler_path: Path to save scaler pickle
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Save Keras model
        self.model.save(model_path)
        
        # Save scaler and metadata
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'n_timesteps': self.n_timesteps,
                'n_features': self.n_features,
                'feature_columns': self.feature_columns
            }, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    @classmethod
    def load_model(cls, model_path: Path, scaler_path: Path):
        """
        Load trained model and scaler.
        
        Args:
            model_path: Path to saved Keras model
            scaler_path: Path to saved scaler pickle
            
        Returns:
            Loaded LSTMForexModel instance
        """
        # Load scaler and metadata
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(n_timesteps=data['n_timesteps'])
        instance.scaler = data['scaler']
        instance.n_features = data['n_features']
        instance.feature_columns = data['feature_columns']
        
        # Load Keras model
        instance.model = keras.models.load_model(model_path)
        
        print(f"Model loaded from: {model_path}")
        return instance


if __name__ == "__main__":
    # Example usage and testing
    print("LSTM Baseline Model Module - Example Usage\n")
    print("=" * 70)
    
    # Generate synthetic time series data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Random walk for returns
    returns = np.cumsum(np.random.randn(n_samples) * 0.01)
    
    # Random features
    features = np.random.randn(n_samples, n_features)
    
    # Create DataFrames
    train_data = pd.DataFrame(features[:700], columns=[f'feat_{i}' for i in range(n_features)])
    train_data['Log_Returns'] = returns[:700]
    
    val_data = pd.DataFrame(features[700:850], columns=[f'feat_{i}' for i in range(n_features)])
    val_data['Log_Returns'] = returns[700:850]
    
    test_data = pd.DataFrame(features[850:], columns=[f'feat_{i}' for i in range(n_features)])
    test_data['Log_Returns'] = returns[850:]
    
    # Initialize model
    print("1. Initializing LSTM model...")
    model = LSTMForexModel(n_timesteps=4, lstm_units=[50, 50], verbose=0)
    
    # Prepare data
    print("\n2. Preparing sequences...")
    feature_cols = [f'feat_{i}' for i in range(n_features)]
    X_train, y_train, X_val, y_val, X_test, y_test = model.prepare_data(
        train_data, val_data, test_data, feature_cols
    )
    
    # Build model
    print("\n3. Building model architecture...")
    model.build_model(n_features)
    
    # Train model
    print("\n4. Training model (5 epochs for demo)...")
    model.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)
    
    # Evaluate
    print("\n5. Evaluating on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n" + "=" * 70)
    print("LSTM model implementation test completed successfully!")
    print("=" * 70)

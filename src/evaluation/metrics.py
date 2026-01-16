"""
Model evaluation metrics module for FOREX prediction.

This module provides standardized evaluation metrics for:
- Regression metrics (RMSE, MAE, MAPE, R²)
- Directional accuracy
- Statistical tests (Diebold-Mariano)
- Financial metrics (Sharpe ratio, max drawdown)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAPE value (in percentage)
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        R² value
    """
    return r2_score(y_true, y_pred)


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy (0-100%)
    """
    actual_direction = np.sign(np.diff(y_true))
    predicted_direction = np.sign(np.diff(y_pred))
    
    correct_predictions = np.sum(actual_direction == predicted_direction)
    total_predictions = len(actual_direction)
    
    return (correct_predictions / total_predictions) * 100


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   model_name: str = "Model") -> Dict[str, float]:
    """
    Compute all evaluation metrics for a model.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model being evaluated
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'Model': model_name,
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'Directional_Accuracy': calculate_directional_accuracy(y_true, y_pred)
    }
    
    return metrics


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple models' performance.
    
    Args:
        results: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame with comparison of all models
    """
    df = pd.DataFrame(results).T
    df = df.sort_values('RMSE')  # Sort by RMSE (lower is better)
    
    return df


# Placeholder for Diebold-Mariano test
def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray) -> Tuple[float, float]:
    """
    Perform Diebold-Mariano test for forecast comparison.
    
    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        
    Returns:
        Tuple of (test statistic, p-value)
        
    Note:
        To be implemented with statistical significance testing
    """
    # TODO: Implement DM test
    raise NotImplementedError("Diebold-Mariano test to be implemented")


if __name__ == "__main__":
    # Example usage
    print("Evaluation metrics module loaded successfully")
    
    # Test with dummy data
    y_true = np.array([1.1, 1.2, 1.15, 1.18, 1.22])
    y_pred = np.array([1.12, 1.19, 1.16, 1.17, 1.21])
    
    metrics = evaluate_model(y_true, y_pred, "Test Model")
    print("\nExample metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

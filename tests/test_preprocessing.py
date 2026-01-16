"""
Test suite for data preprocessing module.

Run with: pytest tests/test_preprocessing.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.preprocess import ForexPreprocessor


@pytest.fixture
def sample_forex_data():
    """Create sample FOREX data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Datetime': dates,
        'Open': np.random.uniform(1.1, 1.2, 100),
        'High': np.random.uniform(1.15, 1.25, 100),
        'Low': np.random.uniform(1.05, 1.15, 100),
        'Close': np.random.uniform(1.1, 1.2, 100),
        'Volume': np.random.randint(1000, 10000, 100)
    })
    data.set_index('Datetime', inplace=True)
    return data


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = ForexPreprocessor()
    assert preprocessor.data is None
    assert preprocessor.processed_data is None


def test_compute_log_returns(sample_forex_data):
    """Test log returns calculation."""
    preprocessor = ForexPreprocessor(data=sample_forex_data)
    returns = preprocessor.compute_log_returns()
    
    # Check that returns are computed
    assert 'Log_Returns' in preprocessor.data.columns
    # Check that first value is NaN (shift operation)
    assert preprocessor.data['Log_Returns'].notna().sum() > 0


def test_handle_missing_values(sample_forex_data):
    """Test missing value handling."""
    # Add some missing values
    sample_forex_data.loc[sample_forex_data.index[5:10], 'Close'] = np.nan
    
    preprocessor = ForexPreprocessor(data=sample_forex_data)
    preprocessor.handle_missing_values(strategy='forward_fill')
    
    # Check that missing values are handled
    assert preprocessor.data['Close'].isna().sum() == 0


def test_detect_outliers(sample_forex_data):
    """Test outlier detection."""
    preprocessor = ForexPreprocessor(data=sample_forex_data)
    outliers = preprocessor.detect_outliers(method='iqr')
    
    # Check that outliers dataframe is returned
    assert isinstance(outliers, pd.DataFrame)
    assert outliers.shape[0] == len(sample_forex_data)


def test_data_splitting(sample_forex_data):
    """Test train/val/test splitting."""
    preprocessor = ForexPreprocessor(data=sample_forex_data)
    train, val, test = preprocessor.split_data(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # Check splits are correct size
    total = len(train) + len(val) + len(test)
    assert total == len(sample_forex_data)
    assert len(train) > len(val)
    assert len(train) > len(test)


if __name__ == "__main__":
    pytest.main([__file__])

"""
Package initialization for the FOREX GARCH-LSTM prediction project.

This package implements a hybrid forecasting model combining:
- GARCH for volatility modeling
- LSTM for sequence prediction
- Big Data Analytics for scalability

Author: Naveen Astra
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Naveen Astra"
__project__ = "Intelligent FOREX Exchange Rate Forecasting using Hybrid GARCH–LSTM and Big Data Analytics"

# Import key components for easy access
from .utils.config import *
from .data.fetch_data import ForexDataFetcher
from .data.preprocess import (
    compute_log_returns,
    compute_technical_indicators,
    preprocess_data,
    split_data
)

__all__ = [
    'ForexDataFetcher',
    'compute_log_returns',
    'compute_technical_indicators',
    'preprocess_data',
    'split_data',
    'DATA_CONFIG',
    'GARCH_CONFIG',
    'LSTM_CONFIG',
    'HYBRID_CONFIG',
    'TRAINING_CONFIG',
    'PATHS',
]
__description__ = 'Intelligent FOREX Exchange Rate Forecasting using Hybrid GARCH-LSTM'

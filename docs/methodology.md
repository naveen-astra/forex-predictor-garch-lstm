# Methodology Documentation

## Overview
This document describes the methodology for the Hybrid GARCH-LSTM FOREX forecasting system.

## 1. Data Preparation

### 1.1 Data Source
- **Currency Pair**: EUR/USD
- **Source**: Yahoo Finance (primary), Alpha Vantage (fallback)
- **Period**: 2010-01-01 to 2025-12-31
- **Frequency**: Daily

### 1.2 Preprocessing Steps
1. Missing value imputation (forward fill)
2. Outlier detection (IQR method, multiplier=3.0)
3. Log transformation of returns
4. Feature engineering (technical indicators)
5. Chronological train/validation/test split (70/15/15)

## 2. Feature Engineering

### 2.1 Price-based Features
- Log returns: ln(P_t / P_{t-1})
- Log trading range: ln(High) - ln(Low)

### 2.2 Technical Indicators
- RSI (Relative Strength Index) - 14 days
- SMA (Simple Moving Average) - 14, 50, 200 days
- EMA (Exponential Moving Average) - 14, 26 days
- MACD (Moving Average Convergence Divergence) - (12, 26, 9)

### 2.3 Volatility Features
- Rolling volatility: 10, 30, 60-day windows
- Computed as standard deviation of log returns

## 3. Model Architecture (To be implemented)

### 3.1 GARCH Component
- Model: GARCH(1,1)
- Input: Log returns
- Output: Conditional variance estimates
- Implementation: PyFlux or arch package

### 3.2 LSTM Component
- Architecture: 2-layer LSTM (200 units each)
- Dropout: 0.2
- Timesteps: 4
- Activation: tanh (hidden), linear (output)

### 3.3 Hybrid Integration
- GARCH predictions as additional LSTM features
- Combined feature vector: [original_features, GARCH_volatility]

## 4. Evaluation Metrics

### 4.1 Accuracy Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

### 4.2 Directional Accuracy
- Percentage of correct directional predictions

### 4.3 Statistical Tests
- Diebold-Mariano test for forecast comparison
- Model Confidence Set (MCS)

## 5. Baseline Models

For comparison:
1. Naive (random walk)
2. ARIMA
3. GARCH-only
4. LSTM-only
5. Hybrid GARCH-LSTM (proposed)

## 6. Reproducibility

- Random seed: 42 (all libraries)
- Deterministic operations: Enabled
- Version control: All dependencies pinned
- Data provenance: Fully documented

## References

1. Original GARCH-LSTM paper: [Link to paper]
2. Reference implementation: https://github.com/tlemenestrel/LSTM_GARCH

# 7-Model Comprehensive Comparison System

This document explains all 7 time series forecasting models implemented in this project for EUR/USD FOREX prediction.

---

## 📊 Model Overview

| Model | Type | Components | Purpose |
|-------|------|------------|---------|
| **ARIMA** | Classical | Autoregressive + Moving Average | Linear time series baseline |
| **GARCH** | Volatility | Conditional heteroscedasticity | Volatility modeling |
| **LSTM** | Deep Learning | Recurrent neural network | Non-linear pattern recognition |
| **GARCH-LSTM** | 2-way Hybrid | GARCH volatility → LSTM | Volatility-aware deep learning |
| **ARIMA-LSTM** | 2-way Hybrid | ARIMA + LSTM residual correction | Linear + non-linear patterns |
| **ARIMA-GARCH** | 2-way Hybrid | ARIMA mean + GARCH variance | Classical econometric decomposition |
| **ARIMA-GARCH-LSTM** | 3-way Hybrid | All three combined | Complete hybrid architecture |

---

## 🔬 Model Details

### 1. ARIMA (Autoregressive Integrated Moving Average)

**File:** `src/models/arima_model.py`

**Architecture:**
- Order: (p, d, q) = (1, 0, 1) - automatically selected
- AR(1): Uses 1 past observation
- MA(1): Uses 1 past error term
- d=0: No differencing (log returns already stationary)

**Strengths:**
- Interpretable parameters
- Captures linear autocorrelation
- Fast training
- Well-established theory

**Weaknesses:**
- Assumes linearity
- Cannot model volatility clustering
- Limited for complex patterns

**Use Case:** Baseline for linear time series patterns

---

### 2. GARCH (Generalized Autoregressive Conditional Heteroscedasticity)

**File:** `src/models/garch_model.py`

**Architecture:**
- GARCH(1, 1) model
- Variance equation: σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1}
- Models time-varying volatility

**Strengths:**
- Captures volatility clustering (high volatility follows high volatility)
- Models fat tails in return distribution
- Provides prediction intervals

**Weaknesses:**
- Doesn't directly forecast returns
- Assumes symmetric response to shocks
- Requires stationary data

**Use Case:** Volatility forecasting, risk management

---

### 3. LSTM (Long Short-Term Memory)

**File:** `src/models/lstm_model.py`

**Architecture:**
- 2 LSTM layers × 200 units each
- Dropout: 0.2 after each LSTM layer
- Input: 13 price-based features, 4 timesteps
- Dense output layer

**Strengths:**
- Learns non-linear patterns
- Handles long-term dependencies
- No distributional assumptions

**Weaknesses:**
- Black box (hard to interpret)
- Requires large data
- Longer training time
- Can overfit

**Use Case:** Complex non-linear pattern recognition

---

### 4. Hybrid GARCH-LSTM

**File:** `src/models/hybrid_garch_lstm.py`

**Architecture:**
1. Run GARCH on log returns → get conditional volatility
2. Add GARCH volatility as 14th feature to LSTM
3. LSTM learns with volatility awareness

**Strengths:**
- LSTM becomes volatility-aware
- Combines volatility modeling with deep learning
- Best of both worlds: interpretable volatility + flexible DL

**Weaknesses:**
- More complex pipeline
- GARCH assumptions still apply
- Increased computational cost

**Use Case:** Volatility-aware forecasting, achieved 86.20% directional accuracy

**Results:** 55.6% improvement over LSTM baseline

---

### 5. ARIMA-LSTM Hybrid

**File:** `src/models/arima_lstm_hybrid.py`

**Architecture:**
1. **Stage 1:** Fit ARIMA on training data
2. **Stage 2:** Calculate ARIMA residuals (what ARIMA can't explain)
3. **Stage 3:** Train LSTM to predict residuals
4. **Final:** y_pred = ARIMA_forecast + LSTM_correction

**Theory:**
- ARIMA: Captures linear autocorrelation (interpretable)
- LSTM: Learns non-linear residual patterns (flexible)
- Decomposition: Explicit separation of linear and non-linear components

**Strengths:**
- Best of classical + modern approaches
- ARIMA provides stable baseline
- LSTM adds flexibility
- Component breakdown for ablation studies

**Weaknesses:**
- Two-stage training (longer)
- Residuals may have limited signal
- Requires both models to be tuned

**Use Case:** When both linear and non-linear patterns exist

**Output Format:**
```
y_pred = arima_component + lstm_component
```

---

### 6. ARIMA-GARCH Hybrid

**File:** `src/models/arima_garch_hybrid.py`

**Architecture:**
1. **Stage 1:** ARIMA models conditional mean E[y_t | past]
2. **Stage 2:** GARCH models conditional variance Var[y_t | past] on ARIMA residuals
3. **Output:** Point forecast + volatility bands

**Theory (Classical Econometrics):**
- Complete probability model: y_t ~ N(μ_t, σ²_t)
- μ_t from ARIMA (mean equation)
- σ²_t from GARCH (variance equation)
- Separate mean-variance decomposition

**Strengths:**
- Standard econometric approach
- Both moments modeled
- 95% prediction intervals
- Well-studied methodology

**Weaknesses:**
- Still assumes linearity in mean
- No deep learning flexibility
- GARCH limitations remain

**Use Case:** Econometric analysis, interval forecasting

**Output Format:**
```
Point forecast: ARIMA
Volatility: GARCH
Bands: forecast ± 1.96 × volatility
```

---

### 7. Complete ARIMA-GARCH-LSTM Hybrid

**File:** `src/models/arima_garch_lstm_hybrid.py`

**Architecture:**
1. **Stage 1:** ARIMA models conditional mean
2. **Stage 2:** GARCH models conditional variance on ARIMA residuals
3. **Stage 3:** Standardize residuals by GARCH volatility
4. **Stage 4:** LSTM predicts standardized residuals
5. **Final:** y_pred = ARIMA + (LSTM_correction × GARCH_volatility)

**Theory:**
- ARIMA: Linear mean patterns
- GARCH: Time-varying volatility
- LSTM: Non-linear correction after accounting for both

**Strengths:**
- Most comprehensive approach
- All three modeling paradigms
- Volatility-adjusted non-linear correction
- Complete decomposition

**Weaknesses:**
- Most complex pipeline
- Longest training time
- Three models to tune
- Risk of overfitting

**Use Case:** Research, when maximum accuracy is needed

**Output Format:**
```
y_pred = arima_component + lstm_correction
Volatility bands from GARCH
```

---

## 📈 Comparison Framework

**File:** `src/evaluation/compare_models.py`

### Metrics Used

1. **RMSE (Root Mean Squared Error)**
   - Lower is better
   - Penalizes large errors more
   - Same units as target variable

2. **MAE (Mean Absolute Error)**
   - Lower is better
   - Average magnitude of errors
   - More robust to outliers

3. **R² (Coefficient of Determination)**
   - Higher is better (max = 1.0)
   - Proportion of variance explained
   - 1.0 = perfect predictions

4. **Directional Accuracy**
   - Higher is better (max = 100%)
   - % of correct direction predictions
   - Critical for trading strategies

### Evaluation Subsets

- **Train:** In-sample performance (2,774 samples)
- **Validation:** Model selection/tuning (595 samples)
- **Test:** Final evaluation (595 samples)

### Fair Comparison

All models use:
- Same train/val/test splits
- Same preprocessing
- Same evaluation metrics
- Same target variable (log returns)
- Same random seed (42)

---

## 🚀 Running the Complete Pipeline

### Option 1: One-Click Demo (Recommended)

**Windows:**
```bash
run_demo.bat
```

**Linux/Mac:**
```bash
bash run_demo.sh
```

**Direct Python:**
```bash
python run_complete_demo.py
```

### Option 2: Individual Models

```bash
# Classical models
python src/models/arima_model.py
python src/models/garch_model.py

# Deep learning
python src/models/lstm_model.py

# Hybrids
python src/models/hybrid_garch_lstm.py
python src/models/arima_lstm_hybrid.py
python src/models/arima_garch_hybrid.py
python src/models/arima_garch_lstm_hybrid.py

# Comparison
python src/evaluation/compare_models.py
```

---

## 📊 Expected Results

### Training Time (per model)

| Model | Time | Complexity |
|-------|------|------------|
| ARIMA | 1-2 min | Low |
| GARCH | 1-2 min | Low |
| LSTM | 10-15 min | High |
| GARCH-LSTM | 10-15 min | High |
| ARIMA-LSTM | 15-20 min | High |
| ARIMA-GARCH | 2-3 min | Low-Medium |
| Complete Hybrid | 20-25 min | Very High |

**Total First Run:** 60-90 minutes
**Subsequent Runs:** 5-10 minutes (with smart skipping)

### Output Structure

```
results/
└── predictions/
    ├── arima_predictions_TIMESTAMP/
    │   ├── train_predictions.csv
    │   ├── val_predictions.csv
    │   ├── test_predictions.csv
    │   ├── metrics_summary.json
    │   └── model_config.json
    ├── garch_predictions_TIMESTAMP/
    ├── lstm_predictions_TIMESTAMP/
    ├── hybrid_predictions_TIMESTAMP/
    ├── arima_lstm_hybrid_TIMESTAMP/
    ├── arima_garch_hybrid_TIMESTAMP/
    └── arima_garch_lstm_hybrid_TIMESTAMP/

figures/
└── comparisons/
    ├── model_comparison_test.png (4-metric bar charts)
    ├── model_comparison_all_subsets.png (train/val/test comparison)
    └── model_comparison_TIMESTAMP.csv (results table)
```

---

## 🎯 Model Selection Guide

**Choose ARIMA when:**
- Need interpretable coefficients
- Linear patterns dominate
- Fast execution required
- Baseline comparison needed

**Choose GARCH when:**
- Volatility forecasting is primary goal
- Risk management focus
- Need prediction intervals
- Financial time series

**Choose LSTM when:**
- Complex non-linear patterns exist
- Large dataset available
- Interpretability not critical
- Maximum flexibility needed

**Choose GARCH-LSTM when:**
- Both volatility and forecasting important
- Have validated GARCH results
- Need volatility-aware DL

**Choose ARIMA-LSTM when:**
- Clear linear + non-linear decomposition
- Want explicit component analysis
- Research/ablation study

**Choose ARIMA-GARCH when:**
- Need complete probability model
- Econometric rigor required
- Interval forecasting critical

**Choose Complete Hybrid when:**
- Maximum accuracy is goal
- Computational resources available
- Research publication
- Comprehensive analysis needed

---

## 📚 References

### ARIMA
- Box, G. E., & Jenkins, G. M. (1970). Time series analysis: forecasting and control.

### GARCH
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.

### LSTM
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

### Hybrid Models
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. Neurocomputing, 50, 159-175.
- Kristjanpoller, W., & Minutolo, M. C. (2018). A hybrid volatility forecasting framework integrating GARCH, artificial neural network, genetic algorithm and the wavelet transform. Expert Systems with Applications, 109, 1-11.

---

## 🤝 Contributing

To add new hybrid architectures:

1. Create new model file in `src/models/`
2. Follow existing interface: `fit()`, `predict()`, `evaluate()`, `save_results()`
3. Save results in standardized format (CSV + JSON)
4. Update `compare_models.py` to include new model
5. Update `run_complete_demo.py` with training step
6. Document architecture in this file

---

## 📞 Support

For issues or questions:
- Check `QUICK_START.md` for common problems
- Review model-specific files for implementation details
- See `README.md` in project root for general setup

---

**Last Updated:** January 19, 2026  
**Author:** Naveen Babu  
**Project:** FOREX Time Series Forecasting - 7-Model Comparison System

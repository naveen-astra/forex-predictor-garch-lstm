# Phase 2: GARCH Quick Reference

## Running the GARCH Notebook

### Prerequisites
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Key package: arch==6.2.0
```

### Execution Steps

1. **Navigate to project root:**
   ```bash
   cd "d:\Class\Amrita_Class\Sem6\Big Data Analytics - Dr. Sreeja. B.P\project\forex-project"
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook notebooks/03_garch_modeling.ipynb
   ```

3. **Run all cells in order** (Kernel → Restart & Run All)

### Expected Runtime
- **Fast:** ~30 seconds (with pre-loaded data)
- Model fitting: ~5 seconds
- Diagnostics: ~10 seconds
- Plotting: ~10 seconds

---

## Quick Test (Before Notebook)

```bash
python tests/test_garch.py
```

**Expected Output:**
```
ALL TESTS PASSED ✓
```

If tests fail, check:
- `arch` package installed correctly
- Data files exist in `data/processed/`
- No import errors

---

## Key Outputs Generated

### 1. Conditional Volatility Files
- `data/processed/train_data_with_garch.csv`
- `data/processed/val_data_with_garch.csv`
- `data/processed/test_data_with_garch.csv`

**New Column:** `GARCH_Volatility` (conditional standard deviation)

### 2. Fitted Model
- `models/saved_models/garch_11_model.pkl`

**Load later with:**
```python
from src.models.garch_model import GARCHModel
model = GARCHModel.load_model('models/saved_models/garch_11_model.pkl')
```

### 3. Figures (300 DPI PNG)
- `results/figures/garch_volatility_training.png`
- `results/figures/garch_clustering.png`
- `results/figures/garch_diagnostics.png`

---

## Interpreting Results

### Parameter Estimates

**Look for:**
- **ω (omega):** Should be small and positive
- **α (alpha[1]):** Typically 0.05-0.15 for FOREX
- **β (beta[1]):** Typically 0.80-0.90 for FOREX
- **α + β:** Should be < 1 (stationarity), often 0.95-0.99

**Example (typical EUR/USD):**
```
ω = 0.000005
α = 0.085
β = 0.900
α + β = 0.985  ← High persistence (common for FOREX)
```

### Diagnostic Tests

| Test | Null Hypothesis | Good Result |
|------|----------------|-------------|
| **Ljung-Box** | No autocorrelation in std. residuals | p > 0.05 |
| **ARCH LM** | No remaining ARCH effects | p > 0.05 |
| **Jarque-Bera** | Residuals are normal | p > 0.05 (often fails) |

**Acceptable Result:**
- ✓ Pass Ljung-Box
- ✓ Pass ARCH LM
- ⚠ Fail Jarque-Bera (expected for financial data)

### Model Comparison

**AIC/BIC Interpretation:**
- Lower values = Better fit
- ΔAIC < 2: Models essentially equivalent
- ΔAIC > 10: Strong evidence against higher-AIC model

**Typical Outcome:**
- GARCH(1,1) Normal ← Usually wins (parsimony)
- GARCH(2,1) Normal: Similar AIC, more complex
- GARCH(1,1) t-dist: Better for fat tails, but minimal improvement

---

## Common Issues & Solutions

### Issue 1: ConvergenceWarning
**Symptom:** Model fit shows convergence warning

**Solution:**
- Returns may need rescaling (already done by default)
- Try different starting values
- Check for data quality issues

### Issue 2: Ljung-Box Test Fails
**Symptom:** p-value < 0.05 for Ljung-Box

**Solution:**
- Try GARCH(2,1) or ARMA(1,1)-GARCH(1,1)
- Check if returns are properly demeaned
- Verify stationarity of returns

### Issue 3: ARCH LM Test Fails
**Symptom:** p-value < 0.05 for ARCH LM

**Solution:**
- Increase GARCH order (try GARCH(2,2))
- Consider EGARCH or GJR-GARCH for asymmetry
- Review preprocessing (outliers may cause this)

### Issue 4: Parameters Don't Make Sense
**Symptom:** α + β > 1 or negative coefficients

**Possible Causes:**
- Returns not stationary (run ADF test)
- Too many outliers in data
- Wrong data transformation

**Solution:**
- Re-check preprocessing in Phase 1
- Apply more aggressive outlier treatment
- Verify log returns calculation

---

## Using GARCH Outputs in Hybrid Model

### Loading Volatility Data

```python
import pandas as pd

# Load data with GARCH volatility
train = pd.read_csv('data/processed/train_data_with_garch.csv', 
                    index_col=0, parse_dates=True)

# Access GARCH volatility
garch_vol = train['GARCH_Volatility'].values
```

### Creating Hybrid Features

```python
# For LSTM input
import numpy as np

# Combine features
X_hybrid = np.column_stack([
    lagged_returns,           # Shape: (n_samples, n_lags)
    garch_vol.reshape(-1, 1), # Shape: (n_samples, 1)
    technical_indicators       # Shape: (n_samples, n_features)
])

# X_hybrid shape: (n_samples, n_lags + 1 + n_features)
```

---

## Next Steps (Phase 3)

1. **Review GARCH Results:**
   - Check parameter estimates are reasonable
   - Verify diagnostics pass
   - Examine volatility plots

2. **Proceed to LSTM Baseline:**
   - Implement LSTM without GARCH features
   - Compare LSTM vs GARCH performance
   - Establish baseline metrics

3. **Prepare for Hybrid Model:**
   - GARCH volatility ready as input feature
   - Can compare: LSTM-only vs GARCH+LSTM

---

## For Journal Paper

### Methodology Section:
```
"We model volatility dynamics using a GARCH(1,1) specification
(Bollerslev, 1986), estimated via Maximum Likelihood. Model adequacy
is verified through Ljung-Box and ARCH LM diagnostic tests."
```

### Results Section:
**Table 1: GARCH Parameter Estimates**
| Parameter | Estimate | Std. Error | t-statistic |
|-----------|----------|------------|-------------|
| ω | 0.000XXX | 0.000XXX | X.XX |
| α | 0.0XXX | 0.0XXX | X.XX |
| β | 0.XXX | 0.0XX | X.XX |

**Table 2: Diagnostic Tests**
| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Ljung-Box(10) | XX.XX | 0.XXX | Pass |
| ARCH LM(10) | XX.XX | 0.XXX | Pass |

### Figure Captions:
- **Figure 1:** EUR/USD log returns (upper panel) and GARCH(1,1) conditional volatility estimates (lower panel) for the training period. Volatility clustering is evident.

- **Figure 2:** Scatter plot of absolute returns versus GARCH conditional volatility, demonstrating the model's ability to capture volatility magnitude.

- **Figure 3:** Diagnostic plots for GARCH(1,1) model. (A) Standardized residuals, (B) histogram with normal overlay, (C) Q-Q plot, (D) ACF of squared standardized residuals.

---

**Ready for Phase 3: LSTM Baseline Implementation**

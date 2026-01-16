# Literature Review

## Hybrid Time Series Forecasting Models

### 1. GARCH Models in Financial Forecasting

**Key Papers:**
- Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity"
- Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation"

**Applications:**
- Volatility modeling in financial markets
- Risk management and VaR estimation
- Option pricing

### 2. LSTM for Time Series

**Key Papers:**
- Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory"
- Fischer, T., & Krauss, C. (2018). "Deep learning with long short-term memory networks for financial market predictions"

**Advantages:**
- Captures long-term dependencies
- Handles non-linear patterns
- Works with multiple input features

### 3. Hybrid GARCH-LSTM Models

**Reference Paper:**
- "A hybrid model for forecasting volatility" - [Add citation]
- GitHub Implementation: https://github.com/tlemenestrel/LSTM_GARCH

**Innovation:**
- Combines statistical rigor of GARCH with pattern recognition of LSTM
- GARCH captures volatility clustering
- LSTM learns complex market dynamics

### 4. FOREX Forecasting

**Challenges:**
- High noise-to-signal ratio
- Non-stationarity in price levels
- Multiple market regimes
- 24/5 trading (no fixed closing time)

**Common Approaches:**
- Economic indicators (interest rates, GDP)
- Technical analysis
- Machine learning models
- Hybrid statistical-ML approaches

## Research Gap

While GARCH and LSTM have been individually successful:
- Limited work on GARCH-LSTM hybrids for FOREX
- Few implementations with full reproducibility
- Need for big data scalability

## Our Contribution

1. Rigorous implementation with reproducibility
2. Comprehensive feature engineering for FOREX
3. Statistical validation and comparison
4. Scalable architecture for big data

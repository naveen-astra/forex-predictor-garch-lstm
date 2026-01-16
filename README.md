# Intelligent FOREX Exchange Rate Forecasting using Hybrid GARCH-LSTM and Big Data Analytics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements a hybrid GARCH-LSTM architecture for forecasting FOREX exchange rate volatility, specifically focusing on the EUR/USD currency pair. The research combines traditional econometric models (GARCH) with deep learning (LSTM) to leverage the strengths of both approaches for superior forecasting performance.

**Research Focus:**
- **GARCH Component**: Captures volatility clustering and mean reversion in financial returns
- **LSTM Component**: Learns complex non-linear temporal patterns and long-term dependencies
- **Hybrid Integration**: GARCH predictions serve as additional features for LSTM, combining statistical rigor with pattern recognition

**Target Audience:** Academic researchers, quantitative finance professionals, data scientists working with time series forecasting.

---

## Research Objectives

1. Implement and validate a hybrid GARCH-LSTM architecture for FOREX forecasting
2. Systematically compare performance against baseline models (Naive, ARIMA, GARCH-only, LSTM-only)
3. Ensure full reproducibility for academic publication
4. Develop scalable architecture suitable for big data processing
5. Generate publication-ready results and visualizations

---

## Project Structure

```
forex-project/
│
├── data/
│   ├── raw/                      # Original downloaded FOREX data
│   ├── processed/                # Cleaned and feature-engineered data
│   └── external/                 # Economic indicators, sentiment data
│
├── src/
│   ├── data/
│   │   ├── fetch_data.py        # Data acquisition from Yahoo Finance/Alpha Vantage
│   │   └── preprocess.py        # Data cleaning and feature engineering
│   ├── models/
│   │   ├── garch_model.py       # GARCH implementation
│   │   ├── lstm_model.py        # LSTM architecture
│   │   └── hybrid_model.py      # Hybrid GARCH-LSTM model
│   ├── evaluation/
│   │   ├── metrics.py           # Performance metrics (RMSE, MAE, R²)
│   │   └── visualization.py     # Plotting utilities
│   └── utils/
│       ├── config.py            # Configuration and hyperparameters
│       └── helpers.py           # Utility functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Initial EDA and stationarity tests
│   ├── 02_feature_engineering.ipynb  # Feature creation and analysis
│   ├── 03_garch_modeling.ipynb       # GARCH model development
│   ├── 04_lstm_modeling.ipynb        # LSTM model development
│   ├── 05_hybrid_model.ipynb         # Hybrid model integration
│   └── 06_evaluation_comparison.ipynb # Benchmarking and results
│
├── models/
│   ├── saved_models/            # Trained model weights (.h5, .pkl)
│   └── checkpoints/             # Training checkpoints
│
├── results/
│   ├── figures/                 # Publication-ready plots
│   ├── tables/                  # Performance metrics tables
│   └── predictions/             # Model forecast outputs
│
├── docs/
│   ├── literature_review.md     # Survey of related work
│   ├── methodology.md           # Detailed methodology
│   └── results_interpretation.md # Analysis and discussion
│
├── tests/                       # Unit tests
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore
└── LICENSE

```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip or conda for package management
- (Optional) GPU for faster LSTM training

### Installation

1. **Clone the repository:**
   ```bash
   cd "d:\Class\Amrita_Class\Sem6\Big Data Analytics - Dr. Sreeja. B.P\project\forex-project"
   ```

2. **Create virtual environment:**
   ```bash
   # Using conda (recommended)
   conda create -n forex-lstm python=3.10
   conda activate forex-lstm

   # OR using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) For GPU support:**
   ```bash
   pip install tensorflow-gpu==2.13.0
   ```

---

## Data Acquisition

### Step 1: Fetch FOREX Data

```bash
python src/data/fetch_data.py
```

**What it does:**
- Downloads EUR/USD historical data from Yahoo Finance
- Fallback to Alpha Vantage if primary source fails
- Date range: 2010-01-01 to 2025-12-31 (configurable in `config.py`)
- Validates data quality and saves in multiple formats (CSV, Parquet)

**Data Sources:**
- **Primary:** Yahoo Finance (`yfinance` package) - Free, no API key required
- **Fallback:** Alpha Vantage - Requires free API key from [alphavantage.co](https://www.alphavantage.co/support/#api-key)

**Output:**
- `data/raw/EUR_USD_raw_YYYYMMDD.csv`
- `data/raw/EUR_USD_raw_YYYYMMDD.parquet`
- `data/raw/EUR_USD_raw_YYYYMMDD_metadata.txt`

### Step 2: Preprocess Data

```bash
python src/data/preprocess.py
```

**What it does:**
- Handles missing values (forward fill strategy)
- Detects outliers using IQR method
- Computes log returns: `ln(P_t / P_{t-1})`
- Engineers features:
  - Log trading range
  - Rolling volatility (10, 30, 60 days)
  - Technical indicators (RSI, SMA, EMA, MACD)
- Splits data chronologically:
  - Training: 70% (earliest data)
  - Validation: 15%
  - Test: 15% (most recent data)

**Output:**
- `data/processed/train_data.csv`
- `data/processed/val_data.csv`
- `data/processed/test_data.csv`

---

## Exploratory Data Analysis

### Run EDA Notebook

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**What it covers:**
1. **Data Quality Checks**: Missing values, duplicates, data types
2. **Descriptive Statistics**: Mean, std, min, max, skewness, kurtosis
3. **Stationarity Tests**: 
   - Augmented Dickey-Fuller (ADF) test
   - KPSS test
4. **Distribution Analysis**: Normality tests, fat tails examination
5. **Volatility Patterns**: Rolling volatility characteristics

**Key Findings (Expected):**
- Price levels are non-stationary
- Log returns are stationary (suitable for GARCH)
- Returns exhibit fat tails and volatility clustering
- Data is clean with comprehensive features

---

## Configuration

All hyperparameters and settings are centralized in `src/utils/config.py`:

### Random Seeds (Reproducibility)
```python
RANDOM_SEED = 42  # Master seed for all operations
```

### Data Settings
```python
DATA_CONFIG = {
    'currency_pair': 'EURUSD=X',
    'start_date': '2010-01-01',
    'end_date': '2025-12-31',
    'frequency': '1d',
}
```

### GARCH Configuration
```python
GARCH_CONFIG = {
    'p': 1,  # GARCH lag order
    'q': 1,  # ARCH lag order
    'target_variable': 'Log_Returns',
}
```

### LSTM Configuration
```python
LSTM_CONFIG = {
    'lstm_units': [200, 200],
    'dropout_rate': 0.2,
    'n_timesteps': 4,
    'batch_size': 700,
    'epochs': 60,
    'learning_rate': 0.01,
}
```

**To modify configuration:**
Edit `src/utils/config.py` directly. All scripts and notebooks read from this central file.

---

## Workflow Overview

### Phase 1: Data Preparation
- Project structure setup
- Configuration file with random seeds
- Data fetching module
- Preprocessing pipeline
- Initial EDA notebook

### Phase 2: GARCH Modeling
- Implement GARCH(1,1) model using PyFlux/arch
- Generate rolling predictions
- Generate forward-looking predictions
- Evaluate GARCH-only performance

### Phase 3: LSTM Modeling
- Build baseline LSTM (without GARCH features)
- Implement time-lagging mechanism
- Train and validate LSTM
- Hyperparameter tuning

### Phase 4: Hybrid Model
- Integrate GARCH predictions as LSTM features
- Train hybrid GARCH-LSTM model
- Compare all model variants

### Phase 5: Evaluation & Publication
- Comprehensive benchmarking
- Statistical significance tests
- Generate publication-ready figures
- Write methodology and results sections

---

## Reproducibility Guidelines

### For Academic Publication

1. **Random Seeds**: All set via `config.py` (NumPy, TensorFlow, Python random)
2. **Dependencies**: Pinned versions in `requirements.txt`
3. **Data Provenance**: Source, date range, and preprocessing documented
4. **Model Architecture**: Fully specified in configuration files
5. **Training Logs**: Saved automatically with timestamps
6. **Results**: Stored in `results/` with metadata

### Running Reproducible Experiments

```python
from src.utils.config import set_random_seeds, RANDOM_SEED

# At the start of every script/notebook
set_random_seeds(RANDOM_SEED)
```

This ensures:
- Identical train/test splits across runs
- Reproducible model initialization
- Consistent stochastic operations

---

## Key Concepts

### Why GARCH?
**GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** models time-varying volatility:
- Captures **volatility clustering** (high volatility tends to follow high volatility)
- Models **mean reversion** in volatility
- Standard in financial econometrics

**Mathematical Form (GARCH(1,1)):**
```
σ²_t = α₀ + α₁ · ε²_{t-1} + β₁ · σ²_{t-1}
```
Where:
- σ²_t: Conditional variance at time t
- ε_t: Error term (return at time t)
- α₀, α₁, β₁: Model parameters

### Why LSTM?
**LSTM (Long Short-Term Memory)** neural networks:
- Learn **long-term dependencies** in sequences
- Capture **non-linear patterns** GARCH cannot model
- Handle multiple input features simultaneously

**Key Components:**
- **Forget Gate**: Decides what information to discard
- **Input Gate**: Determines what new information to store
- **Output Gate**: Controls what to output from memory

### Hybrid Advantage
| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **GARCH** | Volatility clustering, statistical rigor | Linear, single variable |
| **LSTM** | Non-linear, multi-feature | Black-box, requires more data |
| **Hybrid** | Best of both worlds | More complex to implement |

---

## Dependencies

### Core Libraries
- **pandas** (2.0.3): Data manipulation
- **numpy** (1.24.3): Numerical computing
- **tensorflow** (2.13.0): Deep learning framework
- **statsmodels** (0.14.0): Statistical models and tests
- **yfinance** (0.2.28): Financial data download

### Complete List
See `requirements.txt` for all dependencies with pinned versions.

---

## Troubleshooting

### Common Issues

**1. PyFlux Installation Fails**
```bash
# Use arch package instead
pip install arch==6.2.0
```

**2. Yahoo Finance Returns Empty Data**
- Check internet connection
- Verify ticker format: `EURUSD=X` (with `=X` suffix)
- Try Alpha Vantage as fallback

**3. TensorFlow GPU Not Detected**
```bash
# Check CUDA compatibility
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**4. Import Errors**
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/forex-project"
```

---

## Contact & Support


For questions or issues:
1. Check the `docs/` folder for detailed documentation
2. Review error logs in `experiment.log`
3. Open an issue in the project repository (if applicable)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **Reference Paper**: [Hybrid LSTM-GARCH Model](https://www.sciencedirect.com/science/article/pii/S0957417418301416)
- **Data Sources**: Yahoo Finance, Alpha Vantage
- **Libraries**: TensorFlow, PyFlux, Statsmodels communities

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{forex_garch_lstm_2026,
  title={Intelligent FOREX Exchange Rate Forecasting using Hybrid GARCH-LSTM},
  author={Research Team},
  year={2026},
  institution={Amrita Vishwa Vidyapeetham},
  note={Academic Project - Big Data Analytics}
}
```

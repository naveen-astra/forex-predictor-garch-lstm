# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-16

### Added - Phase 1: Project Setup & Data Preparation

#### Project Structure
- Created comprehensive folder structure for research project
- Added `.gitignore` for Python projects
- Added `.gitkeep` files to maintain empty directories
- Initialized Python package structure with `__init__.py` files

#### Documentation
- Added comprehensive `README.md` with project overview
- Added `QUICKSTART.md` for immediate setup guide
- Added `LICENSE` (MIT License)
- Added `CONTRIBUTING.md` for contributors
- Added `docs/methodology.md` for research methodology
- Added `docs/literature_review.md` for related work

#### Configuration & Dependencies
- Created central configuration system in `src/utils/config.py`
- Set random seeds for reproducibility (seed=42)
- Added `requirements.txt` with pinned dependency versions
- Configured GARCH, LSTM, and Hybrid model parameters

#### Data Acquisition
- Implemented `src/data/fetch_data.py` with multi-source support
  - Primary: Yahoo Finance (yfinance)
  - Fallback: Alpha Vantage
- Added data validation and quality checks
- Implemented saving in multiple formats (CSV, Parquet)
- Added metadata tracking for data provenance

#### Data Preprocessing
- Implemented `src/data/preprocess.py` with full pipeline
  - Missing value handling (forward fill strategy)
  - Outlier detection (IQR method)
  - Log returns computation
  - Log trading range calculation
  - Rolling volatility features (10, 30, 60 days)
  - Technical indicators (RSI, SMA, EMA, MACD)
  - Chronological train/validation/test splitting (70/15/15)

#### Exploratory Data Analysis
- Created `notebooks/01_data_exploration.ipynb`
  - Data quality checks
  - Descriptive statistics
  - Stationarity tests (ADF, KPSS)
  - Distribution analysis (Jarque-Bera)
  - Volatility pattern examination

#### Utilities & Testing
- Added `src/utils/helpers.py` with utility functions
- Added `tests/test_preprocessing.py` with unit tests
- Set up testing framework with pytest

### Configuration Highlights
- EUR/USD currency pair
- Date range: 2010-01-01 to 2025-12-31
- Daily frequency
- GARCH(1,1) configuration
- 2-layer LSTM (200 units each)
- Reproducibility: All random seeds set

### Next Phase (Planned)
- [ ] GARCH model implementation
- [ ] LSTM baseline model
- [ ] Hybrid GARCH-LSTM integration
- [ ] Model evaluation and comparison
- [ ] Big data scaling with Spark

---

## Version History

- **v1.0.0** - Initial release with complete data preparation pipeline
- **v0.1.0** - Project structure and documentation setup

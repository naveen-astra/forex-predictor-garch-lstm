# 🚀 Quick Start Guide - One-Click Demo

## For Windows Users

### Option 1: Double-Click (Easiest)
1. **Double-click** `run_demo.bat`
2. Wait 10-30 minutes
3. Dashboard opens automatically ✨

### Option 2: Command Line
```cmd
python run_complete_demo.py
```

---

## For Linux/Mac Users

### Option 1: Run Script
```bash
bash run_demo.sh
```

### Option 2: Direct Python
```bash
python run_complete_demo.py
```

---

## What Happens?

The script automatically runs the **entire pipeline**:

✅ **Step 1:** Download EUR/USD data (2010-2025)  
✅ **Step 2:** Preprocess data (features, splits)  
✅ **Step 3:** Train GARCH volatility model  
✅ **Step 4:** Train ARIMA baseline  
✅ **Step 5:** Train LSTM baseline  
✅ **Step 6:** Train Hybrid GARCH-LSTM  
✅ **Step 7:** Compare all models  
✅ **Step 8:** Generate visualizations  
✅ **Step 9:** Open interactive dashboard  

---

## ⏱️ Time Required

- **First run:** 20-30 minutes (downloads data + trains all models)
- **Subsequent runs:** 5-10 minutes (skips existing data/models)

---

## 📊 After Completion

### 1. Dashboard Opens Automatically
- **URL:** `file:///.../dashboard/index.html`
- Shows all results, metrics, and comparisons

### 2. Check These Folders
```
results/
├── predictions/        # All model predictions
├── figures/           # Publication-quality plots
└── tables/            # Performance metrics

models/saved_models/   # Trained models (.h5, .pkl)
```

### 3. View Detailed Analysis
```
notebooks/
├── 01_data_exploration.ipynb
├── 02_arima_baseline.ipynb
├── 03_garch_modeling.ipynb
├── 04_lstm_baseline.ipynb
├── 05_hybrid_garch_lstm.ipynb
└── 06_final_evaluation.ipynb
```

---

## ⚙️ Prerequisites

**Before running, install dependencies:**
```bash
pip install -r requirements.txt
```

**Required Python:** 3.10 or 3.11  
**Required Packages:** numpy, pandas, tensorflow, arch, statsmodels, matplotlib

---

## 🛠️ Troubleshooting

### Issue: "Module not found"
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Issue: "Data download failed"
**Solution:** Check internet connection or manually download data
```bash
python src/data/fetch_data.py
```

### Issue: "Out of memory"
**Solution:** Reduce batch size in `src/utils/config.py`
```python
LSTM_CONFIG = {
    'batch_size': 350,  # Reduce from 700
}
```

### Issue: Script hangs/freezes
**Solution:** Kill and restart. Existing data/models will be reused
```bash
Ctrl+C  # Stop
python run_complete_demo.py  # Restart
```

---

## 🎯 Expected Results

After successful completion:

| Model | Test RMSE | Directional Accuracy |
|-------|-----------|---------------------|
| ARIMA | 0.003873 | 50.42% |
| GARCH | 0.003162 | 52.10% |
| LSTM  | 0.001414 | 54.51% |
| **Hybrid** | **0.001225** | **86.20%** ⭐ |

**Key Finding:** Hybrid GARCH-LSTM achieves **86.20% directional accuracy** - a **55.6% improvement** over LSTM baseline!

---

## 📝 For Manual Step-by-Step

If you prefer to run steps individually:

```bash
# Step 1: Data
python src/data/fetch_data.py
python src/data/preprocess.py

# Step 2: Models
python src/models/arima_model.py
python src/models/garch_model.py
python src/models/lstm_model.py
python src/models/hybrid_garch_lstm.py

# Step 3: Evaluation
python src/evaluation/compare_models.py

# Step 4: View Results
# Open dashboard/index.html in browser
```

---

## 🎓 Academic Use

For detailed methodology and reproducibility:
1. Read `docs/paper_draft_sections.md` (4,500+ words)
2. Review notebooks for step-by-step explanations
3. Check `docs/reproducibility_statement.md` for seeds and config

---

## 📧 Questions?

Check the main `README.md` for full documentation.

---

**Enjoy your one-click FOREX prediction demo! 🚀📈**

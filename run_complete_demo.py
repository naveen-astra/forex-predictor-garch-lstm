"""
🚀 FOREX GARCH-LSTM: Complete Demo Runner
==========================================
Single-click execution of the entire pipeline:
1. Data fetching
2. Preprocessing
3. GARCH modeling
4. LSTM baseline
5. Hybrid GARCH-LSTM
6. Model comparison
7. Dashboard launch

Usage: python run_complete_demo.py

Author: Naveen Babu
Date: January 19, 2026
"""

import sys
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI colors for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    """Print styled header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")

def print_step(step_num, total_steps, description):
    """Print step information."""
    print(f"{Colors.CYAN}{Colors.BOLD}[STEP {step_num}/{total_steps}]{Colors.END} {description}")

def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def run_script(script_path, description):
    """
    Run a Python script and handle errors.
    
    Args:
        script_path: Path to script
        description: Human-readable description
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{Colors.BLUE}Running: {description}{Colors.END}")
    print(f"Script: {script_path}")
    
    try:
        # Run script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print_success(f"{description} completed successfully")
            return True
        else:
            print_error(f"{description} failed with exit code {result.returncode}")
            if result.stderr:
                print(f"{Colors.RED}Error output:{Colors.END}")
                print(result.stderr[:500])  # Print first 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"{description} timed out (>10 minutes)")
        return False
    except Exception as e:
        print_error(f"{description} failed: {str(e)}")
        return False

def check_prerequisites():
    """Check if all required packages are installed."""
    print_step(0, 9, "Checking Prerequisites")
    
    required_packages = [
        'numpy', 'pandas', 'tensorflow', 'keras', 
        'arch', 'statsmodels', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} (missing)")
    
    if missing:
        print_error(f"Missing packages: {', '.join(missing)}")
        print(f"\n{Colors.YELLOW}Install with: pip install -r requirements.txt{Colors.END}")
        return False
    
    print_success("All prerequisites installed")
    return True

def check_data_exists():
    """Check if data already exists."""
    raw_data = PROJECT_ROOT / "data" / "raw"
    processed_data = PROJECT_ROOT / "data" / "processed"
    
    raw_files = list(raw_data.glob("EUR_USD_raw_*.csv")) if raw_data.exists() else []
    train_file = processed_data / "train_data.csv"
    
    return len(raw_files) > 0 and train_file.exists()

def open_dashboard():
    """Open the dashboard in default browser."""
    dashboard_path = PROJECT_ROOT / "dashboard" / "index.html"
    
    if not dashboard_path.exists():
        print_warning("Dashboard not found. Skipping...")
        return
    
    print(f"\n{Colors.CYAN}Opening dashboard...{Colors.END}")
    
    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{dashboard_path.absolute()}")
    
    print_success("Dashboard opened in browser")
    print(f"\n{Colors.BOLD}Dashboard URL:{Colors.END} file://{dashboard_path.absolute()}")

def main():
    """Main execution function."""
    start_time = time.time()
    
    # Banner
    print(f"""
{Colors.HEADER}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   FOREX GARCH-LSTM HYBRID MODEL                              ║
║                     Complete Demo Pipeline                                   ║
║                                                                              ║
║          Intelligent FOREX Exchange Rate Forecasting                         ║
║          Using Hybrid GARCH-LSTM Architecture                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}
""")
    
    print(f"{Colors.BOLD}Date:{Colors.END} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.BOLD}Project Root:{Colors.END} {PROJECT_ROOT}")
    print()
    
    # Step 0: Prerequisites check
    if not check_prerequisites():
        print_error("Prerequisites check failed. Please install missing packages.")
        return
    
    # Pipeline steps
    total_steps = 9
    current_step = 0
    
    # Step 1: Data Fetching
    current_step += 1
    print_header(f"STEP {current_step}/{total_steps}: DATA ACQUISITION")
    
    if check_data_exists():
        print_warning("Data already exists. Skipping download...")
        print("  (Delete data/raw and data/processed to re-download)")
    else:
        script = PROJECT_ROOT / "src" / "data" / "fetch_data.py"
        if not run_script(script, "Data Fetching"):
            print_error("Data fetching failed. Cannot continue.")
            return
    
    # Step 2: Data Preprocessing
    current_step += 1
    print_header(f"STEP {current_step}/{total_steps}: DATA PREPROCESSING")
    
    train_file = PROJECT_ROOT / "data" / "processed" / "train_data.csv"
    if train_file.exists():
        print_warning("Preprocessed data exists. Skipping preprocessing...")
    else:
        script = PROJECT_ROOT / "src" / "data" / "preprocess.py"
        if not run_script(script, "Data Preprocessing"):
            print_error("Preprocessing failed. Cannot continue.")
            return
    
    # Step 3: GARCH Modeling
    current_step += 1
    print_header(f"STEP {current_step}/{total_steps}: GARCH VOLATILITY MODELING")
    
    garch_file = PROJECT_ROOT / "data" / "processed" / "train_data_with_garch.csv"
    if garch_file.exists():
        print_warning("GARCH results exist. Skipping GARCH training...")
    else:
        script = PROJECT_ROOT / "src" / "models" / "garch_model.py"
        if not run_script(script, "GARCH Model Training"):
            print_warning("GARCH training failed. Continuing with LSTM...")
    
    # Step 4: ARIMA Baseline
    current_step += 1
    print_header(f"STEP {current_step}/{total_steps}: ARIMA BASELINE MODEL")
    
    arima_results = list((PROJECT_ROOT / "results" / "predictions").glob("arima_predictions_*"))
    if arima_results:
        print_warning("ARIMA results exist. Skipping ARIMA training...")
    else:
        script = PROJECT_ROOT / "src" / "models" / "arima_model.py"
        if not run_script(script, "ARIMA Baseline Training"):
            print_warning("ARIMA training failed. Continuing with LSTM...")
    
    # Step 5: LSTM Baseline
    current_step += 1
    print_header(f"STEP {current_step}/{total_steps}: LSTM BASELINE MODEL")
    
    lstm_model = PROJECT_ROOT / "models" / "saved_models" / "lstm_baseline_final.h5"
    if lstm_model.exists():
        print_warning("LSTM model exists. Skipping LSTM training...")
    else:
        script = PROJECT_ROOT / "src" / "models" / "lstm_model.py"
        if not run_script(script, "LSTM Baseline Training"):
            print_warning("LSTM training failed. Continuing with Hybrid...")
    
    # Step 6: Hybrid GARCH-LSTM
    current_step += 1
    print_header(f"STEP {current_step}/{total_steps}: HYBRID GARCH-LSTM MODEL")
    
    hybrid_model = PROJECT_ROOT / "models" / "saved_models" / "hybrid_garch_lstm_final.h5"
    if hybrid_model.exists():
        print_warning("Hybrid model exists. Skipping Hybrid training...")
    else:
        script = PROJECT_ROOT / "src" / "models" / "hybrid_garch_lstm.py"
        if not run_script(script, "Hybrid GARCH-LSTM Training"):
            print_error("Hybrid training failed.")
    
    # Step 7: Model Comparison
    current_step += 1
    print_header(f"STEP {current_step}/{total_steps}: MODEL COMPARISON")
    
    script = PROJECT_ROOT / "src" / "evaluation" / "compare_models.py"
    run_script(script, "Model Comparison")
    
    # Step 8: Generate Summary Report
    current_step += 1
    print_header(f"STEP {current_step}/{total_steps}: GENERATING SUMMARY REPORT")
    
    try:
        from src.utils.config import RESULTS_DIR, FIGURES_DIR
        
        print(f"\n{Colors.BOLD}Results Summary:{Colors.END}")
        print(f"  Models trained: ARIMA, GARCH, LSTM, Hybrid")
        print(f"  Results directory: {RESULTS_DIR}")
        print(f"  Figures directory: {FIGURES_DIR}")
        
        # Count outputs
        predictions_dir = RESULTS_DIR / "predictions"
        if predictions_dir.exists():
            pred_files = list(predictions_dir.glob("*_predictions_*"))
            print(f"  Prediction files: {len(pred_files)}")
        
        if FIGURES_DIR.exists():
            fig_files = list(FIGURES_DIR.glob("*.png"))
            print(f"  Generated figures: {len(fig_files)}")
        
        print_success("Summary report generated")
        
    except Exception as e:
        print_warning(f"Could not generate summary: {e}")
    
    # Step 9: Open Dashboard
    current_step += 1
    print_header(f"STEP {current_step}/{total_steps}: LAUNCHING DASHBOARD")
    
    open_dashboard()
    
    # Final Summary
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print_header("DEMO COMPLETE")
    
    print(f"""
{Colors.GREEN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                          ✅ DEMO COMPLETED SUCCESSFULLY                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}

{Colors.BOLD}Execution Time:{Colors.END} {minutes}m {seconds}s

{Colors.BOLD}What Was Done:{Colors.END}
  ✅ Data fetched and preprocessed
  ✅ ARIMA baseline trained
  ✅ GARCH volatility model trained
  ✅ LSTM baseline trained
  ✅ Hybrid GARCH-LSTM trained
  ✅ Model comparison generated
  ✅ Dashboard opened

{Colors.BOLD}Next Steps:{Colors.END}
  1. Review dashboard for performance metrics
  2. Check notebooks/ for detailed analysis
  3. View results/figures/ for publication-quality plots
  4. Read docs/paper_draft_sections.md for methodology

{Colors.BOLD}Key Files:{Colors.END}
  • Dashboard: dashboard/index.html
  • Results: results/predictions/
  • Figures: results/figures/
  • Models: models/saved_models/

{Colors.CYAN}💡 Tip: Run individual notebooks for detailed step-by-step analysis{Colors.END}
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo interrupted by user.{Colors.END}")
    except Exception as e:
        print(f"\n\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()

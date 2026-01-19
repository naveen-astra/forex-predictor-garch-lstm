@echo off
echo ============================================
echo   COMPREHENSIVE TIME SERIES ANALYSIS
echo ============================================
echo.
echo This script will run:
echo 1. ACF/PACF Analysis
echo 2. GARCH Order Comparison
echo 3. Open xAI Dashboard
echo.
pause
echo.

echo [1/3] Running ACF/PACF Analysis...
echo ----------------------------------------
python src/analysis/acf_pacf_analysis.py
if %errorlevel% neq 0 (
    echo ERROR: ACF/PACF Analysis failed!
    pause
    exit /b 1
)
echo.

echo [2/3] Running GARCH Order Comparison...
echo ----------------------------------------
python src/analysis/garch_order_comparison.py
if %errorlevel% neq 0 (
    echo ERROR: GARCH Order Comparison failed!
    pause
    exit /b 1
)
echo.

echo [3/3] Opening xAI Dashboard...
echo ----------------------------------------
start "" "dashboard\index_xai.html"
echo.

echo ============================================
echo   ANALYSIS COMPLETE!
echo ============================================
echo.
echo Results saved to:
echo - results/figures/analysis/acf_pacf_analysis.png
echo - results/figures/analysis/acf_pacf_statistics.csv
echo - results/figures/analysis/garch_order_comparison.png
echo - results/figures/analysis/garch_order_results.csv
echo - results/figures/analysis/garch_order_summary.json
echo.
echo Dashboard opened in your browser.
echo.
pause

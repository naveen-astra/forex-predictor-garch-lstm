@echo off
REM ============================================================================
REM FOREX GARCH-LSTM: One-Click Demo Launcher (Windows)
REM ============================================================================
REM Usage: Double-click this file or run: run_demo.bat
REM ============================================================================

echo.
echo ================================================================================
echo                    FOREX GARCH-LSTM COMPLETE DEMO
echo ================================================================================
echo.
echo Starting complete pipeline...
echo This will take 10-30 minutes depending on your hardware.
echo.
echo You can safely minimize this window and do other work.
echo ================================================================================
echo.

REM Run the Python script
python run_complete_demo.py

REM Check if successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo                         DEMO COMPLETED SUCCESSFULLY
    echo ================================================================================
    echo.
    echo Dashboard should now be open in your browser.
    echo.
) else (
    echo.
    echo ================================================================================
    echo                            DEMO ENCOUNTERED ERRORS
    echo ================================================================================
    echo.
    echo Please check the output above for details.
    echo.
)

echo Press any key to exit...
pause >nul

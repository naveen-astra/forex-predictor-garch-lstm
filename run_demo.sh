#!/bin/bash

################################################################################
# FOREX GARCH-LSTM: One-Click Demo Launcher (Linux/Mac)
################################################################################
# Usage: bash run_demo.sh
#        or: chmod +x run_demo.sh && ./run_demo.sh
################################################################################

echo ""
echo "================================================================================"
echo "                    FOREX GARCH-LSTM COMPLETE DEMO"
echo "================================================================================"
echo ""
echo "Starting complete pipeline..."
echo "This will take 10-30 minutes depending on your hardware."
echo ""
echo "You can safely minimize this terminal and do other work."
echo "================================================================================"
echo ""

# Run the Python script
python run_complete_demo.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "                         DEMO COMPLETED SUCCESSFULLY"
    echo "================================================================================"
    echo ""
    echo "Dashboard should now be open in your browser."
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "                            DEMO ENCOUNTERED ERRORS"
    echo "================================================================================"
    echo ""
    echo "Please check the output above for details."
    echo ""
fi

echo "Press Enter to exit..."
read

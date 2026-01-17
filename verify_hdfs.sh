#!/bin/bash

###############################################################################
# HDFS Verification Script for FOREX Big Data Pipeline
#
# This script provides commands to verify HDFS integration, check outputs,
# and troubleshoot issues.
#
# Author: Naveen Babu
# Date: January 2026
# Usage: bash verify_hdfs.sh
###############################################################################

echo "=========================================================================="
echo "HDFS VERIFICATION FOR FOREX BIG DATA PIPELINE"
echo "=========================================================================="
echo ""

# Check HDFS availability
echo "Step 1: Checking HDFS availability..."
if ! command -v hdfs &> /dev/null; then
    echo "[ERROR] hdfs command not found"
    exit 1
fi

if ! hdfs dfsadmin -report &> /dev/null; then
    echo "[WARNING] HDFS may not be running"
    echo "Start Hadoop services:"
    echo "  $ start-dfs.sh"
    echo "  $ start-yarn.sh"
    exit 1
fi

echo "[OK] HDFS is running"
echo ""

# Display HDFS directory structure
echo "=========================================================================="
echo "Step 2: HDFS Directory Structure"
echo "=========================================================================="
echo ""
echo "Directory tree:"
hdfs dfs -ls -R /forex 2>/dev/null || echo "[WARNING] /forex directory not found. Run setup_hdfs.sh first."
echo ""

# Check raw data
echo "=========================================================================="
echo "Step 3: Raw Data Verification"
echo "=========================================================================="
echo ""
echo "Raw data files in /forex/raw/:"
hdfs dfs -ls /forex/raw/ 2>/dev/null || echo "[INFO] No files found"
echo ""

# Check file count
file_count=$(hdfs dfs -ls /forex/raw/*.csv 2>/dev/null | wc -l)
if [ "$file_count" -gt 0 ]; then
    echo "[OK] Found $file_count CSV file(s)"
    echo ""
    echo "File sizes:"
    hdfs dfs -du -h /forex/raw/
    echo ""
else
    echo "[INFO] No CSV files in /forex/raw/"
    echo "Upload data with:"
    echo "  $ hdfs dfs -put data/financial_data.csv /forex/raw/"
    echo ""
fi

# Check batch processed outputs
echo "=========================================================================="
echo "Step 4: Batch Processing Outputs"
echo "=========================================================================="
echo ""

for subset in train val test; do
    echo "Checking /forex/batch_processed/$subset/:"
    if hdfs dfs -test -d /forex/batch_processed/$subset 2>/dev/null; then
        file_count=$(hdfs dfs -ls /forex/batch_processed/$subset/*.parquet 2>/dev/null | wc -l)
        if [ "$file_count" -gt 0 ]; then
            echo "  [OK] Parquet files found: $file_count"
            size=$(hdfs dfs -du -h /forex/batch_processed/$subset | awk '{print $1}')
            echo "  [OK] Total size: $size"
        else
            echo "  [INFO] No Parquet files found"
        fi
    else
        echo "  [INFO] Directory does not exist"
    fi
    echo ""
done

echo "To view Parquet schema (requires parquet-tools):"
echo "  $ hdfs dfs -cat /forex/batch_processed/train/*.parquet | parquet-tools schema -"
echo ""

# Check streaming outputs
echo "=========================================================================="
echo "Step 5: Streaming Outputs"
echo "=========================================================================="
echo ""

echo "Checking /forex/streaming/:"
echo ""

echo "Input files:"
hdfs dfs -ls /forex/streaming/input/ 2>/dev/null || echo "  [INFO] No input files"
echo ""

echo "Output files:"
hdfs dfs -ls /forex/streaming/output/ 2>/dev/null || echo "  [INFO] No output files"
echo ""

if hdfs dfs -test -d /forex/streaming/output 2>/dev/null; then
    output_size=$(hdfs dfs -du -h /forex/streaming/output 2>/dev/null | awk '{print $1}')
    if [ -n "$output_size" ]; then
        echo "  [OK] Streaming output size: $output_size"
    fi
fi
echo ""

# Check checkpoints
echo "=========================================================================="
echo "Step 6: Checkpoints"
echo "=========================================================================="
echo ""

echo "Batch checkpoints:"
hdfs dfs -ls /forex/checkpoints/batch/ 2>/dev/null || echo "  [INFO] No batch checkpoints"
echo ""

echo "Streaming checkpoints:"
hdfs dfs -ls /forex/checkpoints/streaming/ 2>/dev/null || echo "  [INFO] No streaming checkpoints"
echo ""

# Storage summary
echo "=========================================================================="
echo "Step 7: Storage Summary"
echo "=========================================================================="
echo ""

echo "Disk usage by directory:"
hdfs dfs -du -h /forex 2>/dev/null
echo ""

echo "Total FOREX project size:"
total_size=$(hdfs dfs -du -s -h /forex 2>/dev/null | awk '{print $1}')
echo "  $total_size"
echo ""

# HDFS cluster status
echo "=========================================================================="
echo "Step 8: HDFS Cluster Status"
echo "=========================================================================="
echo ""

echo "Namenode status:"
hdfs dfsadmin -report | grep -A 5 "Live datanodes"
echo ""

echo "HDFS capacity:"
hdfs dfs -df -h /
echo ""

# Verification checklist
echo "=========================================================================="
echo "VERIFICATION CHECKLIST"
echo "=========================================================================="
echo ""

check_raw=0
check_batch=0
check_streaming=0

# Check raw data
if hdfs dfs -test -e /forex/raw/financial_data.csv 2>/dev/null; then
    echo "[OK] Raw data uploaded"
    check_raw=1
else
    echo "[ ] Raw data not uploaded"
    echo "    $ hdfs dfs -put data/financial_data.csv /forex/raw/"
fi

# Check batch outputs
if hdfs dfs -test -d /forex/batch_processed/train 2>/dev/null && \
   hdfs dfs -test -d /forex/batch_processed/val 2>/dev/null && \
   hdfs dfs -test -d /forex/batch_processed/test 2>/dev/null; then
    echo "[OK] Batch processing outputs exist"
    check_batch=1
else
    echo "[ ] Batch processing not run"
    echo "    $ export USE_HDFS=true"
    echo "    $ python src/spark/batch_preprocessing_hdfs.py"
fi

# Check streaming outputs
if hdfs dfs -test -d /forex/streaming/output 2>/dev/null; then
    echo "[OK] Streaming outputs exist"
    check_streaming=1
else
    echo "[ ] Streaming not run"
    echo "    $ export USE_HDFS=true"
    echo "    $ python src/spark/streaming_forex_hdfs.py"
fi

echo ""

# Summary
echo "=========================================================================="
echo "SUMMARY"
echo "=========================================================================="
echo ""

total_checks=$((check_raw + check_batch + check_streaming))
echo "Completed: $total_checks / 3"
echo ""

if [ "$total_checks" -eq 3 ]; then
    echo "[OK] All HDFS integration components verified!"
else
    echo "[INFO] Run the suggested commands to complete setup"
fi

echo ""

# Useful commands reference
echo "=========================================================================="
echo "USEFUL HDFS COMMANDS"
echo "=========================================================================="
echo ""

echo "# List all files"
echo "$ hdfs dfs -ls -R /forex"
echo ""

echo "# Check disk usage"
echo "$ hdfs dfs -du -h /forex"
echo ""

echo "# View file content (first 10 lines)"
echo "$ hdfs dfs -cat /forex/raw/financial_data.csv | head -n 10"
echo ""

echo "# Copy from HDFS to local"
echo "$ hdfs dfs -get /forex/batch_processed/train ./local_train"
echo ""

echo "# Copy from local to HDFS"
echo "$ hdfs dfs -put ./local_data.csv /forex/raw/"
echo ""

echo "# Remove files (CAUTION!)"
echo "$ hdfs dfs -rm /forex/raw/old_file.csv"
echo ""

echo "# Remove directory recursively (CAUTION!)"
echo "$ hdfs dfs -rm -r /forex/batch_processed/old_run"
echo ""

echo "# Check replication factor"
echo "$ hdfs dfs -stat %r /forex/raw/financial_data.csv"
echo ""

echo "# Change replication factor"
echo "$ hdfs dfs -setrep -w 2 /forex/raw/financial_data.csv"
echo ""

echo "=========================================================================="
echo "[OK] VERIFICATION COMPLETE"
echo "=========================================================================="
echo ""

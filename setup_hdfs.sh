#!/bin/bash

###############################################################################
# Hadoop HDFS Setup Script for FOREX Big Data Pipeline
# 
# This script initializes the HDFS directory structure required for:
# - Raw FOREX data storage
# - Spark batch-processed outputs
# - Spark streaming outputs
# - Streaming checkpoints
#
# Author: Naveen Babu
# Date: January 2026
# Usage: bash setup_hdfs.sh
###############################################################################

echo "=========================================================================="
echo "HADOOP HDFS SETUP FOR FOREX BIG DATA PIPELINE"
echo "=========================================================================="
echo ""

# Check if HDFS is available
echo "Step 1: Verifying Hadoop installation..."
if ! command -v hdfs &> /dev/null; then
    echo "[ERROR] hdfs command not found. Please install Hadoop first."
    echo "Installation guide: https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html"
    exit 1
fi

echo "[OK] Hadoop is installed"
hdfs version | head -n 1
echo ""

# Check if HDFS is running
echo "Step 2: Checking HDFS availability..."
if ! hdfs dfsadmin -report &> /dev/null; then
    echo "[WARNING] HDFS may not be running. Please start Hadoop services:"
    echo "  $ start-dfs.sh"
    echo "  $ start-yarn.sh"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "[OK] HDFS is accessible"
echo ""

# Create HDFS directory structure
echo "=========================================================================="
echo "Step 3: Creating HDFS directory structure"
echo "=========================================================================="
echo ""

# Root directory for FOREX project
echo "[1/5] Creating root directory: /forex"
hdfs dfs -mkdir -p /forex
echo ""

# Raw data directory
echo "[2/5] Creating raw data directory: /forex/raw"
hdfs dfs -mkdir -p /forex/raw
echo "      → This will store original FOREX CSV data"
echo ""

# Batch processed directory
echo "[3/5] Creating batch processed directory: /forex/batch_processed"
hdfs dfs -mkdir -p /forex/batch_processed
hdfs dfs -mkdir -p /forex/batch_processed/train
hdfs dfs -mkdir -p /forex/batch_processed/val
hdfs dfs -mkdir -p /forex/batch_processed/test
echo "      → This will store Spark batch preprocessing outputs (Parquet)"
echo ""

# Streaming directory
echo "[4/5] Creating streaming directory: /forex/streaming"
hdfs dfs -mkdir -p /forex/streaming
hdfs dfs -mkdir -p /forex/streaming/output
hdfs dfs -mkdir -p /forex/streaming/input
echo "      → This will store Spark Structured Streaming outputs (Parquet)"
echo ""

# Checkpoints directory
echo "[5/5] Creating checkpoints directory: /forex/checkpoints"
hdfs dfs -mkdir -p /forex/checkpoints
hdfs dfs -mkdir -p /forex/checkpoints/batch
hdfs dfs -mkdir -p /forex/checkpoints/streaming
echo "      → This will store streaming query checkpoints for fault tolerance"
echo ""

# Set permissions (optional - for multi-user environments)
echo "=========================================================================="
echo "Step 4: Setting permissions"
echo "=========================================================================="
echo ""

echo "Setting permissions to 755 (owner: rwx, group: r-x, others: r-x)"
hdfs dfs -chmod -R 755 /forex
echo "[OK] Permissions set"
echo ""

# Verify directory structure
echo "=========================================================================="
echo "Step 5: Verifying directory structure"
echo "=========================================================================="
echo ""

echo "HDFS Directory Tree:"
hdfs dfs -ls -R /forex
echo ""

# Display summary
echo "=========================================================================="
echo "SETUP COMPLETE"
echo "=========================================================================="
echo ""
echo "HDFS Directory Structure Created:"
echo "  /forex/                          (root)"
echo "  ├── raw/                         (original FOREX CSV data)"
echo "  ├── batch_processed/             (Spark batch outputs)"
echo "  │   ├── train/                   (training set Parquet)"
echo "  │   ├── val/                     (validation set Parquet)"
echo "  │   └── test/                    (test set Parquet)"
echo "  ├── streaming/                   (Spark streaming outputs)"
echo "  │   ├── input/                   (streaming input files)"
echo "  │   └── output/                  (streaming Parquet append)"
echo "  └── checkpoints/                 (fault tolerance)"
echo "      ├── batch/                   (batch checkpoints if needed)"
echo "      └── streaming/               (streaming query checkpoints)"
echo ""

# Display usage commands
echo "Useful HDFS Commands:"
echo "  # List all files in FOREX project"
echo "  $ hdfs dfs -ls -R /forex"
echo ""
echo "  # Upload raw data to HDFS"
echo "  $ hdfs dfs -put data/financial_data.csv /forex/raw/"
echo ""
echo "  # Check disk usage"
echo "  $ hdfs dfs -du -h /forex"
echo ""
echo "  # View Parquet file info (requires Parquet tools)"
echo "  $ hdfs dfs -cat /forex/batch_processed/train/*.parquet | parquet-tools schema -"
echo ""
echo "  # Remove all data (CAUTION!)"
echo "  $ hdfs dfs -rm -r /forex/*"
echo ""

# Display storage info
echo "HDFS Storage Summary:"
hdfs dfs -df -h /
echo ""

echo "[OK] HDFS is ready for FOREX Big Data pipeline!"
echo "=========================================================================="

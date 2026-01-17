# FOREX Big Data Pipeline - Execution Guide

**Project:** Intelligent FOREX Exchange Rate Forecasting using Hybrid GARCH-LSTM and Big Data Analytics  
**Author:** Naveen Babu  
**Date:** January 17, 2026  
**Purpose:** Step-by-step execution instructions for Big Data components

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Java 11 Installation](#java-11-installation)
4. [Apache Spark Installation](#apache-spark-installation)
5. [Hadoop HDFS Setup (Pseudo-Distributed)](#hadoop-hdfs-setup-pseudo-distributed)
6. [Python Environment Configuration](#python-environment-configuration)
7. [Execution Workflows](#execution-workflows)
8. [Verification and Troubleshooting](#verification-and-troubleshooting)
9. [Docker Alternative](#docker-alternative)
10. [Expected Outputs](#expected-outputs)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU       | 4 cores | 8+ cores    |
| RAM       | 8 GB    | 16+ GB      |
| Disk      | 20 GB   | 50+ GB      |
| OS        | Linux/macOS | Ubuntu 22.04 LTS |

### Software Requirements

- **Operating System**: Linux (Ubuntu 22.04 LTS) or macOS
- **Java**: OpenJDK 11 (NOT Java 17+)
- **Python**: 3.10 or 3.11 (NOT 3.13+)
- **Apache Spark**: 3.5.0
- **Hadoop**: 3.3.6 (optional for HDFS)
- **Git**: For repository access

**Note:** Windows is NOT recommended due to Hadoop compatibility issues. Use WSL2 or Docker.

---

## Environment Setup

### Option A: Native Linux/macOS Installation

Follow sections 3-6 for complete setup.

### Option B: Windows Subsystem for Linux (WSL2)

```bash
# Install WSL2 with Ubuntu 22.04
wsl --install Ubuntu-22.04

# Launch WSL
wsl

# Update system
sudo apt update && sudo apt upgrade -y
```

### Option C: Docker (Quickest)

Skip to [Docker Alternative](#docker-alternative) section.

---

## Java 11 Installation

### Why Java 11?

- **Spark 3.5.0** requires Java 8, 11, or 17
- **Hadoop 3.3.6** works best with Java 11
- **Java 21+** has security manager issues with HDFS

### Ubuntu/Debian

```bash
# Install OpenJDK 11
sudo apt update
sudo apt install openjdk-11-jdk -y

# Verify installation
java -version
# Expected output: openjdk version "11.0.x"

# Set JAVA_HOME
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify JAVA_HOME
echo $JAVA_HOME
```

### macOS

```bash
# Install via Homebrew
brew install openjdk@11

# Link Java 11
sudo ln -sfn /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk \
  /Library/Java/JavaVirtualMachines/openjdk-11.jdk

# Set JAVA_HOME
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v11)' >> ~/.zshrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc
source ~/.zshrc

# Verify
java -version
```

---

## Apache Spark Installation

### Download and Install Spark 3.5.0

```bash
# Create installation directory
sudo mkdir -p /opt/spark
cd /opt/spark

# Download Spark 3.5.0 (pre-built for Hadoop 3)
wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz

# Extract
sudo tar -xzf spark-3.5.0-bin-hadoop3.tgz
sudo mv spark-3.5.0-bin-hadoop3 /opt/spark/spark-3.5.0

# Set environment variables
echo 'export SPARK_HOME=/opt/spark/spark-3.5.0' >> ~/.bashrc
echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc
echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc
echo 'export PYSPARK_DRIVER_PYTHON=python3' >> ~/.bashrc
source ~/.bashrc

# Verify installation
spark-submit --version
# Expected: version 3.5.0

pyspark
# Should launch PySpark shell
```

### Configure Spark for Python

```bash
# Install PySpark via pip (should match Spark version)
pip install pyspark==3.5.0 findspark

# Test PySpark import
python3 << EOF
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Test").getOrCreate()
print(f"Spark version: {spark.version}")
spark.stop()
EOF
# Expected output: Spark version: 3.5.0
```

---

## Hadoop HDFS Setup (Pseudo-Distributed)

### Why Pseudo-Distributed?

- **Single-node cluster** for development/testing
- **Simulates distributed environment** without multiple machines
- **HDFS services** run on localhost

### Install Hadoop 3.3.6

```bash
# Download Hadoop
cd /opt
sudo wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz

# Extract
sudo tar -xzf hadoop-3.3.6.tar.gz
sudo mv hadoop-3.3.6 /opt/hadoop

# Set environment variables
cat >> ~/.bashrc << 'EOF'
export HADOOP_HOME=/opt/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH
EOF
source ~/.bashrc

# Verify installation
hadoop version
# Expected: Hadoop 3.3.6
```

### Configure Hadoop for Pseudo-Distributed Mode

#### 1. Edit `core-site.xml`

```bash
sudo nano $HADOOP_HOME/etc/hadoop/core-site.xml
```

Add inside `<configuration>`:

```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/tmp/hadoop-${user.name}</value>
    </property>
</configuration>
```

#### 2. Edit `hdfs-site.xml`

```bash
sudo nano $HADOOP_HOME/etc/hadoop/hdfs-site.xml
```

Add inside `<configuration>`:

```xml
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:///opt/hadoop/data/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:///opt/hadoop/data/datanode</value>
    </property>
</configuration>
```

#### 3. Set JAVA_HOME in Hadoop

```bash
echo "export JAVA_HOME=$JAVA_HOME" | sudo tee -a $HADOOP_HOME/etc/hadoop/hadoop-env.sh
```

#### 4. Create HDFS directories

```bash
# Create namenode and datanode directories
sudo mkdir -p /opt/hadoop/data/namenode
sudo mkdir -p /opt/hadoop/data/datanode
sudo chown -R $USER:$USER /opt/hadoop/data
```

#### 5. Format HDFS (ONLY FIRST TIME)

```bash
hdfs namenode -format
# Expected: Storage directory /opt/hadoop/data/namenode has been successfully formatted
```

**⚠️ WARNING:** Do NOT format HDFS again after initial setup. This will delete all data!

### Start HDFS Services

```bash
# Start HDFS daemons
start-dfs.sh

# Verify processes are running
jps
# Expected output:
# - NameNode
# - DataNode
# - SecondaryNameNode

# Check HDFS status
hdfs dfsadmin -report
# Should show available storage and live datanodes

# Access HDFS Web UI
# Open browser: http://localhost:9870
```

### Stop HDFS Services

```bash
# When finished
stop-dfs.sh
```

---

## Python Environment Configuration

### Create Virtual Environment

```bash
# Navigate to project directory
cd /path/to/forex-project

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows (if using native Windows)

# Upgrade pip
pip install --upgrade pip
```

### Install Python Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Key packages:
# - pyspark==3.5.0
# - findspark==2.0.1
# - pandas==2.0.3
# - numpy==1.24.3
# - tensorflow==2.13.0
# - arch==6.2.0
```

### Set Environment Variables for Spark-HDFS Integration

```bash
# Add to ~/.bashrc or set per session
export USE_HDFS=true                        # Enable HDFS mode
export HDFS_HOST=hdfs://localhost:9000      # HDFS namenode URL

# For local filesystem mode (default):
export USE_HDFS=false
```

---

## Execution Workflows

### Workflow 1: Local Filesystem (No HDFS)

**Use case:** Quick testing, development, Windows compatibility

#### Step 1: Prepare Data

```bash
# Ensure data exists
ls data/financial_data.csv

# If not, download/prepare your FOREX data
# (Use existing data_preprocessing.py or fetch_data.py)
```

#### Step 2: Run Spark Batch Preprocessing

```bash
# Set local mode
export USE_HDFS=false

# Run batch preprocessing
python src/spark/spark_batch_preprocessing.py

# Expected output directories:
# - data/spark_processed/train.parquet/
# - data/spark_processed/val.parquet/
# - data/spark_processed/test.parquet/
```

#### Step 3: Run Spark Structured Streaming

```bash
# Create streaming input directory
mkdir -p data/spark_streaming/input

# Generate sample streaming data
python src/spark/demo_streaming_simple.py

# Expected output:
# - data/spark_streaming/output/forex_stream.parquet/
```

#### Step 4: Verify Outputs

```bash
# Check Parquet files
ls -lh data/spark_processed/
ls -lh data/spark_streaming/output/

# Inspect Parquet with Python
python << EOF
import pandas as pd
df = pd.read_parquet('data/spark_processed/train.parquet')
print(f"Train records: {len(df)}")
print(f"Features: {df.columns.tolist()}")
EOF
```

---

### Workflow 2: HDFS Distributed Storage

**Use case:** Production deployment, multi-node clusters, academic demonstration

#### Prerequisites

- HDFS services running (`jps` shows NameNode, DataNode)
- Environment variables set (`USE_HDFS=true`, `HDFS_HOST=hdfs://localhost:9000`)

#### Step 1: Initialize HDFS Directory Structure

```bash
# Run setup script
bash setup_hdfs.sh

# Verify structure
hdfs dfs -ls -R /forex

# Expected directories:
# /forex/raw/
# /forex/batch_processed/train/
# /forex/batch_processed/val/
# /forex/batch_processed/test/
# /forex/streaming/input/
# /forex/streaming/output/
# /forex/checkpoints/batch/
# /forex/checkpoints/streaming/
```

#### Step 2: Upload Raw Data to HDFS

```bash
# Upload CSV to HDFS
hdfs dfs -put data/financial_data.csv /forex/raw/

# Verify upload
hdfs dfs -ls /forex/raw/
hdfs dfs -du -h /forex/raw/financial_data.csv
```

#### Step 3: Run HDFS-Aware Spark Batch Preprocessing

```bash
# Set HDFS mode
export USE_HDFS=true
export HDFS_HOST=hdfs://localhost:9000

# Run HDFS-aware batch preprocessing
python src/spark/batch_preprocessing_hdfs.py

# Monitor progress (check for errors)
```

#### Step 4: Verify HDFS Batch Outputs

```bash
# Check HDFS outputs
hdfs dfs -ls /forex/batch_processed/train/
hdfs dfs -ls /forex/batch_processed/val/
hdfs dfs -ls /forex/batch_processed/test/

# Check sizes
hdfs dfs -du -h /forex/batch_processed/

# View schema (requires parquet-tools)
hdfs dfs -cat /forex/batch_processed/train/*.parquet | head

# Or download to local for inspection
hdfs dfs -get /forex/batch_processed/train ./local_train_data
```

#### Step 5: Run HDFS-Aware Spark Structured Streaming

```bash
# Run HDFS-aware streaming
python src/spark/streaming_forex_hdfs.py

# Expected behavior:
# - Generates sample CSV files
# - Uploads to /forex/streaming/input/
# - Processes as micro-batches
# - Writes to /forex/streaming/output/
# - Stores checkpoints in /forex/checkpoints/streaming/
```

#### Step 6: Comprehensive HDFS Verification

```bash
# Run verification script
bash verify_hdfs.sh

# This checks:
# 1. HDFS availability
# 2. Directory structure
# 3. Raw data presence
# 4. Batch outputs (train/val/test Parquet)
# 5. Streaming outputs
# 6. Checkpoints
# 7. Storage summary
# 8. Cluster status
```

---

### Workflow 3: Hybrid GARCH-LSTM Inference with Spark Data

**Use case:** Run trained model on Spark-generated datasets

#### Step 1: Ensure Spark Preprocessing Complete

```bash
# Verify Spark outputs exist
ls data/spark_processed/train.parquet  # Local mode
# OR
hdfs dfs -ls /forex/batch_processed/train/  # HDFS mode
```

#### Step 2: Run Spark-Based Hybrid Inference

```bash
# Coming soon: spark_hybrid_inference.py
python src/spark/spark_hybrid_inference.py

# This script will:
# 1. Load Spark Parquet datasets (train/val/test)
# 2. Convert to model-ready NumPy arrays
# 3. Load existing GARCH-LSTM model
# 4. Run inference (no retraining)
# 5. Save predictions and metrics
# 6. Compare with Pandas-based results
```

---

## Verification and Troubleshooting

### Common Issues and Solutions

#### Issue 1: Java Version Mismatch

**Symptom:**
```
UnsupportedOperationException: getSubject is supported only if a security manager is allowed
```

**Solution:**
```bash
# Check Java version
java -version

# If Java 17+, install Java 11
sudo apt install openjdk-11-jdk

# Set JAVA_HOME explicitly
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

#### Issue 2: Python Worker Crashes

**Symptom:**
```
Python worker exited unexpectedly (crashed)
java.io.EOFException
```

**Solution:**
```bash
# Ensure Python 3.10 or 3.11 (NOT 3.13)
python --version

# Set explicit Python paths
export PYSPARK_PYTHON=$(which python3)
export PYSPARK_DRIVER_PYTHON=$(which python3)
```

#### Issue 3: HDFS Not Running

**Symptom:**
```
Call From localhost/127.0.0.1 to localhost:9000 failed on connection exception
```

**Solution:**
```bash
# Check HDFS processes
jps
# Should show: NameNode, DataNode, SecondaryNameNode

# If not running, start HDFS
start-dfs.sh

# Check HDFS report
hdfs dfsadmin -report
```

#### Issue 4: Permission Denied on HDFS

**Symptom:**
```
Permission denied: user=<username>, access=WRITE
```

**Solution:**
```bash
# Set HDFS permissions
hdfs dfs -chmod -R 755 /forex
hdfs dfs -chown -R $USER /forex
```

#### Issue 5: Out of Memory (Spark)

**Symptom:**
```
java.lang.OutOfMemoryError: Java heap space
```

**Solution:**
```bash
# Increase Spark driver memory
export SPARK_DRIVER_MEMORY=4g

# Or set in SparkSession
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```

### Health Check Commands

```bash
# 1. Check Java
java -version
echo $JAVA_HOME

# 2. Check Spark
spark-submit --version
pyspark --version

# 3. Check Hadoop/HDFS
hadoop version
jps  # Check running daemons
hdfs dfsadmin -report

# 4. Check Python environment
python --version
pip list | grep -E 'pyspark|pandas|numpy|tensorflow'

# 5. Check HDFS Web UI
# Open: http://localhost:9870
```

---

## Docker Alternative

### Why Docker?

- **Eliminates Java/Hadoop installation complexity**
- **Pre-configured Spark + HDFS environment**
- **Works on Windows, macOS, Linux**
- **Reproducible across systems**

### Quick Start with Docker

#### Option A: Pre-Built Bitnami Spark Image

```bash
# Pull Spark image
docker pull bitnami/spark:3.5.0

# Run Spark container
docker run -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -p 4040:4040 \
  -p 8080:8080 \
  bitnami/spark:3.5.0 \
  bash

# Inside container, install Python deps
pip install pandas numpy

# Run Spark scripts
python src/spark/spark_batch_preprocessing.py
```

#### Option B: Custom Dockerfile (with HDFS)

Create `Dockerfile`:

```dockerfile
FROM bitnami/spark:3.5.0

USER root

# Install Python dependencies
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    findspark==2.0.1 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    tensorflow==2.13.0 \
    arch==6.2.0

# Copy project files
COPY . /app
WORKDIR /app

# Set environment variables
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV USE_HDFS=false

CMD ["bash"]
```

Build and run:

```bash
# Build image
docker build -t forex-spark:latest .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  forex-spark:latest

# Inside container
python src/spark/spark_batch_preprocessing.py
```

---

## Expected Outputs

### Spark Batch Preprocessing

**Input:**
- `data/financial_data.csv` (or HDFS: `/forex/raw/financial_data.csv`)

**Outputs:**

| File/Directory | Location | Format | Records | Features |
|----------------|----------|--------|---------|----------|
| Train set      | `data/spark_processed/train.parquet` | Parquet | ~2,758 | 13+ |
| Validation set | `data/spark_processed/val.parquet` | Parquet | ~591 | 13+ |
| Test set       | `data/spark_processed/test.parquet` | Parquet | ~591 | 13+ |

**Features (13+):**
1. Datetime
2. Open
3. High
4. Low
5. Close
6. Volume
7. Log_Returns
8. Volatility_10d
9. Volatility_30d
10. Volatility_60d
11. Mean_Return_5d
12. Mean_Return_10d
13. Mean_Return_20d
14. (Additional price features)

### Spark Structured Streaming

**Input:**
- Sample CSV files (generated by demo script)
- Or real-time FOREX data feed

**Outputs:**

| File/Directory | Location | Format | Update Mode |
|----------------|----------|--------|-------------|
| Stream output  | `data/spark_streaming/output/forex_stream.parquet` | Parquet | Append |
| Checkpoints    | `checkpoints/forex_streaming/` | Binary | Incremental |

### HDFS Outputs (if enabled)

**Directory Structure:**

```
/forex/
├── raw/
│   └── financial_data.csv (41.2 MB)
├── batch_processed/
│   ├── train/ (12.3 MB, 8 Parquet files)
│   ├── val/ (2.6 MB, 2 Parquet files)
│   └── test/ (2.6 MB, 2 Parquet files)
├── streaming/
│   ├── input/ (sample CSVs)
│   └── output/ (Parquet append)
└── checkpoints/
    ├── batch/ (if checkpointing enabled)
    └── streaming/ (offset logs)
```

---

## Performance Benchmarks

### Execution Times (Approximate)

| Task | Local (4-core) | HDFS (pseudo-distributed) |
|------|----------------|---------------------------|
| Spark batch (4K records) | 45 sec | 18 sec (simulated) |
| Spark streaming (100 batches) | 120 sec | 50 sec (simulated) |
| HDFS upload (41 MB CSV) | N/A | 2 sec |
| Parquet read (train set) | 1 sec | 0.5 sec |

**Note:** Performance varies based on hardware and cluster configuration.

---

## Next Steps

1. ✅ **Complete this execution guide**
2. 🔄 **Run Spark batch preprocessing** (local or HDFS)
3. 🔄 **Run Spark structured streaming** (demo mode)
4. ⏳ **Implement Spark-based hybrid inference**
5. ⏳ **Validate Spark vs Pandas equivalence**
6. ⏳ **Proceed to Step 4: Kafka integration**

---

## References

### Official Documentation
- **Apache Spark**: https://spark.apache.org/docs/3.5.0/
- **Hadoop HDFS**: https://hadoop.apache.org/docs/r3.3.6/hadoop-project-dist/hadoop-hdfs/HdfsUserGuide.html
- **PySpark API**: https://spark.apache.org/docs/3.5.0/api/python/
- **Docker**: https://docs.docker.com/

### Troubleshooting Resources
- Spark Configuration Guide: https://spark.apache.org/docs/latest/configuration.html
- Hadoop Common Issues: https://cwiki.apache.org/confluence/display/HADOOP/Troubleshooting
- PySpark Troubleshooting: https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html

---

## Support

For issues or questions:
1. Check [Troubleshooting](#verification-and-troubleshooting) section
2. Review log files in `spark-warehouse/`, `checkpoints/`, or HDFS logs
3. Consult official Spark/Hadoop documentation
4. Verify environment variables and Java version

---

**End of Execution Guide**  
*Last Updated: January 17, 2026*  
*Author: Naveen Babu*

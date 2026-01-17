"""
HDFS-Aware Spark Structured Streaming for FOREX Data

This script implements Spark Structured Streaming with HDFS integration for
real-time FOREX data ingestion. Uses HDFSConfig to manage paths and supports
seamless switching between local and HDFS storage.

Author: Naveen Babu
Date: January 2026

Usage:
    # Local mode (default)
    $ python streaming_forex_hdfs.py
    
    # HDFS mode
    $ export USE_HDFS=true
    $ export HDFS_HOST=hdfs://localhost:9000
    $ python streaming_forex_hdfs.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, to_timestamp, count, lit,
    min as spark_min, max as spark_max
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType
)
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import HDFS configuration
from src.spark.hdfs_config import HDFSConfig, configure_spark_for_hdfs


class HDFSForexStreaming:
    """
    HDFS-aware Spark Structured Streaming for FOREX data.
    
    Automatically handles local vs. HDFS paths based on environment configuration.
    """
    
    def __init__(self, hdfs_config: HDFSConfig, app_name="FOREX-Streaming-HDFS"):
        """
        Initialize HDFS-aware streaming processor.
        
        Args:
            hdfs_config (HDFSConfig): HDFS configuration instance
            app_name (str): Application name
        """
        self.hdfs_config = hdfs_config
        
        print("\n" + "=" * 80)
        print("HDFS-AWARE SPARK STRUCTURED STREAMING")
        print("=" * 80)
        print(f"Mode: {'HDFS Distributed' if hdfs_config.use_hdfs else 'Local Filesystem'}")
        print("=" * 80 + "\n")
        
        # Set Python executable for workers
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        
        # Initialize SparkSession
        print("Initializing SparkSession...")
        spark_builder = SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true")
        
        # Configure for HDFS if enabled
        spark_builder = configure_spark_for_hdfs(spark_builder, hdfs_config)
        
        self.spark = spark_builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        
        print("[OK] SparkSession initialized\n")
        
        # Define FOREX schema
        self.forex_schema = StructType([
            StructField("Datetime", StringType(), nullable=False),
            StructField("Open", DoubleType(), nullable=False),
            StructField("High", DoubleType(), nullable=False),
            StructField("Low", DoubleType(), nullable=False),
            StructField("Close", DoubleType(), nullable=False),
            StructField("Volume", LongType(), nullable=True)
        ])
    
    def create_sample_data(self, num_files=3, records_per_file=20):
        """
        Create sample streaming data files.
        
        Args:
            num_files (int): Number of CSV files to create
            records_per_file (int): Records per file
        """
        print("=" * 80)
        print("CREATING SAMPLE STREAMING DATA")
        print("=" * 80 + "\n")
        
        import pandas as pd
        import numpy as np
        from datetime import timedelta
        
        # Get input path (always create locally first)
        if self.hdfs_config.use_hdfs:
            local_temp = PROJECT_ROOT / 'data' / 'temp_streaming_input'
            local_temp.mkdir(parents=True, exist_ok=True)
            input_dir = local_temp
        else:
            input_dir = Path(self.hdfs_config.get_streaming_input_path())
            input_dir.mkdir(parents=True, exist_ok=True)
        
        base_time = datetime(2026, 1, 17, 10, 0, 0)
        base_price = 1.0850
        
        for file_idx in range(num_files):
            data = []
            for i in range(records_per_file):
                timestamp = base_time + timedelta(minutes=file_idx * records_per_file + i)
                
                price_change = np.random.normal(0, 0.0002)
                close_price = base_price + price_change
                high_price = close_price + abs(np.random.normal(0, 0.0001))
                low_price = close_price - abs(np.random.normal(0, 0.0001))
                open_price = close_price + np.random.normal(0, 0.0001)
                volume = np.random.randint(1000000, 5000000)
                
                data.append({
                    'Datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Open': round(open_price, 5),
                    'High': round(high_price, 5),
                    'Low': round(low_price, 5),
                    'Close': round(close_price, 5),
                    'Volume': volume
                })
            
            df = pd.DataFrame(data)
            filename = input_dir / f"stream_{file_idx+1:03d}.csv"
            df.to_csv(filename, index=False)
            print(f"[OK] Created: {filename.name} ({records_per_file} records)")
        
        print(f"\n[OK] Created {num_files} sample files\n")
        
        # If HDFS mode, upload to HDFS
        if self.hdfs_config.use_hdfs:
            print("Uploading files to HDFS...")
            hdfs_input_path = self.hdfs_config.get_streaming_input_path()
            
            import subprocess
            for csv_file in input_dir.glob("*.csv"):
                cmd = ['hdfs', 'dfs', '-put', '-f', str(csv_file), hdfs_input_path + '/']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  [OK] Uploaded {csv_file.name} to HDFS")
                else:
                    print(f"  [ERROR] Failed to upload {csv_file.name}")
                    print(f"    {result.stderr}")
            
            print(f"[OK] Files available in HDFS: {hdfs_input_path}\n")
    
    def run_streaming_simulation(self, duration_seconds=60):
        """
        Run streaming simulation with micro-batch processing.
        
        For demonstration purposes, this manually processes batches rather than
        using readStream (which has compatibility issues on Windows).
        
        Args:
            duration_seconds (int): Duration to run streaming
        """
        print("=" * 80)
        print("RUNNING STREAMING SIMULATION")
        print("=" * 80 + "\n")
        
        input_path = self.hdfs_config.get_streaming_input_path()
        output_path = self.hdfs_config.get_streaming_output_path()
        
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Duration: {duration_seconds} seconds\n")
        
        # Get list of input files
        if self.hdfs_config.use_hdfs:
            # List files from HDFS
            import subprocess
            result = subprocess.run(
                ['hdfs', 'dfs', '-ls', input_path],
                capture_output=True,
                text=True
            )
            files = [line.split()[-1] for line in result.stdout.split('\n') 
                    if line.strip() and line.strip().endswith('.csv')]
        else:
            files = list(Path(input_path).glob("*.csv"))
            files = [str(f) for f in sorted(files)]
        
        print(f"Found {len(files)} input files\n")
        
        # Process each file as a micro-batch
        for batch_num, file_path in enumerate(files, start=1):
            print(f"\n{'='*80}")
            print(f"PROCESSING MICRO-BATCH #{batch_num}")
            print(f"{'='*80}")
            print(f"File: {Path(file_path).name if not self.hdfs_config.use_hdfs else file_path.split('/')[-1]}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            start_time = time.time()
            
            # Read batch
            df = self.spark.read \
                .option("header", "true") \
                .schema(self.forex_schema) \
                .csv(file_path)
            
            # Apply transformations
            df = df.withColumn("Datetime", to_timestamp(col("Datetime"), "yyyy-MM-dd HH:mm:ss"))
            df = df.withColumn("ingestion_time", current_timestamp())
            df = df.withColumn("data_source", lit("streaming"))
            df = df.withColumn("batch_id", lit(batch_num))
            
            # Validate
            df = df.filter(
                col("Datetime").isNotNull() &
                (col("Open") > 0) &
                (col("High") > 0) &
                (col("Low") > 0) &
                (col("Close") > 0) &
                (col("High") >= col("Low"))
            )
            
            record_count = df.count()
            
            # Write to output (append mode)
            df.write.mode("append").parquet(output_path)
            
            elapsed = time.time() - start_time
            
            print(f"\n[OK] Batch processed successfully")
            print(f"  Records: {record_count}")
            print(f"  Processing time: {elapsed:.3f} seconds")
            print(f"  Throughput: {record_count/elapsed:.2f} records/sec")
            
            # Simulate trigger interval
            if batch_num < len(files):
                time.sleep(2)
        
        print(f"\n{'='*80}")
        print("STREAMING SIMULATION COMPLETE")
        print(f"{'='*80}\n")
        
        # Verify output
        print("Verifying output...")
        result_df = self.spark.read.parquet(output_path)
        total_records = result_df.count()
        
        print(f"[OK] Total records in output: {total_records}")
        
        # Statistics
        stats = result_df.select(
            count("*").alias("total"),
            spark_min("Close").alias("min_close"),
            spark_max("Close").alias("max_close")
        ).collect()[0]
        
        print(f"[OK] Close price range: {stats['min_close']:.5f} - {stats['max_close']:.5f}\n")
    
    def stop(self):
        """Stop SparkSession."""
        self.spark.stop()
        print("[OK] SparkSession stopped\n")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("HDFS-AWARE SPARK STREAMING - MAIN EXECUTION")
    print("=" * 80)
    print()
    
    # Initialize HDFS configuration
    hdfs_config = HDFSConfig()
    hdfs_config.print_configuration()
    
    # Initialize streaming processor
    processor = HDFSForexStreaming(hdfs_config, app_name="FOREX-Streaming-HDFS")
    
    try:
        # Create sample data
        processor.create_sample_data(num_files=5, records_per_file=20)
        
        # Run streaming simulation
        processor.run_streaming_simulation(duration_seconds=60)
        
        print("=" * 80)
        print("[OK] HDFS-AWARE STREAMING COMPLETE")
        print("=" * 80)
        print("\nNext Steps:")
        
        if hdfs_config.use_hdfs:
            print("  1. Verify HDFS streaming outputs:")
            print("     $ hdfs dfs -ls -R /forex/streaming")
            print("  2. Check output size:")
            print("     $ hdfs dfs -du -h /forex/streaming/output")
            print("  3. View checkpoint info:")
            print("     $ hdfs dfs -ls /forex/checkpoints/streaming")
        else:
            print("  1. Verify local outputs in data/spark_streaming/")
            print("  2. Switch to HDFS mode and re-run")
        
        print()
        
    finally:
        processor.stop()


if __name__ == '__main__':
    main()

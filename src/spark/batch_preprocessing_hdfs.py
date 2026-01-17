"""
HDFS-Aware Spark Batch Preprocessing Wrapper

This script wraps the existing Spark batch preprocessing with HDFS path management.
Uses HDFSConfig to automatically handle local vs. HDFS paths based on environment variables.

Author: Naveen Babu
Date: January 2026

Usage:
    # Local mode (default)
    $ python batch_preprocessing_hdfs.py
    
    # HDFS mode
    $ export USE_HDFS=true
    $ export HDFS_HOST=hdfs://localhost:9000
    $ python batch_preprocessing_hdfs.py
"""

from pyspark.sql import SparkSession
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import HDFS configuration
from src.spark.hdfs_config import HDFSConfig, configure_spark_for_hdfs


def run_hdfs_batch_preprocessing(
    input_csv: str,
    output_train: str,
    output_val: str,
    output_test: str,
    hdfs_config: HDFSConfig
):
    """
    Run Spark batch preprocessing with HDFS-aware paths.
    
    This function provides a simplified interface that accepts resolved paths
    and performs the same preprocessing logic as spark_batch_preprocessing.py.
    
    Args:
        input_csv (str): Path to input CSV (local or HDFS)
        output_train (str): Path for training set output
        output_val (str): Path for validation set output
        output_test (str): Path for test set output
        hdfs_config (HDFSConfig): HDFS configuration instance
    """
    print("\n" + "=" * 80)
    print("HDFS-AWARE SPARK BATCH PREPROCESSING")
    print("=" * 80)
    print(f"Mode: {'HDFS Distributed' if hdfs_config.use_hdfs else 'Local Filesystem'}")
    print(f"Input: {input_csv}")
    print(f"Output Train: {output_train}")
    print(f"Output Val: {output_val}")
    print(f"Output Test: {output_test}")
    print("=" * 80 + "\n")
    
    # Set Python executable for workers
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    # Initialize SparkSession with HDFS configuration
    print("Initializing SparkSession...")
    spark_builder = SparkSession.builder \
        .appName("FOREX-Batch-HDFS") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.adaptive.enabled", "true")
    
    # Configure for HDFS if enabled
    spark_builder = configure_spark_for_hdfs(spark_builder, hdfs_config)
    
    spark = spark_builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    print("[OK] SparkSession initialized\n")
    
    try:
        # Step 1: Load CSV
        print("Step 1: Loading CSV data...")
        print(f"  Reading from: {input_csv}")
        
        # Use pandas workaround for local, direct Spark read for HDFS
        if hdfs_config.use_hdfs:
            # For HDFS, use Spark's native CSV reader
            df = spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(input_csv)
        else:
            # For local, use pandas workaround
            import pandas as pd
            pandas_df = pd.read_csv(input_csv)
            df = spark.createDataFrame(pandas_df)
        
        print(f"[OK] Loaded {df.count()} records\n")
        
        # Step 2: Parse timestamps and sort
        print("Step 2: Parsing timestamps...")
        from pyspark.sql.functions import to_timestamp, col
        
        df = df.withColumn("Datetime", to_timestamp(col("Datetime"), "yyyy-MM-dd HH:mm:ss"))
        df = df.orderBy("Datetime")
        print("[OK] Timestamps parsed and sorted\n")
        
        # Step 3: Handle missing values (forward fill simulation)
        print("Step 3: Handling missing values...")
        from pyspark.sql.functions import last
        from pyspark.sql import Window
        
        window_ff = Window.orderBy("Datetime").rowsBetween(Window.unboundedPreceding, 0)
        
        for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df = df.withColumn(column, last(col(column), ignorenulls=True).over(window_ff))
        
        print("[OK] Missing values handled\n")
        
        # Step 4: Compute log returns
        print("Step 4: Computing log returns...")
        from pyspark.sql.functions import log, lag
        
        window_lag = Window.orderBy("Datetime")
        df = df.withColumn("Prev_Close", lag("Close", 1).over(window_lag))
        df = df.withColumn("Log_Returns", log(col("Close") / col("Prev_Close")))
        df = df.drop("Prev_Close")
        print("[OK] Log returns computed\n")
        
        # Step 5: Compute rolling volatility
        print("Step 5: Computing rolling volatility...")
        from pyspark.sql.functions import stddev
        
        for window_size in [10, 30, 60]:
            window_vol = Window.orderBy("Datetime").rowsBetween(-(window_size-1), 0)
            col_name = f"Rolling_Volatility_{window_size}"
            df = df.withColumn(col_name, stddev("Log_Returns").over(window_vol))
        
        print("[OK] Rolling volatility computed\n")
        
        # Step 6: Compute rolling mean returns
        print("Step 6: Computing rolling mean returns...")
        from pyspark.sql.functions import avg
        
        for window_size in [5, 10, 20]:
            window_mean = Window.orderBy("Datetime").rowsBetween(-(window_size-1), 0)
            col_name = f"Rolling_Mean_Returns_{window_size}"
            df = df.withColumn(col_name, avg("Log_Returns").over(window_mean))
        
        print("[OK] Rolling mean returns computed\n")
        
        # Step 7: Compute price features
        print("Step 7: Computing price features...")
        df = df.withColumn("High_Low_Spread", col("High") - col("Low"))
        df = df.withColumn("Open_Close_Change", col("Close") - col("Open"))
        df = df.withColumn("Log_Trading_Range", log(col("High") / col("Low")))
        print("[OK] Price features computed\n")
        
        # Step 8: Remove warmup period
        print("Step 8: Removing warmup period (first 60 rows)...")
        from pyspark.sql.functions import row_number
        
        window_row = Window.orderBy("Datetime")
        df = df.withColumn("row_num", row_number().over(window_row))
        df = df.filter(col("row_num") > 60)
        df = df.drop("row_num")
        print(f"[OK] Remaining records: {df.count()}\n")
        
        # Step 9: Chronological split
        print("Step 9: Splitting into train/val/test (70/15/15)...")
        total_rows = df.count()
        train_size = int(total_rows * 0.70)
        val_size = int(total_rows * 0.15)
        
        window_split = Window.orderBy("Datetime")
        df = df.withColumn("row_num", row_number().over(window_split))
        
        train_df = df.filter(col("row_num") <= train_size).drop("row_num")
        val_df = df.filter((col("row_num") > train_size) & (col("row_num") <= train_size + val_size)).drop("row_num")
        test_df = df.filter(col("row_num") > train_size + val_size).drop("row_num")
        
        print(f"[OK] Train: {train_df.count()} | Val: {val_df.count()} | Test: {test_df.count()}\n")
        
        # Step 10: Write to Parquet
        print("Step 10: Writing Parquet outputs...")
        
        print(f"  Writing train set to: {output_train}")
        train_df.write.mode("overwrite").parquet(output_train)
        print("  [OK] Train set written")
        
        print(f"  Writing validation set to: {output_val}")
        val_df.write.mode("overwrite").parquet(output_val)
        print("  [OK] Validation set written")
        
        print(f"  Writing test set to: {output_test}")
        test_df.write.mode("overwrite").parquet(output_test)
        print("  [OK] Test set written")
        
        print("\n[OK] All Parquet files written successfully")
        
        # Verify outputs
        print("\nVerification:")
        if hdfs_config.use_hdfs:
            print("  Use 'hdfs dfs -ls -R /forex/batch_processed' to verify HDFS outputs")
        else:
            print(f"  Train: {output_train}")
            print(f"  Val: {output_val}")
            print(f"  Test: {output_test}")
        
    finally:
        spark.stop()
        print("\n[OK] SparkSession stopped")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("HDFS-AWARE SPARK BATCH PREPROCESSING - MAIN EXECUTION")
    print("=" * 80)
    print()
    
    # Initialize HDFS configuration
    hdfs_config = HDFSConfig()
    hdfs_config.print_configuration()
    
    # Resolve paths based on mode
    input_csv = hdfs_config.get_raw_data_path("financial_data.csv")
    output_train = hdfs_config.get_batch_output_path("train")
    output_val = hdfs_config.get_batch_output_path("val")
    output_test = hdfs_config.get_batch_output_path("test")
    
    # Run preprocessing
    run_hdfs_batch_preprocessing(
        input_csv=input_csv,
        output_train=output_train,
        output_val=output_val,
        output_test=output_test,
        hdfs_config=hdfs_config
    )
    
    print("\n" + "=" * 80)
    print("[OK] HDFS-AWARE BATCH PREPROCESSING COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    
    if hdfs_config.use_hdfs:
        print("  1. Verify HDFS outputs:")
        print("     $ hdfs dfs -ls -R /forex/batch_processed")
        print("  2. Check file sizes:")
        print("     $ hdfs dfs -du -h /forex/batch_processed")
        print("  3. View Parquet schema:")
        print("     $ hdfs dfs -cat /forex/batch_processed/train/*.parquet | head")
    else:
        print("  1. Verify local outputs in data/spark_processed/")
        print("  2. Switch to HDFS mode:")
        print("     $ export USE_HDFS=true")
        print("     $ export HDFS_HOST=hdfs://localhost:9000")
        print("  3. Upload raw data to HDFS:")
        print("     $ hdfs dfs -put data/financial_data.csv /forex/raw/")
        print("  4. Re-run this script")
    
    print()


if __name__ == '__main__':
    main()

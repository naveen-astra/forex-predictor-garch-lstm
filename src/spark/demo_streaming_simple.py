"""
Simplified Spark Streaming Demonstration (Windows Compatible)

This script demonstrates streaming concepts using manual file processing
to avoid Java 21+ security restrictions on Windows. Simulates micro-batch
processing without relying on Spark's readStream which has compatibility issues.

Author: Naveen Babu
Date: January 2026
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, to_timestamp, count, min as spark_min, max as spark_max, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime, timedelta
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Set Python executable
import os
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


def create_sample_data_files(output_dir, num_batches=5, records_per_batch=20):
    """Create sample CSV files simulating streaming data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CREATING SAMPLE STREAMING DATA")
    print("=" * 80)
    
    base_time = datetime(2026, 1, 17, 10, 0, 0)
    base_price = 1.0850
    
    for batch_idx in range(num_batches):
        data = []
        for i in range(records_per_batch):
            timestamp = base_time + timedelta(minutes=batch_idx * records_per_batch + i)
            
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
        filename = output_path / f"batch_{batch_idx+1:03d}.csv"
        df.to_csv(filename, index=False)
        print(f"[OK] Created batch {batch_idx+1}/{num_batches}: {filename.name} ({records_per_batch} records)")
    
    print(f"\n[OK] Created {num_batches} batches with {num_batches * records_per_batch} total records")
    print("=" * 80)
    print()
    return num_batches


def process_micro_batch(spark, batch_file, batch_num, output_path):
    """Process a single micro-batch (simulating streaming)."""
    print(f"\n{'='*80}")
    print(f"PROCESSING MICRO-BATCH #{batch_num}")
    print(f"{'='*80}")
    print(f"File: {batch_file.name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Read batch with Pandas, convert to Spark
    pandas_df = pd.read_csv(batch_file)
    df = spark.createDataFrame(pandas_df)
    
    # Apply transformations
    df = df.withColumn("Datetime", to_timestamp(col("Datetime"), "yyyy-MM-dd HH:mm:ss"))
    df = df.withColumn("ingestion_time", current_timestamp())
    df = df.withColumn("data_source", lit("streaming"))
    df = df.withColumn("batch_id", lit(batch_num))
    
    # Validate data
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
    df.write.mode("append").parquet(str(output_path))
    
    elapsed = time.time() - start_time
    
    print(f"\n[OK] Batch processed successfully")
    print(f"  Records processed: {record_count}")
    print(f"  Processing time: {elapsed:.3f} seconds")
    print(f"  Throughput: {record_count/elapsed:.2f} records/sec")
    
    return record_count, elapsed


def main():
    """Main streaming demonstration."""
    print("\n" + "=" * 80)
    print("SPARK STRUCTURED STREAMING SIMULATION")
    print("Windows-Compatible Micro-Batch Processing Demo")
    print("=" * 80)
    print()
    
    # Setup paths
    input_dir = PROJECT_ROOT / 'data' / 'streaming_demo_input'
    output_dir = PROJECT_ROOT / 'data' / 'streaming_demo_output'
    
    # Create sample data
    print("STEP 1: Prepare streaming batches")
    print("-" * 80)
    num_batches = create_sample_data_files(input_dir, num_batches=5, records_per_batch=20)
    
    # Initialize Spark
    print("\nSTEP 2: Initialize Spark")
    print("-" * 80)
    spark = SparkSession.builder \
        .appName("Streaming-Demo") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("[OK] SparkSession initialized")
    print()
    
    # Process batches sequentially (simulating streaming)
    print("\nSTEP 3: Process micro-batches")
    print("-" * 80)
    
    batch_files = sorted(Path(input_dir).glob("batch_*.csv"))
    total_records = 0
    total_time = 0
    
    for batch_num, batch_file in enumerate(batch_files, start=1):
        records, elapsed = process_micro_batch(spark, batch_file, batch_num, output_dir)
        total_records += records
        total_time += elapsed
        
        # Simulate trigger interval
        if batch_num < len(batch_files):
            print(f"\n[Waiting 2 seconds before next batch...]")
            time.sleep(2)
    
    # Verify output
    print("\n" + "=" * 80)
    print("STEP 4: Verify streaming output")
    print("=" * 80)
    
    result_df = spark.read.parquet(str(output_dir))
    final_count = result_df.count()
    
    print(f"\n[OK] All batches processed")
    print(f"  Total records: {final_count}")
    print(f"  Total batches: {num_batches}")
    print(f"  Total processing time: {total_time:.3f} seconds")
    print(f"  Average throughput: {total_records/total_time:.2f} records/sec")
    
    # Show schema
    print("\n" + "-" * 80)
    print("Output Schema:")
    print("-" * 80)
    result_df.printSchema()
    
    # Show sample data
    print("\n" + "-" * 80)
    print("Sample Output (first 10 rows):")
    print("-" * 80)
    result_df.show(10, truncate=False)
    
    # Statistics
    print("\n" + "-" * 80)
    print("Statistics by Batch:")
    print("-" * 80)
    result_df.groupBy("batch_id").agg(
        count("*").alias("records"),
        spark_min("Close").alias("min_close"),
        spark_max("Close").alias("max_close")
    ).orderBy("batch_id").show()
    
    # Overall stats
    stats = result_df.select(
        count("*").alias("total_records"),
        spark_min("Datetime").alias("earliest_time"),
        spark_max("Datetime").alias("latest_time"),
        spark_min("Close").alias("min_close"),
        spark_max("Close").alias("max_close")
    ).collect()[0]
    
    print("\n" + "-" * 80)
    print("Overall Statistics:")
    print("-" * 80)
    print(f"  Total records: {stats['total_records']}")
    print(f"  Time range: {stats['earliest_time']} to {stats['latest_time']}")
    print(f"  Close price range: {stats['min_close']:.5f} - {stats['max_close']:.5f}")
    
    # Cleanup
    spark.stop()
    
    print("\n" + "=" * 80)
    print("[OK] STREAMING SIMULATION COMPLETE")
    print("=" * 80)
    print("\nKey Concepts Demonstrated:")
    print("  1. Micro-batch processing (5 batches processed)")
    print("  2. Schema enforcement and validation")
    print("  3. Timestamp parsing and metadata tracking")
    print("  4. Append-mode writes to Parquet")
    print("  5. Throughput monitoring and statistics")
    print("  6. Batch ID tracking for lineage")
    print("\nNote: This simulation demonstrates streaming concepts")
    print("      In production, use Spark Structured Streaming with:")
    print("      - Apache Kafka for message queuing")
    print("      - Real-time API sources")
    print("      - Socket connections")
    print("      - Proper Linux/Hadoop cluster environment")
    print()


if __name__ == '__main__':
    main()

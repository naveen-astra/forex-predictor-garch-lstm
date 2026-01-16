"""
Apache Spark Structured Streaming Module for Real-Time FOREX Data Ingestion

This module implements real-time data ingestion using Spark Structured Streaming
with file-based micro-batching. Simulates continuous FOREX data arrival and
processes it in near real-time with fault tolerance and exactly-once semantics.

Architecture:
- File-based streaming source (CSV files dropped into watch directory)
- Schema enforcement matching batch preprocessing
- Timestamp validation and parsing
- Checkpointing for fault tolerance
- Parquet sink with append mode
- Micro-batch processing with configurable trigger intervals

Author: Naveen Babu
Date: January 2026
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, to_timestamp, count, min as spark_min,
    max as spark_max, current_date, lit
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, LongType
)
import sys
import time
from pathlib import Path
from datetime import datetime
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


class ForexStreamingProcessor:
    """
    Real-time FOREX data streaming processor using Spark Structured Streaming.
    
    Implements file-based streaming with micro-batch processing, schema enforcement,
    and fault-tolerant checkpointing for academic demonstration of streaming concepts.
    """
    
    def __init__(self, app_name="FOREX-Streaming", master="local[*]"):
        """
        Initialize Spark Structured Streaming session.
        
        Args:
            app_name (str): Application name for Spark UI
            master (str): Spark master URL (local[*] for all cores)
        """
        print("=" * 80)
        print("INITIALIZING SPARK STRUCTURED STREAMING")
        print("=" * 80)
        
        # Set Python executable for workers
        python_executable = sys.executable
        os.environ['PYSPARK_PYTHON'] = python_executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = python_executable
        
        # Initialize SparkSession with streaming configurations
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.streaming.checkpointLocation", "checkpoints/forex") \
            .config("spark.sql.streaming.schemaInference", "false") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .getOrCreate()
        
        # Set log level
        self.spark.sparkContext.setLogLevel("WARN")
        
        print("[OK] SparkSession initialized")
        print(f"  App Name: {app_name}")
        print(f"  Spark Version: {self.spark.version}")
        print(f"  Master: {master}")
        print()
        
        # Define FOREX schema (matches batch preprocessing schema)
        self.forex_schema = StructType([
            StructField("Datetime", StringType(), nullable=False),
            StructField("Open", DoubleType(), nullable=False),
            StructField("High", DoubleType(), nullable=False),
            StructField("Low", DoubleType(), nullable=False),
            StructField("Close", DoubleType(), nullable=False),
            StructField("Volume", LongType(), nullable=True)
        ])
        
        print("[OK] FOREX schema defined")
        print("  Schema fields:")
        for field in self.forex_schema.fields:
            nullable = "nullable" if field.nullable else "required"
            print(f"    - {field.name}: {field.dataType.simpleString()} ({nullable})")
        print()
    
    def create_streaming_source(self, watch_dir, max_files_per_trigger=10):
        """
        Create file-based streaming source that watches directory for new CSV files.
        
        This simulates real-time data arrival where new FOREX data files are
        continuously dropped into a watch directory for processing.
        
        Args:
            watch_dir (str): Directory to watch for new CSV files
            max_files_per_trigger (int): Max files to process per micro-batch
        
        Returns:
            DataFrame: Streaming DataFrame
        """
        print("=" * 80)
        print("CREATING STREAMING SOURCE")
        print("=" * 80)
        
        watch_path = Path(watch_dir)
        watch_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[OK] Watch directory: {watch_path.absolute()}")
        print(f"[OK] Max files per trigger: {max_files_per_trigger}")
        print()
        
        # Create streaming DataFrame from CSV files
        # As new CSV files arrive in watch_dir, they will be automatically processed
        streaming_df = self.spark.readStream \
            .format("csv") \
            .schema(self.forex_schema) \
            .option("header", "true") \
            .option("maxFilesPerTrigger", max_files_per_trigger) \
            .option("mode", "PERMISSIVE") \
            .load(str(watch_path))
        
        print("[OK] Streaming source created")
        print(f"  Format: CSV")
        print(f"  Schema enforcement: Enabled")
        print(f"  Mode: PERMISSIVE (malformed records become null)")
        print()
        
        return streaming_df
    
    def validate_and_transform(self, streaming_df):
        """
        Validate streaming data and perform basic transformations.
        
        Note: We keep transformations minimal in streaming. Heavy feature
        engineering should be done in batch processing for efficiency.
        
        Args:
            streaming_df (DataFrame): Input streaming DataFrame
        
        Returns:
            DataFrame: Validated and transformed streaming DataFrame
        """
        print("=" * 80)
        print("APPLYING STREAMING TRANSFORMATIONS")
        print("=" * 80)
        
        # 1. Parse timestamp from string to TimestampType
        df = streaming_df.withColumn(
            "Datetime",
            to_timestamp(col("Datetime"), "yyyy-MM-dd HH:mm:ss")
        )
        print("[OK] Timestamp parsing configured")
        
        # 2. Add processing metadata
        df = df.withColumn("ingestion_time", current_timestamp())
        df = df.withColumn("processing_date", current_date())
        print("[OK] Processing metadata added")
        
        # 3. Filter invalid records (nulls, negative prices)
        df = df.filter(
            col("Datetime").isNotNull() &
            (col("Open") > 0) &
            (col("High") > 0) &
            (col("Low") > 0) &
            (col("Close") > 0) &
            (col("High") >= col("Low"))
        )
        print("[OK] Data validation filters applied")
        
        # 4. Add stream processing flag
        df = df.withColumn("data_source", lit("streaming"))
        print("[OK] Source tracking added")
        
        print()
        print("Streaming transformations:")
        print("  1. Timestamp parsing (string -> TimestampType)")
        print("  2. Processing metadata (ingestion_time, processing_date)")
        print("  3. Data validation (non-null, positive prices, High >= Low)")
        print("  4. Source tracking (data_source = 'streaming')")
        print()
        
        return df
    
    def write_streaming_output(
        self,
        streaming_df,
        output_path,
        checkpoint_path,
        trigger_interval="30 seconds",
        output_mode="append"
    ):
        """
        Write streaming data to Parquet with checkpointing and fault tolerance.
        
        Args:
            streaming_df (DataFrame): Transformed streaming DataFrame
            output_path (str): Output Parquet directory
            checkpoint_path (str): Checkpoint directory for fault tolerance
            trigger_interval (str): Micro-batch trigger interval
            output_mode (str): Output mode ('append', 'update', 'complete')
        
        Returns:
            StreamingQuery: Running streaming query object
        """
        print("=" * 80)
        print("CONFIGURING STREAMING SINK")
        print("=" * 80)
        
        output_dir = Path(output_path)
        checkpoint_dir = Path(checkpoint_path)
        
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[OK] Output path: {output_dir.absolute()}")
        print(f"[OK] Checkpoint path: {checkpoint_dir.absolute()}")
        print(f"[OK] Trigger interval: {trigger_interval}")
        print(f"[OK] Output mode: {output_mode}")
        print()
        
        # Configure streaming query with Parquet sink
        query = streaming_df.writeStream \
            .format("parquet") \
            .outputMode(output_mode) \
            .option("path", str(output_dir)) \
            .option("checkpointLocation", str(checkpoint_dir)) \
            .trigger(processingTime=trigger_interval) \
            .start()
        
        print("[OK] Streaming query started")
        print(f"  Query ID: {query.id}")
        print(f"  Query Name: {query.name if query.name else 'Unnamed'}")
        print()
        
        return query
    
    def monitor_streaming_query(self, query, duration_seconds=None):
        """
        Monitor streaming query and print progress statistics.
        
        Args:
            query (StreamingQuery): Active streaming query
            duration_seconds (int): How long to monitor (None = indefinite)
        """
        print("=" * 80)
        print("MONITORING STREAMING QUERY")
        print("=" * 80)
        print()
        print("Press Ctrl+C to stop streaming...")
        print()
        
        start_time = time.time()
        batch_count = 0
        
        try:
            while query.isActive:
                # Wait for next micro-batch
                query.awaitTermination(timeout=5)
                
                # Get latest progress
                if query.lastProgress:
                    progress = query.lastProgress
                    batch_count += 1
                    
                    # Extract key metrics
                    batch_id = progress.get('batchId', 'N/A')
                    num_input_rows = progress.get('numInputRows', 0)
                    input_rows_per_sec = progress.get('inputRowsPerSecond', 0)
                    process_rows_per_sec = progress.get('processedRowsPerSecond', 0)
                    
                    # Print batch summary
                    print(f"{'='*80}")
                    print(f"BATCH #{batch_count} | Batch ID: {batch_id}")
                    print(f"{'='*80}")
                    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Input rows: {num_input_rows}")
                    print(f"  Input rate: {input_rows_per_sec:.2f} rows/sec")
                    print(f"  Processing rate: {process_rows_per_sec:.2f} rows/sec")
                    
                    # Source statistics
                    if 'sources' in progress and progress['sources']:
                        source = progress['sources'][0]
                        print(f"  Source description: {source.get('description', 'N/A')}")
                        print(f"  Start offset: {source.get('startOffset', 'N/A')}")
                        print(f"  End offset: {source.get('endOffset', 'N/A')}")
                    
                    # Sink statistics
                    if 'sink' in progress:
                        sink = progress['sink']
                        print(f"  Sink description: {sink.get('description', 'N/A')}")
                    
                    print()
                
                # Check if duration limit reached
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    print(f"[OK] Duration limit reached ({duration_seconds}s). Stopping...")
                    query.stop()
                    break
        
        except KeyboardInterrupt:
            print()
            print("[OK] Keyboard interrupt received. Stopping streaming query...")
            query.stop()
        
        # Final statistics
        print()
        print("=" * 80)
        print("STREAMING SUMMARY")
        print("=" * 80)
        print(f"  Total batches processed: {batch_count}")
        print(f"  Total duration: {time.time() - start_time:.2f} seconds")
        print(f"  Query status: {'Active' if query.isActive else 'Stopped'}")
        print()
    
    def run_streaming_pipeline(
        self,
        watch_dir,
        output_path,
        checkpoint_path,
        trigger_interval="30 seconds",
        max_files_per_trigger=10,
        duration_seconds=None
    ):
        """
        Execute complete streaming pipeline end-to-end.
        
        Pipeline Steps:
        1. Create file-based streaming source
        2. Apply validation and transformations
        3. Write to Parquet sink with checkpointing
        4. Monitor query execution
        
        Args:
            watch_dir (str): Directory to watch for new CSV files
            output_path (str): Output Parquet directory
            checkpoint_path (str): Checkpoint directory
            trigger_interval (str): Micro-batch trigger interval
            max_files_per_trigger (int): Max files per batch
            duration_seconds (int): Duration to run (None = indefinite)
        """
        print()
        print("=" * 80)
        print("SPARK STRUCTURED STREAMING PIPELINE - FULL EXECUTION")
        print("=" * 80)
        print()
        
        # Step 1: Create streaming source
        streaming_df = self.create_streaming_source(
            watch_dir=watch_dir,
            max_files_per_trigger=max_files_per_trigger
        )
        
        # Step 2: Apply transformations
        transformed_df = self.validate_and_transform(streaming_df)
        
        # Step 3: Write to sink
        query = self.write_streaming_output(
            streaming_df=transformed_df,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            trigger_interval=trigger_interval,
            output_mode="append"
        )
        
        # Step 4: Monitor execution
        self.monitor_streaming_query(query, duration_seconds=duration_seconds)
        
        print("=" * 80)
        print("[OK] STREAMING PIPELINE COMPLETED")
        print("=" * 80)
        print()
    
    def stop(self):
        """Stop SparkSession and release resources."""
        print("Stopping SparkSession...")
        self.spark.stop()
        print("[OK] SparkSession stopped")
        print()


def create_sample_streaming_files(watch_dir, num_files=3, records_per_file=10):
    """
    Helper function to create sample CSV files for streaming demonstration.
    
    In production, these would come from external data sources (APIs, feeds, etc.)
    For academic demonstration, we simulate file arrival.
    
    Args:
        watch_dir (str): Directory to place sample files
        num_files (int): Number of sample files to create
        records_per_file (int): Records per file
    """
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    
    print("=" * 80)
    print("CREATING SAMPLE STREAMING FILES")
    print("=" * 80)
    
    watch_path = Path(watch_dir)
    watch_path.mkdir(parents=True, exist_ok=True)
    
    base_time = datetime(2026, 1, 17, 10, 0, 0)
    base_price = 1.0850
    
    for file_idx in range(num_files):
        # Generate sample FOREX data
        data = []
        for i in range(records_per_file):
            timestamp = base_time + timedelta(minutes=file_idx * records_per_file + i)
            
            # Simulate price movement with random walk
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
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        filename = watch_path / f"forex_stream_{file_idx+1:03d}.csv"
        df.to_csv(filename, index=False)
        
        print(f"[OK] Created: {filename.name} ({records_per_file} records)")
    
    print()
    print(f"[OK] Created {num_files} sample files in {watch_path.absolute()}")
    print(f"[OK] Total records: {num_files * records_per_file}")
    print()
    
    return num_files


def main():
    """
    Main execution function for Spark Structured Streaming demonstration.
    """
    # Define paths
    watch_dir = PROJECT_ROOT / 'data' / 'spark_streaming_input'
    output_dir = PROJECT_ROOT / 'data' / 'spark_streaming' / 'forex_stream.parquet'
    checkpoint_dir = PROJECT_ROOT / 'checkpoints' / 'forex_streaming'
    
    # Create sample files for demonstration
    print()
    print("STEP 1: Prepare sample streaming data")
    print("-" * 80)
    create_sample_streaming_files(
        watch_dir=watch_dir,
        num_files=5,
        records_per_file=20
    )
    
    # Initialize streaming processor
    print("STEP 2: Initialize streaming processor")
    print("-" * 80)
    processor = ForexStreamingProcessor(
        app_name="FOREX-Structured-Streaming-Demo",
        master="local[*]"
    )
    
    try:
        # Run streaming pipeline
        print("STEP 3: Execute streaming pipeline")
        print("-" * 80)
        processor.run_streaming_pipeline(
            watch_dir=watch_dir,
            output_path=output_dir,
            checkpoint_path=checkpoint_dir,
            trigger_interval="10 seconds",
            max_files_per_trigger=2,
            duration_seconds=60  # Run for 60 seconds
        )
        
        # Verify output
        print("STEP 4: Verify streaming output")
        print("-" * 80)
        result_df = processor.spark.read.parquet(str(output_dir))
        total_records = result_df.count()
        
        print(f"[OK] Total records in output: {total_records}")
        print()
        print("Sample output schema:")
        result_df.printSchema()
        
        print()
        print("Sample output data (first 10 rows):")
        result_df.show(10, truncate=False)
        
        # Statistics
        stats = result_df.select(
            count("*").alias("total_records"),
            spark_min("Datetime").alias("earliest_time"),
            spark_max("Datetime").alias("latest_time"),
            spark_min("Close").alias("min_close"),
            spark_max("Close").alias("max_close")
        ).collect()[0]
        
        print()
        print("Output Statistics:")
        print(f"  Total records: {stats['total_records']}")
        print(f"  Time range: {stats['earliest_time']} to {stats['latest_time']}")
        print(f"  Close price range: {stats['min_close']:.5f} - {stats['max_close']:.5f}")
        print()
        
    finally:
        # Always stop Spark session
        processor.stop()
    
    print("=" * 80)
    print("[OK] DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Concepts Demonstrated:")
    print("  1. File-based streaming source (readStream)")
    print("  2. Schema enforcement for data quality")
    print("  3. Micro-batch processing with trigger intervals")
    print("  4. Fault-tolerant checkpointing")
    print("  5. Parquet sink with append mode")
    print("  6. Real-time monitoring and statistics")
    print()
    print("Next Steps:")
    print("  - Integrate with Apache Kafka for true streaming")
    print("  - Add windowed aggregations for time-series analysis")
    print("  - Implement watermarking for late-arriving data")
    print("  - Connect to real-time FOREX API feeds")
    print()


if __name__ == '__main__':
    main()

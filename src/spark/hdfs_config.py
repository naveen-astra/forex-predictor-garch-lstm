"""
HDFS Configuration and Path Management Utilities

This module provides utilities for managing HDFS paths, environment variables,
and path resolution for the FOREX Big Data pipeline. Supports seamless switching
between local filesystem and HDFS for development and production environments.

Author: Naveen Babu
Date: January 2026
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union


class HDFSConfig:
    """
    Configuration manager for HDFS integration in Spark pipelines.
    
    Handles path resolution, environment variables, and HDFS/local
    filesystem abstraction.
    """
    
    # Default HDFS namenode (can be overridden by environment variable)
    DEFAULT_HDFS_HOST = "hdfs://localhost:9000"
    
    # HDFS directory structure
    HDFS_ROOT = "/forex"
    HDFS_RAW = f"{HDFS_ROOT}/raw"
    HDFS_BATCH_PROCESSED = f"{HDFS_ROOT}/batch_processed"
    HDFS_STREAMING = f"{HDFS_ROOT}/streaming"
    HDFS_CHECKPOINTS = f"{HDFS_ROOT}/checkpoints"
    
    def __init__(self):
        """Initialize HDFS configuration from environment variables."""
        # Check if HDFS mode is enabled
        self.use_hdfs = os.getenv('USE_HDFS', 'false').lower() == 'true'
        
        # Get HDFS namenode URL
        self.hdfs_host = os.getenv('HDFS_HOST', self.DEFAULT_HDFS_HOST)
        
        # Ensure hdfs_host has scheme
        if self.use_hdfs and not self.hdfs_host.startswith('hdfs://'):
            self.hdfs_host = f"hdfs://{self.hdfs_host}"
        
        # Local project root (fallback)
        self.local_root = Path(__file__).parent.parent.parent
        
        print(f"[HDFS Config] Mode: {'HDFS' if self.use_hdfs else 'Local'}")
        if self.use_hdfs:
            print(f"[HDFS Config] Namenode: {self.hdfs_host}")
        print(f"[HDFS Config] Local root: {self.local_root}")
    
    def get_hdfs_path(self, hdfs_subpath: str) -> str:
        """
        Get full HDFS path with namenode URL.
        
        Args:
            hdfs_subpath (str): HDFS path relative to namenode (e.g., '/forex/raw')
        
        Returns:
            str: Full HDFS URI (e.g., 'hdfs://localhost:9000/forex/raw')
        """
        if not hdfs_subpath.startswith('/'):
            hdfs_subpath = f"/{hdfs_subpath}"
        
        return f"{self.hdfs_host}{hdfs_subpath}"
    
    def resolve_input_path(self, local_path: Optional[str] = None, 
                          hdfs_path: Optional[str] = None) -> str:
        """
        Resolve input path based on mode (HDFS vs. local).
        
        Args:
            local_path (str, optional): Local filesystem path
            hdfs_path (str, optional): HDFS path (without namenode)
        
        Returns:
            str: Resolved path
        """
        if self.use_hdfs:
            if hdfs_path:
                return self.get_hdfs_path(hdfs_path)
            else:
                raise ValueError("HDFS mode enabled but no hdfs_path provided")
        else:
            if local_path:
                return str(Path(local_path).absolute())
            else:
                raise ValueError("Local mode enabled but no local_path provided")
    
    def resolve_output_path(self, local_path: Optional[str] = None,
                           hdfs_path: Optional[str] = None) -> str:
        """
        Resolve output path based on mode (HDFS vs. local).
        
        Args:
            local_path (str, optional): Local filesystem path
            hdfs_path (str, optional): HDFS path (without namenode)
        
        Returns:
            str: Resolved path
        """
        return self.resolve_input_path(local_path, hdfs_path)
    
    def get_raw_data_path(self, filename: str = "financial_data.csv") -> str:
        """Get path to raw FOREX data."""
        if self.use_hdfs:
            return self.get_hdfs_path(f"{self.HDFS_RAW}/{filename}")
        else:
            return str(self.local_root / 'data' / filename)
    
    def get_batch_output_path(self, subset: str = "train") -> str:
        """
        Get path for batch preprocessing outputs.
        
        Args:
            subset (str): 'train', 'val', or 'test'
        
        Returns:
            str: Path to Parquet output directory
        """
        if self.use_hdfs:
            return self.get_hdfs_path(f"{self.HDFS_BATCH_PROCESSED}/{subset}")
        else:
            return str(self.local_root / 'data' / 'spark_processed' / subset)
    
    def get_streaming_output_path(self) -> str:
        """Get path for streaming Parquet outputs."""
        if self.use_hdfs:
            return self.get_hdfs_path(f"{self.HDFS_STREAMING}/output")
        else:
            return str(self.local_root / 'data' / 'spark_streaming' / 'forex_stream.parquet')
    
    def get_streaming_input_path(self) -> str:
        """Get path for streaming input files (watch directory)."""
        if self.use_hdfs:
            return self.get_hdfs_path(f"{self.HDFS_STREAMING}/input")
        else:
            return str(self.local_root / 'data' / 'spark_streaming_input')
    
    def get_checkpoint_path(self, checkpoint_type: str = "streaming") -> str:
        """
        Get path for Spark checkpoints.
        
        Args:
            checkpoint_type (str): 'batch' or 'streaming'
        
        Returns:
            str: Checkpoint directory path
        """
        if self.use_hdfs:
            return self.get_hdfs_path(f"{self.HDFS_CHECKPOINTS}/{checkpoint_type}")
        else:
            return str(self.local_root / 'checkpoints' / checkpoint_type)
    
    def print_configuration(self):
        """Print current configuration for debugging."""
        print("\n" + "=" * 80)
        print("HDFS CONFIGURATION")
        print("=" * 80)
        print(f"Mode: {'HDFS Distributed' if self.use_hdfs else 'Local Filesystem'}")
        print(f"HDFS Host: {self.hdfs_host if self.use_hdfs else 'N/A'}")
        print(f"Local Root: {self.local_root}")
        print("\nPath Mappings:")
        print(f"  Raw Data:        {self.get_raw_data_path()}")
        print(f"  Batch Train:     {self.get_batch_output_path('train')}")
        print(f"  Batch Val:       {self.get_batch_output_path('val')}")
        print(f"  Batch Test:      {self.get_batch_output_path('test')}")
        print(f"  Streaming Out:   {self.get_streaming_output_path()}")
        print(f"  Streaming In:    {self.get_streaming_input_path()}")
        print(f"  Checkpoint:      {self.get_checkpoint_path('streaming')}")
        print("=" * 80 + "\n")


def configure_spark_for_hdfs(spark_builder, hdfs_config: HDFSConfig):
    """
    Configure SparkSession builder with HDFS-aware settings.
    
    Args:
        spark_builder: SparkSession.builder object
        hdfs_config (HDFSConfig): HDFS configuration instance
    
    Returns:
        SparkSession.builder: Configured builder
    """
    if hdfs_config.use_hdfs:
        # Extract host and port from HDFS URL
        hdfs_url = hdfs_config.hdfs_host.replace('hdfs://', '')
        
        spark_builder = spark_builder \
            .config("spark.hadoop.fs.defaultFS", hdfs_config.hdfs_host) \
            .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
            .config("spark.hadoop.dfs.replication", "1")  # For pseudo-distributed
        
        print(f"[OK] Spark configured for HDFS: {hdfs_config.hdfs_host}")
    else:
        print("[OK] Spark configured for local filesystem")
    
    return spark_builder


def check_hdfs_availability() -> bool:
    """
    Check if HDFS is available and accessible.
    
    Returns:
        bool: True if HDFS is accessible, False otherwise
    """
    try:
        import subprocess
        result = subprocess.run(
            ['hdfs', 'dfs', '-ls', '/'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    """Demonstration of HDFS configuration utilities."""
    print("\n" + "=" * 80)
    print("HDFS CONFIGURATION UTILITIES DEMONSTRATION")
    print("=" * 80)
    
    # Test with local mode
    print("\n[1] Testing LOCAL mode...")
    os.environ['USE_HDFS'] = 'false'
    config_local = HDFSConfig()
    config_local.print_configuration()
    
    # Test with HDFS mode
    print("\n[2] Testing HDFS mode...")
    os.environ['USE_HDFS'] = 'true'
    os.environ['HDFS_HOST'] = 'hdfs://localhost:9000'
    config_hdfs = HDFSConfig()
    config_hdfs.print_configuration()
    
    # Check HDFS availability
    print("\n[3] Checking HDFS availability...")
    hdfs_available = check_hdfs_availability()
    if hdfs_available:
        print("[OK] HDFS is accessible")
    else:
        print("[WARNING] HDFS is not accessible")
        print("  Make sure Hadoop services are running:")
        print("    $ start-dfs.sh")
        print("    $ start-yarn.sh")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nEnvironment Variables:")
    print("  USE_HDFS=true|false    # Enable/disable HDFS mode")
    print("  HDFS_HOST=hdfs://...   # HDFS namenode URL")
    print("\nExample Usage in Code:")
    print("  from src.spark.hdfs_config import HDFSConfig")
    print("  config = HDFSConfig()")
    print("  input_path = config.get_raw_data_path()")
    print("  output_path = config.get_batch_output_path('train')")
    print()


if __name__ == '__main__':
    main()

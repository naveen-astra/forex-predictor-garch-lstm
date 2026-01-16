"""
Spark Module for Big Data Processing

This module contains Apache Spark implementations for distributed
data processing in the FOREX GARCH-LSTM project.

Modules:
- spark_batch_preprocessing: Batch processing with Spark DataFrames
- spark_streaming_forex: Real-time streaming with Structured Streaming
"""

from .spark_batch_preprocessing import SparkForexPreprocessor
from .spark_streaming_forex import ForexStreamingProcessor

__all__ = ['SparkForexPreprocessor', 'ForexStreamingProcessor']

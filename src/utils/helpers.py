"""
Helper utility functions for the FOREX GARCH-LSTM project.

Author: Research Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path


def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (Path or str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def format_currency(value, decimals=4):
    """
    Format currency value for display.
    
    Args:
        value (float): Currency value
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted currency string
    """
    return f"{value:.{decimals}f}"


def calculate_returns(prices, log=True):
    """
    Calculate returns from price series.
    
    Args:
        prices (pd.Series): Price series
        log (bool): If True, calculate log returns; else simple returns
        
    Returns:
        pd.Series: Returns series
    """
    if log:
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def create_lagged_features(df, columns, lags):
    """
    Create lagged features for time series.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Columns to lag
        lags (int): Number of lags
        
    Returns:
        pd.DataFrame: DataFrame with lagged features
    """
    df_lagged = df.copy()
    
    for col in columns:
        for lag in range(1, lags + 1):
            df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df_lagged


def get_date_range_summary(df):
    """
    Get summary of date range in dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index
        
    Returns:
        dict: Date range summary
    """
    return {
        'start_date': df.index.min(),
        'end_date': df.index.max(),
        'num_days': len(df),
        'date_range_str': f"{df.index.min().date()} to {df.index.max().date()}"
    }


def print_section_header(title, width=80):
    """
    Print formatted section header.
    
    Args:
        title (str): Section title
        width (int): Header width
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def save_results(data, filename, output_dir, formats=['csv', 'parquet']):
    """
    Save results in multiple formats.
    
    Args:
        data (pd.DataFrame): Data to save
        filename (str): Base filename (without extension)
        output_dir (Path): Output directory
        formats (list): List of formats to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    if 'csv' in formats:
        csv_path = output_dir / f"{filename}.csv"
        data.to_csv(csv_path)
        saved_files.append(csv_path)
    
    if 'parquet' in formats:
        parquet_path = output_dir / f"{filename}.parquet"
        data.to_parquet(parquet_path)
        saved_files.append(parquet_path)
    
    return saved_files

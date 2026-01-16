"""
Data Fetching Module for FOREX Historical Data

This module provides functions to download historical FOREX exchange rate data
from multiple sources (Yahoo Finance, Alpha Vantage) with robust error handling
and data validation.

Author: Research Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
from datetime import datetime, timedelta
import time
import warnings
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import (
    DATA_CONFIG, RAW_DATA_DIR, RANDOM_SEED, set_random_seeds
)

warnings.filterwarnings('ignore')


class ForexDataFetcher:
    """
    Fetch historical FOREX data from multiple sources.
    
    Features:
    - Multiple data sources with fallback mechanism
    - Data validation and quality checks
    - Automatic retry on failure
    - Save data in multiple formats (CSV, Parquet)
    """
    
    def __init__(self, currency_pair='EURUSD=X', start_date='2010-01-01', 
                 end_date='2025-12-31', source='yfinance'):
        """
        Initialize the data fetcher.
        
        Args:
            currency_pair (str): Currency pair ticker (Yahoo Finance format)
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            source (str): Data source ('yfinance' or 'alpha_vantage')
        """
        self.currency_pair = currency_pair
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.data = None
        
        print(f"Initialized ForexDataFetcher for {currency_pair}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Data source: {source}")
    
    def fetch_from_yfinance(self):
        """
        Fetch data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: FOREX data with OHLC columns
        """
        print(f"\n{'='*60}")
        print("Fetching data from Yahoo Finance...")
        print(f"{'='*60}")
        
        try:
            # Download data using yfinance
            ticker = yf.Ticker(self.currency_pair)
            df = ticker.history(start=self.start_date, end=self.end_date, interval='1d')
            
            if df.empty:
                raise ValueError("No data returned from Yahoo Finance")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Keep relevant columns
            columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
            
            # Convert timezone-aware datetime to timezone-naive
            if df['Datetime'].dt.tz is not None:
                df['Datetime'] = df['Datetime'].dt.tz_localize(None)
            
            # Ensure datetime format
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            print(f"✓ Successfully fetched {len(df)} records from Yahoo Finance")
            print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
            print(f"  Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data from Yahoo Finance: {str(e)}")
            return None
    
    def fetch_from_alpha_vantage(self, api_key=None):
        """
        Fetch data from Alpha Vantage API.
        
        Args:
            api_key (str): Alpha Vantage API key
            
        Returns:
            pd.DataFrame: FOREX data with OHLC columns
        """
        print(f"\n{'='*60}")
        print("Fetching data from Alpha Vantage...")
        print(f"{'='*60}")
        
        if api_key is None or api_key == 'YOUR_API_KEY_HERE':
            print("✗ Alpha Vantage API key not provided")
            print("  Get a free API key at: https://www.alphavantage.co/support/#api-key")
            return None
        
        try:
            # Initialize Alpha Vantage client
            fx = ForeignExchange(key=api_key, output_format='pandas')
            
            # Get currency codes from pair
            from_currency = DATA_CONFIG['alpha_vantage_from_currency']
            to_currency = DATA_CONFIG['alpha_vantage_to_currency']
            
            # Fetch daily data (note: Alpha Vantage has rate limits)
            data, meta_data = fx.get_currency_exchange_daily(
                from_symbol=from_currency,
                to_symbol=to_currency,
                outputsize='full'  # Get full historical data
            )
            
            if data.empty:
                raise ValueError("No data returned from Alpha Vantage")
            
            # Reset index and rename columns
            df = data.reset_index()
            df.rename(columns={
                'date': 'Datetime',
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close'
            }, inplace=True)
            
            # Convert to datetime
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            # Filter by date range
            df = df[(df['Datetime'] >= self.start_date) & (df['Datetime'] <= self.end_date)]
            
            # Sort by date
            df.sort_values('Datetime', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            # Add Volume column (FOREX doesn't have volume, set to NaN)
            df['Volume'] = np.nan
            
            print(f"✓ Successfully fetched {len(df)} records from Alpha Vantage")
            print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
            print(f"  Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data from Alpha Vantage: {str(e)}")
            return None
    
    def fetch_data(self, api_key=None):
        """
        Fetch data with automatic fallback between sources.
        
        Args:
            api_key (str): Alpha Vantage API key (required if using that source)
            
        Returns:
            pd.DataFrame: FOREX data
        """
        # Try primary source
        if self.source == 'yfinance':
            df = self.fetch_from_yfinance()
            if df is not None:
                self.data = df
                return df
            # Fallback to Alpha Vantage
            print("\nFalling back to Alpha Vantage...")
            df = self.fetch_from_alpha_vantage(api_key)
            
        elif self.source == 'alpha_vantage':
            df = self.fetch_from_alpha_vantage(api_key)
            if df is not None:
                self.data = df
                return df
            # Fallback to Yahoo Finance
            print("\nFalling back to Yahoo Finance...")
            df = self.fetch_from_yfinance()
        
        if df is None:
            raise RuntimeError("Failed to fetch data from all sources")
        
        self.data = df
        return df
    
    def validate_data(self):
        """
        Validate fetched data for quality issues.
        
        Returns:
            dict: Validation report
        """
        if self.data is None:
            raise ValueError("No data to validate. Call fetch_data() first.")
        
        print(f"\n{'='*60}")
        print("Validating data quality...")
        print(f"{'='*60}")
        
        report = {}
        df = self.data
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        report['missing_values'] = missing_counts.to_dict()
        print(f"\nMissing values:")
        for col, count in missing_counts.items():
            if count > 0:
                pct = (count / len(df)) * 100
                print(f"  {col}: {count} ({pct:.2f}%)")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['Datetime']).sum()
        report['duplicate_dates'] = duplicates
        print(f"\nDuplicate dates: {duplicates}")
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        negative_prices = {}
        for col in price_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                negative_prices[col] = neg_count
                if neg_count > 0:
                    print(f"  Warning: {neg_count} negative values in {col}")
        report['negative_prices'] = negative_prices
        
        # Check for zero prices
        zero_prices = {}
        for col in price_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                zero_prices[col] = zero_count
                if zero_count > 0:
                    print(f"  Warning: {zero_count} zero values in {col}")
        report['zero_prices'] = zero_prices
        
        # Check date continuity (business days)
        date_gaps = []
        dates = pd.to_datetime(df['Datetime']).sort_values()
        for i in range(1, len(dates)):
            delta = (dates.iloc[i] - dates.iloc[i-1]).days
            if delta > 7:  # More than a week gap
                date_gaps.append({
                    'from': dates.iloc[i-1],
                    'to': dates.iloc[i],
                    'days': delta
                })
        report['large_date_gaps'] = len(date_gaps)
        if date_gaps:
            print(f"\nLarge date gaps (>7 days): {len(date_gaps)}")
            for gap in date_gaps[:5]:  # Show first 5
                print(f"  {gap['from'].date()} to {gap['to'].date()} ({gap['days']} days)")
        
        # Summary statistics
        print(f"\nData summary:")
        print(f"  Total records: {len(df)}")
        print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        print(f"  Completeness: {((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100):.2f}%")
        
        print(f"\n✓ Validation complete")
        
        return report
    
    def save_data(self, filename=None, format='both'):
        """
        Save fetched data to disk.
        
        Args:
            filename (str): Base filename (without extension)
            format (str): 'csv', 'parquet', or 'both'
        """
        if self.data is None:
            raise ValueError("No data to save. Call fetch_data() first.")
        
        if filename is None:
            # Generate filename with currency pair and date
            currency_name = DATA_CONFIG['currency_pair_name']
            today = datetime.now().strftime('%Y%m%d')
            filename = f"{currency_name}_raw_{today}"
        
        print(f"\n{'='*60}")
        print("Saving data to disk...")
        print(f"{'='*60}")
        
        # Save as CSV
        if format in ['csv', 'both']:
            csv_path = RAW_DATA_DIR / f"{filename}.csv"
            self.data.to_csv(csv_path, index=False)
            print(f"✓ Saved CSV: {csv_path}")
            print(f"  Size: {csv_path.stat().st_size / 1024:.2f} KB")
        
        # Save as Parquet (more efficient for large datasets)
        if format in ['parquet', 'both']:
            parquet_path = RAW_DATA_DIR / f"{filename}.parquet"
            self.data.to_parquet(parquet_path, index=False)
            print(f"✓ Saved Parquet: {parquet_path}")
            print(f"  Size: {parquet_path.stat().st_size / 1024:.2f} KB")
        
        # Save metadata
        metadata = {
            'currency_pair': self.currency_pair,
            'source': self.source,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_records': len(self.data),
            'columns': list(self.data.columns)
        }
        
        metadata_path = RAW_DATA_DIR / f"{filename}_metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write("FOREX Data Metadata\n")
            f.write("="*60 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"✓ Saved metadata: {metadata_path}")


def main():
    """
    Main function to fetch FOREX data.
    """
    # Set random seeds for reproducibility
    set_random_seeds(RANDOM_SEED)
    
    print("\n" + "="*80)
    print("FOREX DATA ACQUISITION MODULE")
    print("="*80)
    
    # Initialize fetcher with config parameters
    fetcher = ForexDataFetcher(
        currency_pair=DATA_CONFIG['currency_pair'],
        start_date=DATA_CONFIG['start_date'],
        end_date=DATA_CONFIG['end_date'],
        source='yfinance'  # Primary source
    )
    
    # Fetch data
    try:
        df = fetcher.fetch_data(api_key=DATA_CONFIG.get('alpha_vantage_api_key'))
        
        # Validate data
        validation_report = fetcher.validate_data()
        
        # Display sample
        print(f"\n{'='*60}")
        print("Sample of fetched data (first 5 rows):")
        print(f"{'='*60}")
        print(df.head())
        
        print(f"\n{'='*60}")
        print("Sample of fetched data (last 5 rows):")
        print(f"{'='*60}")
        print(df.tail())
        
        # Save data
        fetcher.save_data(format='both')
        
        print("\n" + "="*80)
        print("✓ Data acquisition completed successfully!")
        print("="*80)
        
        return df
        
    except Exception as e:
        print(f"\n✗ Error in data acquisition: {str(e)}")
        raise


if __name__ == "__main__":
    # Run data fetching
    data = main()

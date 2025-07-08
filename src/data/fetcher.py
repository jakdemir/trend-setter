"""
📊 Data Fetcher - Elegant Weekly Data Loading
═════════════════════════════════════════════

Handles fetching and preprocessing of weekly OHLCV data 
with proper error handling and data validation.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


class DataFetcher:
    """
    🎯 Fetches and preprocesses weekly financial data with style
    """
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache for session
    
    def get_weekly_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        📈 Fetch weekly OHLCV data for a ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with weekly OHLCV data
        """
        cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}"
        
        if cache_key in self.cache:
            print(f"📦 Using cached data for {ticker}")
            return self.cache[cache_key].copy()
        
        try:
            # Download daily data first
            stock = yf.Ticker(ticker)
            daily_data = stock.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                prepost=True
            )
            
            if daily_data.empty:
                print(f"⚠️ No data found for {ticker}")
                return pd.DataFrame()
            
            # Convert to weekly data
            weekly_data = self._convert_to_weekly(daily_data)
            
            # Clean and validate
            weekly_data = self._clean_data(weekly_data)
            
            # Cache the result
            self.cache[cache_key] = weekly_data.copy()
            
            return weekly_data
            
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def _convert_to_weekly(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        📅 Convert daily data to weekly (Friday close or last trading day of week)
        """
        # Ensure we have the right columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in daily_data.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {daily_data.columns.tolist()}")
        
        # Resample to weekly data (ending on Friday)
        weekly = daily_data.resample('W-FRI').agg({
            'Open': 'first',    # Monday open (or first trading day)
            'High': 'max',      # Week's highest high
            'Low': 'min',       # Week's lowest low  
            'Close': 'last',    # Friday close (or last trading day)
            'Volume': 'sum'     # Total weekly volume
        }).dropna()
        
        # Rename columns to lowercase for consistency
        weekly.columns = ['open', 'high', 'low', 'close', 'volume']
        
        return weekly
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🧹 Clean and validate the data
        """
        if df.empty:
            return df
        
        # Remove any rows with missing data
        df = df.dropna()
        
        # Ensure positive prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df = df[df[col] > 0]
        
        # Ensure volume is non-negative
        df = df[df['volume'] >= 0]
        
        # Sort by date
        df = df.sort_index()
        
        # Add some basic derived fields
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['weekly_range'] = df['high'] - df['low']
        df['weekly_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        
        return df
    
    def get_sp500_tickers(self, sample_size: Optional[int] = None) -> list:
        """
        📋 Get list of S&P 500 tickers
        
        For now, returns a curated list of major stocks.
        In a real implementation, you'd fetch from a reliable source.
        """
        # Major S&P 500 stocks for testing
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
            'NFLX', 'ADBE', 'CRM', 'CMCSA', 'XOM', 'VZ', 'KO', 'ABT', 'PFE',
            'T', 'PEP', 'INTC', 'WMT', 'CSCO', 'CVX', 'MRK', 'TMO', 'DHR',
            'COST', 'AVGO', 'ACN', 'TXN', 'NEE', 'LLY', 'NKE', 'LIN', 'QCOM'
        ]
        
        if sample_size:
            return tickers[:sample_size]
        
        return tickers 
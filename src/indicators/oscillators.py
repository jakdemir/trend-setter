"""
🌊 Oscillators - RSI and Momentum Indicators
═══════════════════════════════════════════

Beautiful implementations of momentum oscillators for trend detection.
"""

import pandas as pd
import numpy as np
from typing import Optional


class RSICalculator:
    """
    🎯 Relative Strength Index - Momentum oscillator (0-100)
    
    Classic RSI with elegant implementation and proper handling
    of edge cases for weekly data.
    """
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Calculate RSI for the given dataframe
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column to use for RSI calculation
            
        Returns:
            DataFrame with RSI column added
        """
        df = df.copy()
        
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in DataFrame")
        
        # Calculate price changes
        delta = df[price_col].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate rolling averages (use SMA for initial, then EMA)
        avg_gains = gains.rolling(window=self.period, min_periods=self.period).mean()
        avg_losses = losses.rolling(window=self.period, min_periods=self.period).mean()
        
        # For subsequent periods, use EMA-style calculation
        for i in range(self.period, len(df)):
            avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (self.period - 1) + gains.iloc[i]) / self.period
            avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (self.period - 1) + losses.iloc[i]) / self.period
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50)  # Neutral RSI for NaN values
        
        df[f'rsi_{self.period}'] = rsi
        
        # Add RSI signal interpretation
        df[f'rsi_{self.period}_signal'] = self._interpret_rsi(rsi)
        
        return df
    
    def _interpret_rsi(self, rsi: pd.Series) -> pd.Series:
        """
        🔍 Interpret RSI values into actionable signals
        """
        conditions = [
            rsi >= 70,  # Overbought
            rsi <= 30,  # Oversold
            (rsi > 50) & (rsi < 70),  # Bullish
            (rsi < 50) & (rsi > 30),  # Bearish
        ]
        
        choices = ['overbought', 'oversold', 'bullish', 'bearish']
        
        return pd.Series(
            np.select(conditions, choices, default='neutral'),
            index=rsi.index
        )


class StochasticCalculator:
    """
    📊 Stochastic Oscillator (%K and %D)
    
    Another momentum oscillator that compares closing price 
    to the price range over a given period.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator
        """
        df = df.copy()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Calculate %K
        lowest_low = df['low'].rolling(window=self.k_period).min()
        highest_high = df['high'].rolling(window=self.k_period).max()
        
        k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (moving average of %K)
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        df[f'stoch_k_{self.k_period}'] = k_percent
        df[f'stoch_d_{self.d_period}'] = d_percent
        
        # Add signal interpretation
        df['stoch_signal'] = self._interpret_stochastic(k_percent, d_percent)
        
        return df
    
    def _interpret_stochastic(self, k_percent: pd.Series, d_percent: pd.Series) -> pd.Series:
        """
        🔍 Interpret Stochastic signals
        """
        conditions = [
            (k_percent >= 80) & (d_percent >= 80),  # Overbought
            (k_percent <= 20) & (d_percent <= 20),  # Oversold
            (k_percent > d_percent) & (k_percent > 50),  # Bullish
            (k_percent < d_percent) & (k_percent < 50),  # Bearish
        ]
        
        choices = ['overbought', 'oversold', 'bullish', 'bearish']
        
        return pd.Series(
            np.select(conditions, choices, default='neutral'),
            index=k_percent.index
        ) 
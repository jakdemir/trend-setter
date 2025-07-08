"""
📈 Trend Indicators - MACD, EMA, and Trend Following
"""

import pandas as pd
import numpy as np
from typing import List


class MACDCalculator:
    """🌊 MACD - Moving Average Convergence Divergence"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow  
        self.signal = signal
    
    def calculate(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate MACD, Signal line, and Histogram"""
        df = df.copy()
        
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in DataFrame")
        
        # Calculate EMAs
        ema_fast = df[price_col].ewm(span=self.fast).mean()
        ema_slow = df[price_col].ewm(span=self.slow).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        signal_line = macd_line.ewm(span=self.signal).mean()
        
        # Histogram (MACD - Signal)
        histogram = macd_line - signal_line
        
        # Add to dataframe
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        return df


class EMACalculator:
    """📊 Exponential Moving Average Calculator"""
    
    def calculate(self, df: pd.DataFrame, periods: List[int] = [10, 30], price_col: str = 'close') -> pd.DataFrame:
        """Calculate multiple EMAs"""
        df = df.copy()
        
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in DataFrame")
        
        # Calculate EMAs for each period
        for period in periods:
            ema_col = f'ema_{period}'
            df[ema_col] = df[price_col].ewm(span=period).mean()
        
        return df


class ADXCalculator:
    """
    💪 Average Directional Index - Trend Strength Indicator
    
    Measures the strength of a trend without regard to direction.
    Values above 25 typically indicate a strong trend.
    """
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX, +DI, and -DI
        """
        df = df.copy()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # True Range calculation
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Directional Movement
        high_diff = df['high'] - df['high'].shift(1)
        low_diff = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Smoothed values
        tr_smooth = pd.Series(true_range).rolling(window=self.period).mean()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=self.period).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=self.period).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # ADX calculation
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.period).mean()
        
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Trend strength interpretation
        df['trend_strength'] = self._interpret_adx(adx, plus_di, minus_di)
        
        return df
    
    def _interpret_adx(self, adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
        """
        🔍 Interpret ADX trend strength
        """
        conditions = [
            (adx >= 50) & (plus_di > minus_di),  # Very strong uptrend
            (adx >= 25) & (adx < 50) & (plus_di > minus_di),  # Strong uptrend
            (adx >= 50) & (plus_di < minus_di),  # Very strong downtrend
            (adx >= 25) & (adx < 50) & (plus_di < minus_di),  # Strong downtrend
            adx < 25,  # Weak trend
        ]
        
        choices = ['very_strong_up', 'strong_up', 'very_strong_down', 'strong_down', 'weak_trend']
        
        return pd.Series(
            np.select(conditions, choices, default='neutral'),
            index=adx.index
        ) 
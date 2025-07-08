"""
🎯 Weekly Trend Scorer - The Heart of Trend Detection
═════════════════════════════════════════════════════

Combines multiple indicators into a unified trend score (0-100)
and detects trend regime changes with elegant logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class WeeklyTrendScorer:
    """
    🧠 Intelligent trend scoring that combines multiple indicators
    
    Creates a unified trend score (0-100) and classifies market regimes:
    - 0-30: Strong Bearish
    - 30-40: Bearish  
    - 40-60: Neutral
    - 60-70: Bullish
    - 70-100: Strong Bullish
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Default weights for different indicator types
        self.weights = weights or {
            'rsi': 0.25,      # RSI momentum
            'macd': 0.30,     # MACD trend following
            'ema': 0.25,      # EMA trend direction
            'price_action': 0.20  # Raw price momentum
        }
    
    def calculate_trend_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎯 Calculate the unified weekly trend score
        """
        df = df.copy()
        
        # Individual component scores
        rsi_score = self._calculate_rsi_score(df)
        macd_score = self._calculate_macd_score(df)
        ema_score = self._calculate_ema_score(df)
        price_score = self._calculate_price_action_score(df)
        
        # Weighted combination
        trend_score = (
            rsi_score * self.weights['rsi'] +
            macd_score * self.weights['macd'] +
            ema_score * self.weights['ema'] +
            price_score * self.weights['price_action']
        )
        
        # Normalize to 0-100 range and smooth
        trend_score = np.clip(trend_score, 0, 100)
        trend_score = trend_score.rolling(window=2, min_periods=1).mean()  # Light smoothing
        
        df['trend_score'] = trend_score
        df['trend_regime'] = self._classify_regime(trend_score)
        
        return df
    
    def _calculate_rsi_score(self, df: pd.DataFrame) -> pd.Series:
        """
        📊 Convert RSI to 0-100 trend score
        """
        if 'rsi_14' not in df.columns:
            return pd.Series(50, index=df.index)  # Neutral if no RSI
        
        rsi = df['rsi_14']
        
        # Transform RSI to trend score
        # RSI 30-70 maps to score 20-80, with extensions for extremes
        conditions = [
            rsi <= 20,    # Severely oversold -> Score boost
            rsi <= 30,    # Oversold
            rsi >= 80,    # Severely overbought -> Score reduction
            rsi >= 70,    # Overbought
        ]
        
        choices = [
            np.minimum(rsi * 1.5, 40),     # Boost oversold
            rsi * 0.8 + 16,                # Mild boost for oversold
            np.maximum(rsi * 0.6 + 32, 60), # Reduce overbought  
            rsi * 0.9 + 7,                 # Mild reduction for overbought
        ]
        
        return pd.Series(
            np.select(conditions, choices, default=rsi),
            index=df.index
        )
    
    def _calculate_macd_score(self, df: pd.DataFrame) -> pd.Series:
        """
        🌊 Convert MACD signals to trend score
        """
        if not all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            return pd.Series(50, index=df.index)  # Neutral if no MACD
        
        macd = df['macd']
        signal = df['macd_signal']
        histogram = df['macd_histogram']
        
        # Base score from MACD position relative to signal
        macd_above_signal = macd > signal
        macd_above_zero = macd > 0
        histogram_increasing = histogram > histogram.shift(1)
        
        # Calculate score components
        base_score = np.where(macd_above_signal, 60, 40)  # Above/below signal line
        zero_line_boost = np.where(macd_above_zero, 10, -10)  # Above/below zero
        momentum_boost = np.where(histogram_increasing, 5, -5)  # Histogram direction
        
        score = base_score + zero_line_boost + momentum_boost
        
        return pd.Series(np.clip(score, 0, 100), index=df.index)
    
    def _calculate_ema_score(self, df: pd.DataFrame) -> pd.Series:
        """
        📈 Convert EMA relationship to trend score
        """
        if not all(col in df.columns for col in ['close', 'ema_10', 'ema_30']):
            return pd.Series(50, index=df.index)  # Neutral if no EMAs
        
        price = df['close']
        ema_10 = df['ema_10']
        ema_30 = df['ema_30']
        
        # Price relative to EMAs
        price_above_10 = price > ema_10
        price_above_30 = price > ema_30
        ema_10_above_30 = ema_10 > ema_30
        
        # EMA slope (trend direction)
        ema_10_rising = ema_10 > ema_10.shift(1)
        ema_30_rising = ema_30 > ema_30.shift(1)
        
        # Build score
        conditions = [
            price_above_10 & price_above_30 & ema_10_above_30 & ema_10_rising & ema_30_rising,  # 85
            price_above_10 & ema_10_above_30 & ema_10_rising,  # 70
            price_above_10 & ema_10_above_30,  # 60
            ema_10_above_30,  # 55
            ~price_above_10 & ~ema_10_above_30 & ~ema_10_rising & ~ema_30_rising,  # 15
            ~price_above_10 & ~ema_10_above_30 & ~ema_10_rising,  # 30
            ~price_above_10 & ~ema_10_above_30,  # 40
            ~ema_10_above_30,  # 45
        ]
        
        choices = [85, 70, 60, 55, 15, 30, 40, 45]
        
        return pd.Series(
            np.select(conditions, choices, default=50),
            index=df.index
        )
    
    def _calculate_price_action_score(self, df: pd.DataFrame) -> pd.Series:
        """
        💹 Raw price momentum and volatility score
        """
        if 'close' not in df.columns:
            return pd.Series(50, index=df.index)
        
        price = df['close']
        
        # Weekly returns
        weekly_return = price.pct_change()
        
        # Multi-period momentum
        momentum_2w = price.pct_change(2)
        momentum_4w = price.pct_change(4)
        
        # Volatility (rolling standard deviation of returns)
        volatility = weekly_return.rolling(window=8).std()
        
        # Score based on momentum
        momentum_score = np.where(
            momentum_4w > 0.02,  # +2% over 4 weeks
            70 + np.clip(momentum_4w * 500, 0, 25),  # Boost for strong momentum
            np.where(
                momentum_4w < -0.02,  # -2% over 4 weeks
                30 + np.clip(momentum_4w * 500, -25, 0),  # Penalty for weak momentum
                50 + momentum_4w * 250  # Neutral zone
            )
        )
        
        # Adjust for volatility (high vol = uncertainty)
        vol_adjustment = np.where(
            volatility > volatility.rolling(window=12).mean() * 1.5,
            -5,  # Penalty for high volatility
            0
        )
        
        score = momentum_score + vol_adjustment
        
        return pd.Series(np.clip(score, 0, 100), index=df.index)
    
    def _classify_regime(self, trend_score: pd.Series) -> pd.Series:
        """
        🏷️ Classify trend regime based on score
        """
        conditions = [
            trend_score >= 75,
            trend_score >= 60,
            trend_score >= 40,
            trend_score >= 25,
        ]
        
        choices = ['strong_bullish', 'bullish', 'neutral', 'bearish']
        
        return pd.Series(
            np.select(conditions, choices, default='strong_bearish'),
            index=trend_score.index
        )
    
    def detect_trend_changes(self, df: pd.DataFrame, min_change_threshold: int = 15) -> pd.DataFrame:
        """
        🚨 Detect significant trend regime changes
        """
        if 'trend_regime' not in df.columns or 'trend_score' not in df.columns:
            return pd.DataFrame()
        
        # Find regime changes
        df_copy = df.copy()
        df_copy['prev_regime'] = df_copy['trend_regime'].shift(1)
        df_copy['regime_changed'] = df_copy['trend_regime'] != df_copy['prev_regime']
        
        # Also detect significant score changes within same regime
        df_copy['score_change'] = df_copy['trend_score'].diff().abs()
        df_copy['significant_change'] = df_copy['score_change'] >= min_change_threshold
        
        # Filter for meaningful changes
        changes = df_copy[
            df_copy['regime_changed'] | 
            (df_copy['significant_change'] & (df_copy['score_change'] >= min_change_threshold))
        ].copy()
        
        if changes.empty:
            return pd.DataFrame()
        
        # Add change metadata
        changes['change_type'] = np.where(
            changes['regime_changed'], 
            'regime_change', 
            'score_shift'
        )
        
        changes['date'] = changes.index
        
        return changes[['date', 'trend_regime', 'prev_regime', 'trend_score', 'change_type']].reset_index(drop=True) 
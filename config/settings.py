"""
⚙️ Configuration Settings for Trend Setter
═══════════════════════════════════════════

Central configuration for the weekly trend analyzer.
"""

from datetime import timedelta

# Data Settings
DATA_LOOKBACK_DAYS = 730  # 2 years of history
CACHE_ENABLED = True

# Indicator Settings
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_PERIODS = [10, 30]

# Trend Scoring Weights
TREND_WEIGHTS = {
    'rsi': 0.25,
    'macd': 0.30,
    'ema': 0.25,
    'price_action': 0.20
}

# Trend Regime Thresholds
REGIME_THRESHOLDS = {
    'strong_bearish': 25,
    'bearish': 40,
    'neutral': 60,
    'bullish': 75
}

# Chart Settings
CHART_STYLE = 'seaborn-v0_8-darkgrid'
CHART_DPI = 300
CHART_FIGSIZE = (15, 12)

# Export Settings
EXPORT_FORMAT = 'csv'
OUTPUTS_DIR = 'outputs'
DATA_DIR = 'data' 
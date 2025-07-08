# 🎯 Trend Setter - SPX Trend Analysis System

**Clean, precise trend analysis for the S&P 500 with high-precision reversal detection.**

## 🚀 Quick Start

```bash
# Run SPX historical analysis with visual arrows (35+ years)
python verify_spx_analysis.py

# Quick historical events summary
python historical_summary.py

# Run quick SPX summary
python multi_stock_analyzer.py

# Quick 6-month demo
python demo.py
```

## 📊 What It Does

**Trend Setter** analyzes **35+ years of S&P 500 history** and identifies **major trend reversals** with surgical precision:

- **🟢 Bullish Arrows**: Significant market bottoms with confirmed recoveries (5%+ price moves)
- **🔴 Bearish Arrows**: Significant market tops with confirmed declines (5%+ price moves)
- **No Noise**: Only shows meaningful directional changes, minimum 8 weeks apart

## 🎯 Key Features

### ✨ High-Precision Detection
- Uses rolling window peak/trough detection
- Confirms reversals with 5%+ price moves
- Validates with trend score changes (10+ point shifts)
- Prevents consecutive same-direction arrows

### 📈 Clean Visualization
- Single-panel SPX price chart
- Large, clear directional arrows at exact reversal points
- Professional styling with currency formatting
- Focus on actionable signals only

### 🔧 Technical Indicators
- **RSI (14)**: Momentum oscillator
- **MACD (12,26,9)**: Trend following
- **EMA (10,30)**: Moving averages
- **Trend Score**: Proprietary 0-100 scoring system

## 📁 Output Files

```
outputs/
├── spx_historical_trend_reversals.png    # 35+ year chart with arrows
├── spx_historical_data.csv               # Complete historical analysis data
└── spx_summary.csv                       # Current status summary
```

## 🎯 Trend Regimes

- **Strong Bullish** (80-100): 🟢🟢 Aggressive growth phase
- **Bullish** (60-79): 🟢 Healthy uptrend  
- **Neutral** (40-59): ⚪ Sideways/uncertain
- **Bearish** (21-39): 🔴 Declining trend
- **Strong Bearish** (0-20): 🔴🔴 Major correction

## 🔍 System Architecture

```
src/
├── data/fetcher.py           # yfinance data retrieval
├── indicators/
│   ├── oscillators.py        # RSI calculations
│   └── trend.py              # MACD, EMA calculations
└── scoring/trend_scorer.py   # Proprietary trend scoring
```

## 🎯 Use Cases

- **Market Timing**: Identify major S&P 500 turning points
- **Risk Management**: Spot potential trend reversals early
- **Strategic Planning**: Long-term trend analysis for portfolios
- **Education**: Learn to spot high-quality technical signals

## 🛠️ Requirements

```
pandas>=2.0.0
yfinance>=0.2.0
matplotlib>=3.7.0
ta>=0.11.0
```

## 📊 Historical Performance

**35+ Year Analysis (1990-2025)**:
- **Total Return**: +1,656% 
- **Annualized Return**: +8.4%
- **Current Status**: Strong Bullish regime (score: 76.9)
- **Trend Reversals Captured**: 60 major turning points
- **Major Cycles Covered**: Dot-com bubble, 9/11, Financial Crisis, COVID-19

---

*Built with ❤️ for precise, actionable market analysis.* 
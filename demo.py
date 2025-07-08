#!/usr/bin/env python3
"""
🎯 SPX Trend Analysis Demo
==========================

Quick demonstration of the refined SPX trend analysis system.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.fetcher import DataFetcher
from src.scoring.trend_scorer import WeeklyTrendScorer

def demo_spx_analysis():
    """
    🚀 Quick SPX analysis demo
    """
    print("🎯 SPX Trend Analysis Demo")
    print("=" * 35)
    
    # Initialize components
    fetcher = DataFetcher()
    scorer = WeeklyTrendScorer()
    
    # Get recent SPX data
    ticker = "^GSPC"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # ~6 months
    
    print(f"📊 Fetching {ticker} data...")
    df = fetcher.get_weekly_data(ticker, start_date, end_date)
    
    if df.empty:
        print("❌ No data available")
        return
    
    # Calculate trend score
    df = scorer.calculate_trend_score(df)
    
    # Current status
    latest = df.iloc[-1]
    print(f"\n📍 Current SPX Status:")
    print(f"   • Price: ${latest['close']:,.2f}")
    print(f"   • Trend Score: {latest['trend_score']:.1f}/100")
    print(f"   • Regime: {latest['trend_regime']}")
    
    # Regime emoji mapping
    regime_emoji = {
        'strong_bullish': '🟢🟢',
        'bullish': '🟢',
        'neutral': '⚪',
        'bearish': '🔴',
        'strong_bearish': '🔴🔴'
    }
    
    emoji = regime_emoji.get(latest['trend_regime'], '❓')
    print(f"   • Signal: {emoji} {latest['trend_regime'].replace('_', ' ').title()}")
    
    # Recent performance
    performance = ((latest['close'] - df.iloc[0]['close']) / df.iloc[0]['close']) * 100
    print(f"   • 6M Performance: {performance:+.1f}%")
    
    # Recent trend changes
    changes = scorer.detect_trend_changes(df)
    print(f"\n🚨 Recent Activity:")
    print(f"   • Trend changes (6M): {len(changes)}")
    
    if not changes.empty:
        latest_change = changes.iloc[-1]
        change_date = latest_change['date'].strftime('%Y-%m-%d') if hasattr(latest_change['date'], 'strftime') else str(latest_change['date'])[:10]
        print(f"   • Last change: {change_date}")
        print(f"   • Changed to: {latest_change['trend_regime']}")
    
    print(f"\n🎯 For detailed historical analysis with arrows:")
    print(f"   Run: python verify_spx_analysis.py")
    print(f"   Chart: outputs/spx_historical_trend_reversals.png")
    print(f"   🕰️ Covers 35+ years of market history!")

if __name__ == "__main__":
    demo_spx_analysis() 
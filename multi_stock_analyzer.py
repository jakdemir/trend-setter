#!/usr/bin/env python3
"""
🎯 Multi-Stock Trend Analyzer - Extensibility Demo
═══════════════════════════════════════════════

Demonstrates analyzing multiple S&P 500 stocks simultaneously
to show the modular power of our trend detection system.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.fetcher import DataFetcher
from src.indicators.oscillators import RSICalculator
from src.indicators.trend import MACDCalculator, EMACalculator
from src.scoring.trend_scorer import WeeklyTrendScorer


def analyze_stock(ticker: str, fetcher: DataFetcher, scorer: WeeklyTrendScorer) -> dict:
    """
    🔍 Analyze a single stock and return key metrics
    """
    try:
        # Load data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year for speed
        
        df = fetcher.get_weekly_data(ticker, start_date, end_date)
        
        if df.empty:
            return None
        
        # Calculate indicators
        rsi_calc = RSICalculator(period=14)
        df = rsi_calc.calculate(df)
        
        macd_calc = MACDCalculator(fast=12, slow=26, signal=9)
        df = macd_calc.calculate(df)
        
        ema_calc = EMACalculator()
        df = ema_calc.calculate(df, periods=[10, 30])
        
        # Calculate trend score
        df = scorer.calculate_trend_score(df)
        
        # Get current metrics
        latest = df.iloc[-1]
        
        # Count trend changes in last 6 months
        recent_changes = scorer.detect_trend_changes(df.tail(26))  # ~6 months
        
        return {
            'ticker': ticker,
            'current_price': latest['close'],
            'current_regime': latest['trend_regime'],
            'current_score': latest['trend_score'],
            'rsi': latest['rsi_14'],
            'weeks_analyzed': len(df),
            'recent_changes': len(recent_changes),
            'price_change_52w': ((latest['close'] - df.iloc[0]['close']) / df.iloc[0]['close']) * 100
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {str(e)}")
        return None


def main():
    """
    🚀 SPX trend analysis
    """
    print("🎯 SPX Trend Analyzer")
    print("=" * 25)
    
    # Initialize components
    fetcher = DataFetcher()
    scorer = WeeklyTrendScorer()
    
    # Focus only on SPX analysis
    tickers = ['^GSPC']
    
    print(f"📊 Analyzing SPX trend...")
    print()
    
    results = []
    
    for ticker in tickers:
        print(f"🔍 Analyzing {ticker}...", end=" ")
        result = analyze_stock(ticker, fetcher, scorer)
        
        if result:
            results.append(result)
            print("✅")
        else:
            print("❌")
    
    if not results:
        print("No successful analyses :(")
        return
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(results)
    
    # Sort by trend score (strongest trends first)
    df_summary = df_summary.sort_values('current_score', ascending=False)
    
    print(f"\n📈 SPX Trend Analysis Summary")
    print("=" * 80)
    
    # Display formatted table
    print(f"{'Ticker':<8} {'Price':<8} {'Score':<6} {'Regime':<15} {'RSI':<6} {'52W %':<8} {'Changes'}")
    print("-" * 80)
    
    for _, row in df_summary.iterrows():
        regime_emoji = {
            'strong_bullish': '🟢🟢',
            'bullish': '🟢',
            'neutral': '⚪',
            'bearish': '🔴', 
            'strong_bearish': '🔴🔴'
        }.get(row['current_regime'], '❓')
        
        # Highlight SPX (market index)
        ticker_display = f"📊 {row['ticker']}" if row['ticker'] == '^GSPC' else row['ticker']
        
        print(f"{ticker_display:<8} ${row['current_price']:<7.2f} "
              f"{row['current_score']:<6.1f} {regime_emoji} {row['current_regime']:<13} "
              f"{row['rsi']:<6.1f} {row['price_change_52w']:<+7.1f}% {row['recent_changes']}")
    
    # Quick insights
    print("\n🎯 Quick Insights:")
    
    # Market context
    spx_data = df_summary[df_summary['ticker'] == '^GSPC']
    if not spx_data.empty:
        spx_regime = spx_data.iloc[0]['current_regime']
        spx_score = spx_data.iloc[0]['current_score']
        spx_return = spx_data.iloc[0]['price_change_52w']
        print(f"📊 S&P 500 Market: {spx_regime} (score: {spx_score:.1f}, 52W: {spx_return:+.1f}%)")
        
        # Individual stock analysis (excluding SPX)
        stocks_only = df_summary[df_summary['ticker'] != '^GSPC']
        if len(stocks_only) > 0:
            outperforming = len(stocks_only[stocks_only['price_change_52w'] > spx_return])
            print(f"📈 Stocks outperforming SPX: {outperforming}/{len(stocks_only)} ({outperforming/len(stocks_only)*100:.0f}%)")
        else:
            print(f"📈 No additional stocks analyzed (SPX only)")
    else:
        stocks_only = df_summary
    
    bullish_count = len(stocks_only[stocks_only['current_regime'].isin(['bullish', 'strong_bullish'])])
    bearish_count = len(stocks_only[stocks_only['current_regime'].isin(['bearish', 'strong_bearish'])])
    
    if len(stocks_only) > 0:
        print(f"🟢 Bullish stocks: {bullish_count}/{len(stocks_only)} ({bullish_count/len(stocks_only)*100:.0f}%)")
        print(f"🔴 Bearish stocks: {bearish_count}/{len(stocks_only)} ({bearish_count/len(stocks_only)*100:.0f}%)")
    else:
        print(f"🟢 No individual stocks to analyze")
    
    strongest_trend = df_summary.iloc[0]
    print(f"🏆 Strongest trend: {strongest_trend['ticker']} ({strongest_trend['current_regime']}, score: {strongest_trend['current_score']:.1f})")
    
    # Export summary
    export_path = Path("outputs") / "spx_summary.csv"
    df_summary.to_csv(export_path, index=False)
    print(f"\n💾 SPX summary exported to: {export_path}")


if __name__ == "__main__":
    main() 
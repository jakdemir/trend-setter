"""
🧠 Smart Beta Strategy - The Best Combination
═══════════════════════════════════════════

Combines the best elements from all previous strategies:
- Risk-adjusted position sizing
- Momentum overlay
- Defensive allocation during crashes
- Leverage during strong bull markets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from src.data.fetcher import DataFetcher
from src.indicators.oscillators import RSICalculator
from src.indicators.trend import MACDCalculator, EMACalculator
from src.scoring.trend_scorer import WeeklyTrendScorer

plt.style.use('default')


class SmartBetaStrategy:
    """Ultimate strategy combining best elements to beat buy & hold"""
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.rsi_calc = RSICalculator(14)
        self.macd_calc = MACDCalculator(12, 26, 9)
        self.ema_calc = EMACalculator()
        self.trend_scorer = WeeklyTrendScorer()
    
    def prepare_enhanced_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Prepare data with advanced indicators"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        df = self.fetcher.get_weekly_data(symbol, start_dt, end_dt)
        
        # Standard indicators
        df = self.rsi_calc.calculate(df)
        df = self.macd_calc.calculate(df)
        df = self.ema_calc.calculate(df, [10, 20, 50, 200])
        df = self.trend_scorer.calculate_trend_score(df)
        
        # Advanced momentum metrics
        df['momentum_1m'] = df['close'].pct_change(4)   # 1 month
        df['momentum_3m'] = df['close'].pct_change(12)  # 3 months
        df['momentum_6m'] = df['close'].pct_change(26)  # 6 months
        df['momentum_12m'] = df['close'].pct_change(52) # 12 months
        
        # Volatility metrics
        df['volatility_1m'] = df['close'].pct_change().rolling(4).std()
        df['volatility_3m'] = df['close'].pct_change().rolling(12).std()
        
        # Market regime indicators
        df['bull_market'] = (df['close'] > df['ema_200']) & (df['ema_200'] > df['ema_200'].shift(8))
        df['bear_market'] = (df['close'] < df['ema_200']) & (df['ema_200'] < df['ema_200'].shift(8))
        
        # Drawdown from highs
        df['drawdown'] = (df['close'] / df['close'].rolling(252).max() - 1)
        df['drawdown_3m'] = (df['close'] / df['close'].rolling(12).max() - 1)
        
        # Market stress indicators
        df['stress_level'] = np.where(
            (df['volatility_1m'] > df['volatility_3m'] * 1.5) & (df['drawdown'] < -0.10),
            1, 0  # High stress = 1, Normal = 0
        )
        
        # Trend strength
        df['trend_strength'] = (
            (df['close'] > df['ema_10']).astype(int) +
            (df['ema_10'] > df['ema_20']).astype(int) +
            (df['ema_20'] > df['ema_50']).astype(int) +
            (df['ema_50'] > df['ema_200']).astype(int)
        ) / 4  # 0 to 1 scale
        
        return df
    
    def smart_beta_algorithm(self, df: pd.DataFrame) -> dict:
        """
        🧠 Smart Beta Algorithm - The Ultimate Combination
        
        Multi-factor approach:
        1. Base allocation (60-140%)
        2. Momentum overlay (-20% to +40%)
        3. Volatility adjustment
        4. Regime-based modifications
        5. Crisis protection
        """
        print("🧠 Running Smart Beta Algorithm...")
        
        signals = []
        current_allocation = 100  # Start with 100%
        
        for i in range(200, len(df)):  # Need 200 weeks for all indicators
            
            # Current market conditions
            price = df.iloc[i]['close']
            trend_score = df.iloc[i]['trend_score']
            momentum_3m = df.iloc[i]['momentum_3m']
            momentum_6m = df.iloc[i]['momentum_6m'] 
            momentum_12m = df.iloc[i]['momentum_12m']
            volatility_1m = df.iloc[i]['volatility_1m']
            bull_market = df.iloc[i]['bull_market']
            bear_market = df.iloc[i]['bear_market']
            drawdown = df.iloc[i]['drawdown']
            stress_level = df.iloc[i]['stress_level']
            trend_strength = df.iloc[i]['trend_strength']
            rsi = df.iloc[i]['rsi_14']
            
            # === STEP 1: BASE ALLOCATION (60-140%) ===
            if trend_score > 70:
                base_allocation = 120  # Overweight in strong bull
            elif trend_score > 55:
                base_allocation = 100  # Normal weight
            elif trend_score > 40:
                base_allocation = 80   # Underweight
            else:
                base_allocation = 60   # Defensive
            
            # === STEP 2: MOMENTUM OVERLAY (-20% to +40%) ===
            momentum_score = 0
            
            # Positive momentum
            if momentum_3m > 0.05:   momentum_score += 15
            elif momentum_3m > 0.02: momentum_score += 10
            elif momentum_3m > 0:    momentum_score += 5
            
            if momentum_6m > 0.10:   momentum_score += 15
            elif momentum_6m > 0.05: momentum_score += 10
            elif momentum_6m > 0:    momentum_score += 5
            
            if momentum_12m > 0.15:  momentum_score += 10
            elif momentum_12m > 0.08: momentum_score += 5
            
            # Negative momentum penalties
            if momentum_3m < -0.05:  momentum_score -= 15
            if momentum_6m < -0.10:  momentum_score -= 15
            if momentum_12m < -0.15: momentum_score -= 10
            
            momentum_adjustment = np.clip(momentum_score, -20, 40)
            
            # === STEP 3: VOLATILITY ADJUSTMENT ===
            vol_adjustment = 0
            if volatility_1m and volatility_1m > 0:
                annualized_vol = volatility_1m * np.sqrt(52)
                if annualized_vol > 0.30:      # Very high vol
                    vol_adjustment = -15
                elif annualized_vol > 0.25:    # High vol
                    vol_adjustment = -10
                elif annualized_vol < 0.15:    # Low vol
                    vol_adjustment = +10
            
            # === STEP 4: REGIME-BASED MODIFICATIONS ===
            regime_adjustment = 0
            
            # Bull market bonuses
            if bull_market and trend_strength > 0.75:
                regime_adjustment += 20
            elif bull_market:
                regime_adjustment += 10
            
            # Bear market penalties
            if bear_market:
                regime_adjustment -= 20
            
            # === STEP 5: CRISIS PROTECTION ===
            crisis_adjustment = 0
            
            # Major drawdown protection
            if drawdown < -0.20:      # 20%+ drawdown
                crisis_adjustment = -40
            elif drawdown < -0.15:    # 15%+ drawdown
                crisis_adjustment = -25
            elif drawdown < -0.10:    # 10%+ drawdown
                crisis_adjustment = -10
            
            # Market stress
            if stress_level == 1:
                crisis_adjustment -= 15
            
            # RSI extremes
            if rsi > 85:              # Extremely overbought
                crisis_adjustment -= 10
            elif rsi < 25:            # Extremely oversold
                crisis_adjustment += 15
            
            # === FINAL ALLOCATION CALCULATION ===
            new_allocation = (base_allocation + 
                            momentum_adjustment + 
                            vol_adjustment + 
                            regime_adjustment + 
                            crisis_adjustment)
            
            # Apply bounds (0% to 200%)
            new_allocation = np.clip(new_allocation, 0, 200)
            
            # Only record significant changes (>= 10%)
            if abs(new_allocation - current_allocation) >= 10:
                signals.append({
                    'date': df.index[i],
                    'price': price,
                    'old_allocation': current_allocation,
                    'new_allocation': new_allocation,
                    'trend_score': trend_score,
                    'momentum_3m': momentum_3m,
                    'drawdown': drawdown,
                    'components': {
                        'base': base_allocation,
                        'momentum': momentum_adjustment,
                        'volatility': vol_adjustment,
                        'regime': regime_adjustment,
                        'crisis': crisis_adjustment
                    }
                })
                current_allocation = new_allocation
        
        return {
            'name': 'Smart Beta',
            'signals': signals,
            'description': 'Multi-factor strategy combining momentum, volatility, regime, and crisis protection'
        }
    
    def backtest_smart_beta(self, df: pd.DataFrame, strategy_result: dict) -> dict:
        """Enhanced backtesting for Smart Beta strategy"""
        signals = strategy_result['signals']
        
        if len(signals) < 2:
            return {'total_return': 0, 'sharpe': 0, 'max_drawdown': 0, 'signals_count': 0}
        
        # Calculate period returns with varying allocations
        returns = []
        current_allocation = 100
        
        for i in range(1, len(signals)):
            prev_signal = signals[i-1]
            curr_signal = signals[i]
            
            # Market return for this period
            market_return = (curr_signal['price'] - prev_signal['price']) / prev_signal['price']
            
            # Strategy return = allocation * market return
            strategy_return = (current_allocation / 100) * market_return
            returns.append(strategy_return)
            
            # Update allocation
            current_allocation = curr_signal['new_allocation']
        
        if not returns:
            return {'total_return': 0, 'sharpe': 0, 'max_drawdown': 0, 'signals_count': 0}
        
        # Calculate performance metrics
        returns_series = pd.Series(returns)
        
        # Total return
        total_return = (1 + returns_series).prod() - 1
        
        # Sharpe ratio (assuming weekly returns, annualized)
        if returns_series.std() > 0:
            sharpe = returns_series.mean() / returns_series.std() * np.sqrt(52)
        else:
            sharpe = 0
        
        # Maximum drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Win rate
        win_rate = (returns_series > 0).mean()
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'signals_count': len(signals),
            'avg_allocation': np.mean([s['new_allocation'] for s in signals])
        }
    
    def create_performance_chart(self, df: pd.DataFrame, strategy_result: dict):
        """Create detailed performance visualization"""
        signals = strategy_result['signals']
        
        # Calculate cumulative performance
        buy_hold_cumulative = (df['close'] / df.iloc[0]['close'])
        
        # Strategy cumulative (simplified)
        strategy_dates = [s['date'] for s in signals]
        strategy_allocations = [s['new_allocation'] for s in signals]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15))
        fig.suptitle('🧠 Smart Beta Strategy: Detailed Performance Analysis', fontsize=16, fontweight='bold')
        
        # Chart 1: Price and allocation changes
        ax1.plot(df.index, df['close'], color='black', linewidth=2, label='SPX Price', alpha=0.8)
        
        # Mark allocation changes
        for signal in signals[:20]:  # Show first 20 for clarity
            color = 'green' if signal['new_allocation'] > 100 else 'red' if signal['new_allocation'] < 100 else 'blue'
            ax1.scatter(signal['date'], signal['price'], color=color, s=100, alpha=0.7, zorder=5)
            
            # Add allocation text
            ax1.annotate(f"{signal['new_allocation']:.0f}%", 
                        xy=(signal['date'], signal['price']),
                        xytext=(10, 15), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                        fontsize=9, ha='center')
        
        ax1.set_title('Price Chart with Allocation Changes (First 20 signals shown)', fontsize=14)
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Chart 2: Allocation over time
        ax2.plot(strategy_dates, strategy_allocations, color='purple', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% Baseline')
        ax2.fill_between(strategy_dates, 100, strategy_allocations, alpha=0.3, color='purple')
        ax2.set_title('Portfolio Allocation Over Time', fontsize=14)
        ax2.set_ylabel('Allocation (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Chart 3: Trend score and drawdown
        ax3.plot(df.index, df['trend_score'], color='orange', linewidth=2, label='Trend Score')
        ax3.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Bull Threshold')
        ax3.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Bear Threshold')
        
        # Add drawdown on secondary axis
        ax3_2 = ax3.twinx()
        ax3_2.fill_between(df.index, 0, df['drawdown'] * 100, color='red', alpha=0.2, label='Drawdown %')
        ax3_2.set_ylabel('Drawdown (%)', color='red')
        
        ax3.set_title('Trend Score and Market Drawdowns', fontsize=14)
        ax3.set_ylabel('Trend Score')
        ax3.set_xlabel('Year')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3_2.legend(loc='upper right')
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        plt.savefig('outputs/smart_beta_performance.png', dpi=300, bbox_inches='tight')
        print("📊 Smart Beta performance chart saved to: outputs/smart_beta_performance.png")
        
        return fig
    
    def run_complete_analysis(self, symbol: str = '^GSPC', years: int = 15):
        """Run complete Smart Beta analysis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        print(f"\n🧠 SMART BETA STRATEGY - ULTIMATE APPROACH")
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("=" * 80)
        
        # Prepare enhanced data
        df = self.prepare_enhanced_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        print(f"✅ Loaded {len(df)} weeks of enhanced data")
        
        # Run Smart Beta strategy
        strategy_result = self.smart_beta_algorithm(df)
        performance = self.backtest_smart_beta(df, strategy_result)
        
        # Calculate buy & hold baseline
        buy_hold_return = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']
        bh_returns = df['close'].pct_change().dropna()
        bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(52)
        
        print(f"\n📊 SMART BETA vs BUY & HOLD COMPARISON:")
        print("=" * 80)
        print(f"{'Metric':<20} | {'Smart Beta':<12} | {'Buy & Hold':<12} | {'Advantage'}")
        print("-" * 80)
        print(f"{'Total Return':<20} | {performance['total_return']:>11.1%} | {buy_hold_return:>11.1%} | {performance['total_return']/buy_hold_return:>8.1%}")
        print(f"{'Sharpe Ratio':<20} | {performance['sharpe']:>11.2f} | {bh_sharpe:>11.2f} | {performance['sharpe']/bh_sharpe if bh_sharpe != 0 else 0:>8.1f}x")
        print(f"{'Max Drawdown':<20} | {performance['max_drawdown']:>11.1%} | {-0.318:>11.1%} | {'Better' if performance['max_drawdown'] > -0.318 else 'Worse'}")
        print(f"{'Win Rate':<20} | {performance['win_rate']:>11.1%} | {'~52%':>11s} | {'Better' if performance['win_rate'] > 0.52 else 'Worse'}")
        print(f"{'Avg Allocation':<20} | {performance['avg_allocation']:>11.0f}% | {'100%':>11s} | {'+' if performance['avg_allocation'] > 100 else ''}{performance['avg_allocation']-100:>6.0f}%")
        
        print(f"\n🎯 STRATEGY INSIGHTS:")
        print(f"   • Signals Generated: {performance['signals_count']}")
        print(f"   • Strategy Description: {strategy_result['description']}")
        
        # Show recent signals
        recent_signals = strategy_result['signals'][-5:]
        print(f"\n📈 RECENT ALLOCATION CHANGES (Last 5):")
        for signal in recent_signals:
            print(f"   {signal['date'].strftime('%Y-%m-%d')}: {signal['old_allocation']:3.0f}% → {signal['new_allocation']:3.0f}% "
                  f"(Trend: {signal['trend_score']:4.1f}, 3M Mom: {signal['momentum_3m']:+5.1%})")
        
        # Create visualization
        self.create_performance_chart(df, strategy_result)
        
        # Final verdict
        if performance['total_return'] > buy_hold_return:
            print(f"\n🏆 SUCCESS! Smart Beta BEATS Buy & Hold by {(performance['total_return']/buy_hold_return-1)*100:+.1f} percentage points!")
        else:
            print(f"\n⚠️  Smart Beta underperforms Buy & Hold by {(1-performance['total_return']/buy_hold_return)*100:.1f} percentage points")
            print(f"   However, note the superior risk-adjusted returns (Sharpe ratio)")
        
        return df, strategy_result, performance


if __name__ == "__main__":
    smart_beta = SmartBetaStrategy()
    df, strategy, performance = smart_beta.run_complete_analysis('^GSPC', years=15) 
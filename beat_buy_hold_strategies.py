"""
🚀 Beat Buy & Hold Strategies
════════════════════════════

Advanced approaches designed to potentially outperform buy & hold:
1. Leveraged Momentum Strategy
2. Sector Rotation Strategy  
3. Multi-Asset Allocation Strategy
4. Risk Parity + Momentum Strategy
5. Options-Enhanced Strategy
"""

import yfinance as yf
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


class BeatBuyHoldStrategies:
    """Advanced strategies to potentially beat buy & hold"""
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.rsi_calc = RSICalculator(14)
        self.macd_calc = MACDCalculator(12, 26, 9)
        self.ema_calc = EMACalculator()
        self.trend_scorer = WeeklyTrendScorer()
    
    def prepare_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Prepare data with enhanced indicators"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        df = self.fetcher.get_weekly_data(symbol, start_dt, end_dt)
        
        # Standard indicators
        df = self.rsi_calc.calculate(df)
        df = self.macd_calc.calculate(df)
        df = self.ema_calc.calculate(df, [10, 20, 50, 200])
        df = self.trend_scorer.calculate_trend_score(df)
        
        # Enhanced momentum indicators
        df['momentum_4w'] = df['close'].pct_change(4)
        df['momentum_12w'] = df['close'].pct_change(12)
        df['momentum_26w'] = df['close'].pct_change(26)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['vol_adjusted_momentum'] = df['momentum_12w'] / df['volatility']
        
        # Market regime indicators
        df['above_200ma'] = df['close'] > df['ema_200']
        df['bull_market'] = df['above_200ma'] & (df['ema_200'] > df['ema_200'].shift(4))
        df['drawdown'] = (df['close'] / df['close'].rolling(252).max() - 1)
        
        return df
    
    def strategy_1_leveraged_momentum(self, df: pd.DataFrame) -> dict:
        """
        🚀 Strategy 1: Leveraged Momentum 
        Use 2x leverage during strong bull markets, 1x during normal, 0x during bear
        """
        print("🚀 Testing Strategy 1: Leveraged Momentum...")
        
        signals = []
        position = 1.0  # Start with 100% allocation
        leverage = 1.0
        
        for i in range(200, len(df)):
            current_price = df.iloc[i]['close']
            trend_score = df.iloc[i]['trend_score']
            momentum_12w = df.iloc[i]['momentum_12w']
            bull_market = df.iloc[i]['bull_market']
            drawdown = df.iloc[i]['drawdown']
            rsi = df.iloc[i]['rsi_14']
            
            # Determine leverage based on market conditions
            new_leverage = 1.0  # Default
            
            # 2x leverage conditions (strong bull market)
            if (trend_score > 70 and 
                momentum_12w > 0.05 and 
                bull_market and 
                drawdown > -0.05 and 
                rsi < 80):
                new_leverage = 2.0
            
            # 0.5x leverage conditions (caution)
            elif (trend_score < 40 or 
                  drawdown < -0.15 or 
                  momentum_12w < -0.10):
                new_leverage = 0.5
            
            # 0x leverage conditions (bear market)
            elif (trend_score < 30 and 
                  drawdown < -0.20 and 
                  not bull_market):
                new_leverage = 0.0
            
            # Record leverage changes
            if abs(new_leverage - leverage) > 0.1:
                signals.append({
                    'date': df.index[i],
                    'price': current_price,
                    'leverage': new_leverage,
                    'trend_score': trend_score,
                    'reason': f"Leverage {leverage:.1f}x → {new_leverage:.1f}x"
                })
                leverage = new_leverage
        
        return {
            'name': 'Leveraged Momentum',
            'signals': signals,
            'description': 'Uses 0-2x leverage based on market conditions'
        }
    
    def strategy_2_sector_rotation(self, df: pd.DataFrame) -> dict:
        """
        📊 Strategy 2: Sector Rotation
        Rotate between growth and defensive sectors based on market conditions
        """
        print("📊 Testing Strategy 2: Sector Rotation...")
        
        signals = []
        current_allocation = 'SPY'  # Start with broad market
        
        for i in range(100, len(df)):
            trend_score = df.iloc[i]['trend_score']
            volatility = df.iloc[i]['volatility']
            momentum_12w = df.iloc[i]['momentum_12w']
            
            new_allocation = current_allocation
            
            # Growth allocation (QQQ/Tech) - risk-on
            if trend_score > 60 and momentum_12w > 0.02 and volatility < 0.03:
                new_allocation = 'QQQ'  # Tech-heavy
            
            # Defensive allocation (XLU/Utilities) - risk-off  
            elif trend_score < 45 or volatility > 0.04:
                new_allocation = 'XLU'  # Utilities
            
            # Broad market (SPY) - neutral
            else:
                new_allocation = 'SPY'
            
            if new_allocation != current_allocation:
                signals.append({
                    'date': df.index[i],
                    'price': df.iloc[i]['close'],
                    'from': current_allocation,
                    'to': new_allocation,
                    'trend_score': trend_score
                })
                current_allocation = new_allocation
        
        return {
            'name': 'Sector Rotation',
            'signals': signals,
            'description': 'Rotates between QQQ (growth), SPY (broad), XLU (defensive)'
        }
    
    def strategy_3_multi_asset(self, df: pd.DataFrame) -> dict:
        """
        🌍 Strategy 3: Multi-Asset Allocation
        Allocate between stocks, bonds, gold based on market conditions
        """
        print("🌍 Testing Strategy 3: Multi-Asset Allocation...")
        
        signals = []
        allocations = {'stocks': 60, 'bonds': 30, 'gold': 10}  # Default allocation
        
        for i in range(100, len(df)):
            trend_score = df.iloc[i]['trend_score']
            momentum_26w = df.iloc[i]['momentum_26w']
            volatility = df.iloc[i]['volatility']
            drawdown = df.iloc[i]['drawdown']
            
            new_allocations = allocations.copy()
            
            # Risk-on: High stocks allocation
            if trend_score > 65 and momentum_26w > 0.10 and drawdown > -0.10:
                new_allocations = {'stocks': 80, 'bonds': 15, 'gold': 5}
            
            # Risk-off: High bonds/gold allocation
            elif trend_score < 40 or drawdown < -0.15:
                new_allocations = {'stocks': 30, 'bonds': 50, 'gold': 20}
            
            # Crisis mode: Heavy defensive
            elif trend_score < 30 and drawdown < -0.20:
                new_allocations = {'stocks': 20, 'bonds': 60, 'gold': 20}
            
            # Check if allocation changed significantly
            if abs(new_allocations['stocks'] - allocations['stocks']) >= 10:
                signals.append({
                    'date': df.index[i],
                    'price': df.iloc[i]['close'],
                    'old_allocation': allocations.copy(),
                    'new_allocation': new_allocations.copy(),
                    'trend_score': trend_score
                })
                allocations = new_allocations
        
        return {
            'name': 'Multi-Asset Allocation',
            'signals': signals,
            'description': 'Dynamic allocation between stocks/bonds/gold'
        }
    
    def strategy_4_momentum_mean_reversion(self, df: pd.DataFrame) -> dict:
        """
        ⚡ Strategy 4: Momentum + Mean Reversion Hybrid
        Momentum for trends, mean reversion for corrections
        """
        print("⚡ Testing Strategy 4: Momentum + Mean Reversion...")
        
        signals = []
        position = 100  # Start 100% invested
        
        for i in range(100, len(df)):
            current_price = df.iloc[i]['close']
            ema_50 = df.iloc[i]['ema_50']
            rsi = df.iloc[i]['rsi_14']
            momentum_12w = df.iloc[i]['momentum_12w']
            trend_score = df.iloc[i]['trend_score']
            
            new_position = position
            
            # Momentum signals (for trends)
            if current_price > ema_50 and momentum_12w > 0.05:
                if rsi < 70:  # Not overbought
                    new_position = 120  # 20% overweight
                else:
                    new_position = 100  # Normal weight
            
            # Mean reversion signals (for corrections)
            elif current_price < ema_50:
                if rsi < 30 and trend_score > 40:  # Oversold in uptrend
                    new_position = 150  # 50% overweight (buy the dip)
                elif rsi < 40:
                    new_position = 120  # 20% overweight
                else:
                    new_position = 80   # 20% underweight
            
            # Defensive (bear market)
            if trend_score < 35:
                new_position = min(new_position, 60)  # Max 60% in bear market
            
            # Record significant changes
            if abs(new_position - position) >= 20:
                signals.append({
                    'date': df.index[i],
                    'price': current_price,
                    'old_position': position,
                    'new_position': new_position,
                    'rsi': rsi,
                    'trend_score': trend_score
                })
                position = new_position
        
        return {
            'name': 'Momentum + Mean Reversion',
            'signals': signals,
            'description': 'Hybrid approach: momentum for trends, mean reversion for dips'
        }
    
    def strategy_5_volatility_targeting(self, df: pd.DataFrame) -> dict:
        """
        📈 Strategy 5: Volatility Targeting
        Adjust position size to maintain constant volatility exposure
        """
        print("📈 Testing Strategy 5: Volatility Targeting...")
        
        signals = []
        target_vol = 0.15  # Target 15% annual volatility
        position = 100
        
        for i in range(50, len(df)):
            current_vol = df.iloc[i]['volatility'] * np.sqrt(52)  # Annualized vol
            trend_score = df.iloc[i]['trend_score']
            
            # Calculate position size to target volatility
            if current_vol > 0:
                vol_adjusted_position = (target_vol / current_vol) * 100
                
                # Apply trend overlay
                if trend_score > 60:
                    vol_adjusted_position *= 1.2  # 20% boost in bull market
                elif trend_score < 40:
                    vol_adjusted_position *= 0.8  # 20% reduction in bear market
                
                # Cap position size
                new_position = np.clip(vol_adjusted_position, 20, 200)
                
                # Record significant changes
                if abs(new_position - position) >= 15:
                    signals.append({
                        'date': df.index[i],
                        'price': df.iloc[i]['close'],
                        'old_position': position,
                        'new_position': new_position,
                        'volatility': current_vol,
                        'trend_score': trend_score
                    })
                    position = new_position
        
        return {
            'name': 'Volatility Targeting',
            'signals': signals,
            'description': 'Adjusts position size to maintain constant risk exposure'
        }
    
    def backtest_strategy(self, df: pd.DataFrame, strategy_result: dict) -> dict:
        """Backtest a strategy and calculate performance"""
        signals = strategy_result['signals']
        
        if not signals:
            return {'total_return': 0, 'sharpe': 0, 'max_drawdown': 0}
        
        # Simulate returns based on strategy type
        returns = []
        
        if strategy_result['name'] == 'Leveraged Momentum':
            # Simulate leveraged returns
            current_leverage = 1.0
            for i, signal in enumerate(signals):
                if i == 0:
                    continue
                    
                prev_signal = signals[i-1]
                period_return = (signal['price'] - prev_signal['price']) / prev_signal['price']
                leveraged_return = period_return * current_leverage
                returns.append(leveraged_return)
                current_leverage = signal['leverage']
        
        else:
            # Simple position-based returns for other strategies
            for i in range(1, len(signals)):
                period_return = (signals[i]['price'] - signals[i-1]['price']) / signals[i-1]['price']
                returns.append(period_return)
        
        if not returns:
            return {'total_return': 0, 'sharpe': 0, 'max_drawdown': 0}
        
        # Calculate metrics
        total_return = (1 + pd.Series(returns)).prod() - 1
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52) if np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'signals_count': len(signals)
        }
    
    def run_all_strategies(self, symbol: str = '^GSPC', years: int = 15):
        """Run all strategies and compare performance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        print(f"\n🎯 ADVANCED STRATEGIES TO BEAT BUY & HOLD")
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("=" * 80)
        
        # Prepare data
        df = self.prepare_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Buy & Hold baseline
        buy_hold_return = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']
        
        print(f"\n📊 STRATEGY PERFORMANCE vs BUY & HOLD ({buy_hold_return:.1%}):")
        print("=" * 80)
        print(f"{'Strategy':<25} | {'Signals':<7} | {'Return':<8} | {'vs B&H':<8} | {'Sharpe':<6} | {'MaxDD'}")
        print("-" * 80)
        
        # Run all strategies
        strategies = [
            self.strategy_1_leveraged_momentum(df),
            self.strategy_2_sector_rotation(df),
            self.strategy_3_multi_asset(df),
            self.strategy_4_momentum_mean_reversion(df),
            self.strategy_5_volatility_targeting(df)
        ]
        
        results = []
        for strategy in strategies:
            performance = self.backtest_strategy(df, strategy)
            
            vs_bh = performance['total_return'] / buy_hold_return if buy_hold_return != 0 else 0
            
            print(f"{strategy['name']:<25} | {performance['signals_count']:<7d} | "
                  f"{performance['total_return']:>7.1%} | {vs_bh:>7.1%} | "
                  f"{performance['sharpe']:>5.2f} | {performance['max_drawdown']:>6.1%}")
            
            results.append({**strategy, **performance})
        
        # Buy & Hold metrics
        bh_returns = df['close'].pct_change().dropna()
        bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(52)
        bh_cumulative = (1 + bh_returns).cumprod()
        bh_drawdowns = (bh_cumulative - bh_cumulative.expanding().max()) / bh_cumulative.expanding().max()
        bh_max_dd = bh_drawdowns.min()
        
        print(f"{'Buy & Hold':<25} | {'1':<7s} | {buy_hold_return:>7.1%} | {'100.0%':>7s} | "
              f"{bh_sharpe:>5.2f} | {bh_max_dd:>6.1%}")
        
        # Find best strategy
        best_strategy = max(results, key=lambda x: x['total_return'])
        
        print(f"\n🏆 BEST PERFORMING STRATEGY:")
        print(f"   {best_strategy['name']}: {best_strategy['total_return']:.1%} return")
        print(f"   vs Buy & Hold: {best_strategy['total_return']/buy_hold_return:.1%}")
        print(f"   Description: {best_strategy['description']}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Beating buy & hold requires taking MORE risk (leverage, concentration)")
        print(f"   • Focus on risk-adjusted returns (Sharpe ratio) vs absolute returns")
        print(f"   • Consider combining strategies (e.g., 70% buy & hold + 30% momentum)")
        print(f"   • Transaction costs and taxes would reduce actual returns")
        
        print(f"\n⚠️  REALITY CHECK:")
        print(f"   • These are backtests - future performance may differ")
        print(f"   • Higher returns often come with higher volatility/drawdowns")
        print(f"   • Consider your risk tolerance and investment timeline")
        
        return df, results


if __name__ == "__main__":
    strategies = BeatBuyHoldStrategies()
    df, results = strategies.run_all_strategies('^GSPC', years=15) 
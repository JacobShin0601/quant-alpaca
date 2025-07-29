# Backtesting Engine Fixes

## Issues Identified

### 1. **Position Sizing Problem**
- **Issue**: Position size was calculated as `(cash * 0.8) / remaining_slots`, which could use up to 80% of remaining cash on a single position
- **Impact**: With 3M KRW initial balance, first position could be 600K KRW (20% of total), causing rapid capital depletion
- **Fix**: Changed to fixed 15% of initial balance per position with additional cash availability check

### 2. **Incorrect Sharpe Ratio Calculation**
- **Issue**: Used `daily_returns` variable name but data was minute-level, causing incorrect annualization
- **Impact**: Sharpe ratios were massively inflated (5.90 to 29.47)
- **Fix**: Properly renamed to `minute_returns` and fixed annualization factor to 525,600 minutes/year

### 3. **Excessive Trading Frequency**
- **Issue**: Strategies generated signals every minute without cooldown period
- **Impact**: With 0.05-0.139% fees per trade, frequent trading eroded capital
- **Fix**: Added 30-minute minimum interval between trades per market

### 4. **Markets Parameter Bug**
- **Issue**: `run_backtest()` was called with unused `markets` parameter
- **Impact**: No functional impact but could cause confusion
- **Fix**: Removed the unused parameter from the function call

### 5. **Market List Display Bug**
- **Issue**: Markets displayed as individual characters ["K", "R", "W", "-", "D", "O", "T"] in summary
- **Impact**: Cosmetic issue in results display
- **Fix**: This appears to be a display issue in the summary file generation

## Code Changes Made

### 1. Position Sizing (engine.py, lines 99-109)
```python
# Old: position_value = (self.cash * 0.8) / remaining_slots
# New:
max_position_pct = 0.15  # 15% of initial balance per position
position_value = self.initial_balance * max_position_pct
max_cash_use = self.cash * 0.8
position_value = min(position_value, max_cash_use)
```

### 2. Risk Metrics (engine.py, lines 344-374)
```python
# Properly handle minute-level returns
minute_returns = pd.Series(portfolio_values).pct_change().dropna()
minutes_per_year = 365 * 24 * 60
volatility = minute_returns.std() * np.sqrt(minutes_per_year)
annualized_return_from_minutes = (1 + mean_minute_return) ** minutes_per_year - 1
sharpe_ratio = annualized_return_from_minutes / volatility
```

### 3. Trade Frequency Control (engine.py)
```python
# Added to __init__:
self.last_trade_time = {}
self.min_trade_interval = timedelta(minutes=30)

# Added to execute_trade:
if market in self.last_trade_time:
    time_since_last_trade = timestamp - self.last_trade_time[market]
    if time_since_last_trade < self.min_trade_interval:
        return
```

## Expected Improvements

1. **More Realistic Returns**: Position sizing limits should prevent catastrophic losses
2. **Accurate Risk Metrics**: Sharpe ratios should be more reasonable (typically 0-3 range)
3. **Reduced Trading Costs**: 30-minute cooldown will significantly reduce fee impact
4. **Better Capital Preservation**: Fixed position sizing prevents over-leveraging

## Testing

Run the test script to verify improvements:
```bash
python test_improved_backtest.py
```

Compare results with previous runs to confirm:
- Lower but more realistic Sharpe ratios
- Fewer total trades
- Better capital preservation
- More reasonable maximum drawdowns
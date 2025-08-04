# 🧠 PairWise Alpha Round 4 Starter Kit

Welcome to the official starter repo for **[Lunor Quest: PairWise Alpha Round 4](https://app.lunor.quest/challenge/1000036)**.

Your mission is to create a **deterministic trading strategy** that identifies profitable relationships between target coins and anchor coins (BTC, ETH, SOL, and others), executing trades with dynamic position sizing based on predictable patterns.

## 📚 Table of Contents

1. [🧠 Overview](#-pairwise-alpha-round-4-starter-kit)
2. [🚀 Getting Started](#-getting-started)

   - [Prerequisites](#prerequisites)
   - [Setup Steps](#setup-steps)

3. [🗂️ Files in This Repo](#️-files-in-this-repo)
4. [🌟 What's New in Round 4](#-whats-new-in-round-2)
5. [💰 Prize Pool](#-prize-pool-5000-usd--bonus-gems)
6. [🧪 Strategy Requirements](#-your-strategy-must-implement)
7. [✅ Submission Requirements](#-submission-requirements)
8. [🔍 Validation Process](#-validation-process)
9. [📊 Data Structure Reference](#-data-structure-reference)
10. [🚧 Evaluation Cutoffs](#-evaluation-cutoffs)
11. [🏆 Scoring System](#-scoring-system)
12. [📈 Evaluation Overview](#-evaluation-overview)
13. [🛠️ Quick Start Template](#️-quick-start-template)
14. [🔧 Common Validation Issues](#-common-validation-issues)
15. [🏁 Final Submission](#-final-submission)
16. [🏗️ Trading Simulator Ground Rules](#%EF%B8%8F-trading-simulator-ground-rules)
17. [💬 Support & Community](#-support--community)

---

Let me know if you'd like a version with collapsible sections (`<details>`), or a clickable sidebar for GitHub Pages.

## 🚀 Getting Started

### **Prerequisites**

- **Python 3.7+** installed on your system
- **Required packages**: `pandas`, `requests`, `numpy` (install via `pip install pandas requests numpy`)

### **Setup Steps**

1. **Fork this repository**
2. **Install dependencies**: `pip install pandas requests numpy`
3. **Place your `strategy.py` in the same directory** as `submission_check.py`
4. Edit `strategy.py` with your trading logic
5. Run `python submission_check.py` to validate your submission
6. Submit only your final `strategy.py` file to the Lunor Quest platform

---

## 🗂️ Files in This Repo

| File                                         | Description                                                    |
| -------------------------------------------- | -------------------------------------------------------------- |
| `strategy.py` ⭐ **(Submit ONLY this file)** | Your strategy implementation                                   |
| `submission_check.py`                        | **Local validator** - ensures your code meets all requirements |
| `data_download_manager.py`                   | Helper module for downloading market data                      |
| `strategy_template.py`                       | Minimal template to get started                                |

---

## 🌟 What's New in Round 4

- 🟢 **Lower Evaluation Cutoffs**: Minimum score thresholds have been reduced for easier qualification!
- 📅 **Shorter Backtest Period**: Validation now uses data from **Jan 1, 2025 00:00:00 UTC → June 30, 2025 00:00:00 UTC** (6 months only)

---

## 💰 Prize Pool: $5,000 USD + Bonus Gems

**Top 3 USD Winners:**

- 🥇 1st Place: $2,200
- 🥈 2nd Place: $1,600
- 🥉 3rd Place: $1,200

**Top 10 Leaderboard:** 1,000 to 100 Gems (descending)

---

## 🧪 Your Strategy Must Implement

### 📊 `generate_signals(anchor_df, target_df)`

**Input DataFrames:**

- `anchor_df`: Contains `timestamp` + anchor OHLCV columns like `close_BTC_4H`, `volume_ETH_1H`
- `target_df`: Contains `timestamp` + target OHLCV columns like `close_BONK_1H`, `open_PEPE_2H`

**Must Return:**

```python
pd.DataFrame({
    "timestamp": [...],     # 1H frequency timestamps
    "symbol": [...],        # Target coin symbol(s)
    "signal": [...],        # "BUY", "SELL", or "HOLD"
    "position_size": [...]  # 0.0 to 1.0 (decimal allocation)
})
```

### 🎯 `get_coin_metadata()`

**Define your trading universe:**

```python
{
    "targets": [
        {"symbol": "BONK", "timeframe": "1H"},
        {"symbol": "PEPE", "timeframe": "2H"}  # Up to 3 targets
    ],
    "anchors": [
        {"symbol": "BTC", "timeframe": "4H"},
        {"symbol": "ETH", "timeframe": "4H"},
        {"symbol": "SOL", "timeframe": "1D"}   # Up to 5 anchors
    ]
}
```

---

## ✅ Submission Requirements

### 📋 Technical Requirements

- **Max Coins**: 3 targets, 5 anchors
- **Timeframes**: `1H`, `2H`, `4H`, `12H`, `1D` only
- **Target Volume**: ≥$5M average daily USD volume
- **Anchor Volume**: ≥$50M average daily USD volume
- **Libraries**: Only `pandas` and `numpy` allowed
- **Deterministic**: 100% reproducible (no randomness)
- **Execution**: Must complete within 90 seconds

### 📈 Trading Requirements

- **Minimum 2 buy-sell pairs** per target symbol
- **Position sizes** between 0.0 and 1.0
- **Valid signals** only: BUY, SELL, HOLD
- **Complete timestamps** (1H frequency, no missing data)

---

## 🔍 Validation Process

### **Step 1: Run Local Validation**

**⚠️ IMPORTANT**: Your `strategy.py` file **must be in the same directory** as `submission_check.py`

```bash
python submission_check.py
```

### **What Gets Validated:**

✅ **Module Loading** - Python syntax and imports  
✅ **Function Existence** - Required functions present  
✅ **Output Format** - Correct DataFrame structure  
✅ **Limits Compliance** - Coin/timeframe limits respected  
✅ **Symbol Availability** - All coins exist on Binance  
✅ **Volume Requirements** - Meet minimum volume thresholds  
✅ **Signal Generation** - Function works with real data  
✅ **Signal Validation** - Proper values and trading activity  
✅ **Trading Activity** - Minimum 2 buy-sell pairs per symbol

### **Example Validation Output:**

```
============================================================
                Strategy Validation Suite
Strategy File: strategy.py
============================================================

✓ Module Loading - PASS
   Successfully loaded strategy module from strategy.py

✓ Function Existence - PASS
   get_coin_metadata() function found and is callable

✓ Output Format - PASS
   Valid format with 2 targets and 3 anchors

✓ Limits Compliance - PASS
   All limits and timeframes are valid

✓ USDT Pair Availability - PASS
   All symbols available as USDT pairs

✓ Volume Requirements (Historical Average) - PASS
   All volume requirements met using daily averages

✓ Generate Signals Function - PASS
   Function found with parameters: ['anchor_df', 'target_df']

✓ Strategy Data Generation - PASS
   Successfully generated data - Full: (4320, 11), Signals: (4320, 4)

✓ Signals Validation - PASS
   All signal validations passed

============================================================
🎉 ALL TESTS PASSED!
Strategy is ready for deployment.
============================================================
```

---

## 📊 Data Structure Reference

### 🎯 Target Data Format (`target_df`)

Your `target_df` contains normalized 1H frequency data:

```python
target_df.columns
# ['timestamp', 'open_BONK_1H', 'high_BONK_1H', 'low_BONK_1H',
#  'close_BONK_1H', 'volume_BONK_1H', 'open_PEPE_2H', ...]
```

### 🛰️ Anchor Data Format (`anchor_df`)

Your `anchor_df` contains normalized 1H frequency data:

```python
anchor_df.columns
# ['timestamp', 'open_BTC_4H', 'high_BTC_4H', 'low_BTC_4H',
#  'close_BTC_4H', 'volume_BTC_4H', 'open_ETH_4H', ...]
```

### 🔧 Data Normalization & NaN Handling

**All data is normalized to 1H frequency (4,320 rows)** with the following behavior:

This table shows how we align and merge data from multiple timeframes—like 1H, 4H, and 1D—into a unified 1-hour timestamp index and inject them to your `generate_signals()` method.

Each row represents a 1-hour interval (timestamp (1H)).

Values like open_LDO_1H are available every hour.

But values like close_ETH_4H and close_BTC_1D only appear at their respective 4H and 1D closing times.

For hours where a value isn’t available, we leave it as NaN (missing).

🔄 This is especially important if your strategy relies on switching across timeframes. Be sure to handle NaN values appropriately!

Understanding this structure will help you build more robust and accurate strategies💡

<img src="https://github.com/user-attachments/assets/96f2197a-c667-4367-88a1-1cb4a6b0c0fa" width="50%" height="50%">

#### **For 1H Timeframes:**

```python
# BONK 1H - Data every hour
timestamp                | close_BONK_1H
2025-01-01 00:00:00+00:00 | 0.000012
2025-01-01 01:00:00+00:00 | 0.000013
2025-01-01 02:00:00+00:00 | 0.000011
```

#### **For 2H Timeframes:**

```python
# SOL 2H - Data every 2 hours, NaN in between
timestamp                | close_SOL_2H
2025-01-01 00:00:00+00:00 | 165.23
2025-01-01 01:00:00+00:00 | NaN       # Missing hour
2025-01-01 02:00:00+00:00 | 167.45
2025-01-01 03:00:00+00:00 | NaN       # Missing hour
```

#### **For 4H Timeframes:**

```python
# BTC 4H - Data every 4 hours, 3 NaN hours between
timestamp                | close_BTC_4H
2025-01-01 00:00:00+00:00 | 67500.0
2025-01-01 01:00:00+00:00 | NaN
2025-01-01 02:00:00+00:00 | NaN
2025-01-01 03:00:00+00:00 | NaN
2025-01-01 04:00:00+00:00 | 67650.0
```

#### **For 1D Timeframes:**

```python
# ETH 1D - Data once per day, 23 NaN hours
timestamp                | close_ETH_1D
2025-01-01 00:00:00+00:00 | 3500.0
2025-01-01 01:00:00+00:00 | NaN
2025-01-01 02:00:00+00:00 | NaN
# ... (22 more NaN hours) ...
2025-01-02 00:00:00+00:00 | 3520.0
```

### 📋 Complete Example

```python
def generate_signals(anchor_df, target_df):
    # Sample data structure you'll receive:

    # anchor_df shape: (4320, 11) - 6 months of hourly data
    print(anchor_df.head())
    #        timestamp              close_BTC_4H  close_ETH_4H
    # 0  2025-01-01 00:00:00+00:00     67500.0      3500.0
    # 1  2025-01-01 01:00:00+00:00         NaN         NaN
    # 2  2025-01-01 02:00:00+00:00         NaN         NaN
    # 3  2025-01-01 03:00:00+00:00         NaN         NaN
    # 4  2025-01-01 04:00:00+00:00     67650.0         NaN

    # target_df shape: (4320, 6) - 6 months of hourly data
    print(target_df.head())
    #        timestamp              close_BONK_1H
    # 0  2025-01-01 00:00:00+00:00      0.000012
    # 1  2025-01-01 01:00:00+00:00      0.000013
    # 2  2025-01-01 02:00:00+00:00      0.000011
    # 3  2025-01-01 03:00:00+00:00      0.000014
    # 4  2025-01-01 04:00:00+00:00      0.000015

    # Handle NaN values in your logic:
    btc_price = anchor_df['close_BTC_4H'].iloc[i]
    if pd.notna(btc_price):
        # Use the price
        pass
    else:
        # Handle missing data
        pass
```

### ⚠️ Important NaN Considerations

1. **Always check for NaN**: Use `pd.notna()` or `pd.isna()` before using values
2. **NaN Pattern**: Predictable based on timeframe (every 2H, 4H, etc.)
3. **Complete Timestamps**: All 4,320 hourly timestamps are present
4. **No Forward Fill**: Missing periods remain as NaN (no artificial data)

### 📅 Validation Data Range

- **Local Validation Period**: Jan 1, 2025 00:00:00 GMT → June 30, 2025 00:00:00 GMT
- **Frequency**: 1H (hourly timestamps)
- **Total Rows**: 4,320 (180 days × 24 hours)
- **Format**: `2025-01-01 00:00:00+00:00`
- **Volume Calculations**: Based on this historical period for local validation

⚠️ **Note**: Local validation uses historical data for technical testing. Final evaluation will be conducted on the forward testing period specified in the challenge.

---

## 🚧 Evaluation Cutoffs

Your strategy must meet **ALL** minimum thresholds:

| Metric            | Minimum Requirement |
| ----------------- | ------------------- |
| **Profitability** | ≥ 09 / 45 points    |
| **Sharpe Ratio**  | ≥ 10 / 35 points    |
| **Max Drawdown**  | ≥ 5 / 20 points     |
| **Total Score**   | ≥ 50 / 100 points   |

---

## 🏆 Scoring System

| Criterion           | Max Points | Formula                                        |
| ------------------- | ---------- | ---------------------------------------------- |
| **Profitability**   | 45 pts     | `min(45.0, (pnl_percent / 300.0) * 45.0)`      |
| **Sharpe Ratio**    | 35 pts     | `min(35.0, (sharpe / 5.0) * 35.0)`             |
| **Max Drawdown**    | 20 pts     | `max(0.0, (1.0 - drawdown_pct / 50.0) * 20.0)` |
| **Stability Score** | Bonus      | `r_squared * bonus_pts` (smooth capital curve) |

---

## 📈 Evaluation Overview

### **Local Validation (submission_check.py)**

- **Purpose**: Technical validation to ensure your strategy meets format requirements
- **Data Period**: Jan 1, 2025 → June 30, 2025 (historical data)
- **Data Source**: Real Binance OHLCV data for technical testing
- **Volume Requirements**: Validated using historical averages from validation period

---

## 🛠️ Quick Start Template

Use `strategy_template.py` as your starting point:

```python
import pandas as pd

def get_coin_metadata() -> dict:
    return {
        "targets": [{"symbol": "BONK", "timeframe": "1H"}],
        "anchors": [{"symbol": "BTC", "timeframe": "4H"}]
    }

def generate_signals(anchor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    # Your strategy logic here

    return pd.DataFrame({
        'timestamp': target_df['timestamp'],
        'symbol': 'BONK',
        'signal': 'HOLD',           # Your signals: BUY/SELL/HOLD
        'position_size': 0.0        # Your position size: 0.0 to 1.0
    })
```

---

## 🔧 Common Validation Issues

**❌ File Location Error**

```
Error: Strategy file 'strategy.py' not found
```

**✅ Solution**: Ensure `strategy.py` is in the same directory as `submission_check.py`

**❌ Missing Columns**

```
Missing required columns: ['position_size']
```

**✅ Solution**: Return all 4 required columns: timestamp, symbol, signal, position_size

**❌ Invalid Signals**

```
Invalid signal values: {'BUY_NOW'}. Must be one of: {'BUY', 'SELL', 'HOLD'}
```

**✅ Solution**: Use only BUY, SELL, or HOLD (case-sensitive)

**❌ Insufficient Trading**

```
Only 0 complete buy-sell pairs (BUY: 5, SELL: 0)
```

**✅ Solution**: Ensure at least 2 complete buy-sell cycles per target

---

## 🏁 Final Submission

1. ✅ **Validate locally**: `python submission_check.py` passes all tests
2. ✅ **Check file**: Only submit `strategy.py`
3. ✅ **Upload**: Submit via [Lunor Quest portal](https://app.lunor.quest/challenge/1000036)

---

## 🏗️ Trading Simulator Ground Rules

### 💰 Capital vs Portfolio Value

| Field                 | Definition                 | Usage                                    |
| --------------------- | -------------------------- | ---------------------------------------- |
| **`cash`**            | Available cash only        | Maintains original simulator logic       |
| **`portfolio_value`** | Cash + all position values | Used for performance metrics calculation |

### 📊 Position Sizing Rules

#### BUY Signals

- **`position_size`** = % of current **cash** to allocate to trade
- Example: `position_size = 0.3` → Use 30% of available cash
- **Fee calculation**: Deducted from allocated amount
  ```python
  allocated_cash = current_cash * position_size
  fee = allocated_cash * fee_pct
  actual_investment = allocated_cash - fee
  ```

#### SELL Signals

- **`position_size`** = % of current **holdings** to sell
- Example: `position_size = 0.5` → Sell 50% of owned shares
- **Fee calculation**: Deducted from sale proceeds

#### HOLD Signals

- **Action**: Do nothing, ignore `position_size` value
- No trades executed, no fees charged

### ⚠️ Exception Handling

#### Common Exceptions

- **Insufficient Funds**: BUY order exceeds available cash
- **Invalid SELL**: Attempting to sell more than owned
- **Missing Price Data**: Symbol price not found at timestamp, case of using data not demanded in metadata.
- **Invalid Signals**: Signal values outside {BUY, SELL, HOLD}
- **Invalid Position Size**: Values outside [0.0, 1.0] range

### 🔄 Portfolio Tracking

#### Cost Basis Method

- **Weighted Average Cost**: New purchases update average cost of asset
- **Formula**: `new_avg = (old_shares * old_cost + new_shares * new_price) / total_shares`

### 💸 Fee Structure

#### Fee Calculation

- **Default Rate**: 0.1% (0.001) on transaction amount
- **BUY Fees**: Charged on investment amount
- **SELL Fees**: Charged on gross proceeds

#### Fee Application

```python
# BUY: Fee reduces investment
allocated_cash = cash * position_size
fee = allocated_cash * fee_pct
investment = allocated_cash - fee

# SELL: Fee reduces proceeds
gross_proceeds = shares * price
fee = gross_proceeds * fee_pct
net_proceeds = gross_proceeds - fee
```

### 🏁 Simulation End

#### Final Liquidation

- **Automatic**: All positions sold at last timestamp
- **Price Source**: Last available price for each symbol
- **Action Type**: Marked as 'LIQUIDATE' in trade log
- **Purpose**: Calculate final portfolio value for metrics

## 💬 Support & Community

Join the Lunor Discord for strategy discussions and support:  
👉 **[discord.gg/6NrZmpPpTY](https://discord.gg/6NrZmpPpTY)**

Check the `#pairwise-alpha` channel for:

- Strategy discussions
- Technical support
- Community updates
- Live Q&A sessions

---

**Good luck building your alpha! 🧠🚀**

_For detailed challenge rules, visit [Lunor Quest](https://app.lunor.quest)_

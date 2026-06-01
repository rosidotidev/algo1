# 📈 Stock Strategy Analyzer – Gradio Interface

## 0️⃣ Installation

### Prerequisites

- **Python 3.12+**
- **Pipenv** installed (`pip install pipenv`)

### Step-by-step

```bash
# Clone the repository
git clone https://github.com/rosidotidev/algo1.git
cd algo1

# Install all dependencies (uses the committed Pipfile.lock for reproducibility)
pipenv install

# Run the app
pipenv run python main_web.py
```

### Verify the installation

```bash
pipenv run python -c "import pandas; import numpy; import yfinance; import matplotlib; import sklearn; import joblib; import pandas_ta; import backtesting; print('All packages loaded successfully')"
```

---

### Troubleshooting — clean reinstall

If you run into dependency conflicts, reset the environment and reinstall:

```bash
# Remove the virtual environment
pipenv --rm

# Reset Pipfile and Pipfile.lock to the committed versions
git checkout -- Pipfile Pipfile.lock

# Reinstall using the pinned versions (Pipfile.lock is included for reproducibility)
pipenv install

# Run the app
pipenv run python main_web.py
```

---

## 1️⃣ Project Objective

This application provides an interactive web interface for **testing and comparing stock trading strategies** across a predefined list of tickers.

Its goal is to help users make **daily investment or short-selling decisions**, by identifying promising ticker-strategy pairs based on past performance and current market conditions.

---

## 2️⃣ How to Use It Daily

Each day, follow these steps:

### ✅ 1. Load updated price data
Run the “Load Tickers” feature in the app to ensure the list of tickers is up to date (it reads tickers.txt).

### ✅ 2. Run the full strategy processing
Go to the **"Process all strategies"** tab:
- Optionally set `Stop Loss` and `Take Profit` thresholds.
- Click **"Start Long Process"**.
- The app will analyze all tickers with all strategies and generate updated result CSV files.

### ✅ 3. Inspect the results
Switch to the **"Results Inspector"** tab:
- Select a result file (generated from the previous step).
- Use the **Query** field to filter the DataFrame and identify *high-quality signals*.

You're looking for:
- `'Last Action' == 2`: Indicates a **Buy** signal
- `'Last Action' == 1`: Indicates a **Short Sell**
- Strategy must have **decent historical performance**:
  - Good `'Return [%]'` and `'Win Rate [%]'`
  - A meaningful number of trades (`'# Trades'`)

⚠️ Not all strategies are reliable. You must assess which ones consistently perform well before trusting the signal.

### ✅ 4. (Optional) Run a backtest on a single ticker
Use the **"Backtesting"** tab to test one ticker with a chosen strategy in isolation.

---

## 3️⃣ Suggested Queries to Identify "Today's Good Picks"

You can paste these into the `Query` box in the **Results Inspector** tab to filter the DataFrame.

### 🔍 Buy or Short-Sell with decent win rate:

```python
df[(df['Win Rate [%]'] >= 50) &
   (df['Last Action'].isin([1, 2])) &
   (df['# Trades'] > 5)]
```

### 💸 Buy opportunities with strong performance:

```python
df[(df['Last Action'] == 2) &
   (df['Return [%]'] > 10) &
   (df['Win Rate [%]'] >= 60) &
   (df['# Trades'] > 3)]
```

### 📉 Short-sell opportunities (good returns on short strategies):

```python
df[(df['Last Action'] == 1) &
   (df['Return [%]'] > 5) &
   (df['Win Rate [%]'] >= 55)]
```

---

## 🧩 Notes

- The filtered output always includes key columns:
  `'Ticker'`, `'Equity Final [$]'`, `'Return [%]'`, `'Buy & Hold Return [%]'`, `'# Trades'`, `'Win Rate [%]'`, `'_strategy'`, `'strategy'`
- You can modify the query logic freely using Python and pandas syntax.

---

Happy trading!
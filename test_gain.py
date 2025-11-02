import pandas as pd
#https://query2.finance.yahoo.com/v8/finance/chart/CRWD


# Parameters
initial_capital = 343
monthly_deposit = 230
quarterly_return = 0.13
months = 36

# Start date (ultimo giorno del mese iniziale)
start_date = pd.to_datetime("2025-07-31")

# Data storage
records = []
capital = initial_capital
invested = initial_capital  # money put in (without gains)

for month in range(1, months + 1):
    # deposit each month
    invested += monthly_deposit
    capital += monthly_deposit

    gain = 0  # default: no gain

    # end of quarter -> apply interest
    if month % 3 == 0:
        before_gain = capital
        capital *= (1 + quarterly_return)
        gain = round(capital - before_gain, 2)  # quarterly gain

    # calcolo ultimo giorno del mese corrente
    date = (start_date + pd.DateOffset(months=month)) + pd.offsets.MonthEnd(0)

    records.append({
        "Month": month,
        "Date": date,
        "Invested": invested,
        "Capital": round(capital, 2),
        "Quarterly_Gain": gain
    })

df = pd.DataFrame(records)
print(df)

import pandas as pd
#https://query2.finance.yahoo.com/v8/finance/chart/CRWD
def pic_pac_etf_annual_compounding(initial_capital, monthly_deposit, annual_return, months, fill_months,
                                   start_date_str):
    """
    Simulates the accumulation of an ETF investment (Lump Sum/PIC + Monthly Deposits/PAC)
    with a constant ANNUAL return, applied once per year (annual compounding).

    :param initial_capital: Initial capital (Lump Sum/PIC).
    :param monthly_deposit: Monthly deposit amount (PAC).
    :param annual_return: ANNUAL return rate (e.g., 0.08 for 8%).
    :param months: Total duration of the simulation in months.
    :param fill_months: Number of months the PAC deposit is made.
    :param start_date_str: Start date (YYYY-MM-DD format).
    :return: DataFrame with the simulation results.
    """

    # Initial Variables setup
    capital = initial_capital
    invested = initial_capital  # Total money put in (excluding gains)
    deposit_count = 0

    # Date handling
    start_date = pd.to_datetime(start_date_str) + pd.offsets.MonthEnd(0)

    # Data storage
    records = []

    print(f"Annual Return set: {annual_return:.2%}")
    print("NB: Gains are applied ONLY at the end of each full investment year (annual compounding).\n")

    # Variable to track capital at the start of the year (before PACs)
    capital_start_of_year = initial_capital

    # List to accumulate PAC deposits made within the current year for interest calculation
    pac_deposits_in_year = []

    for month in range(1, months + 1):

        # --- 1. Handle Monthly Deposit (PAC) ---
        is_deposit_month = deposit_count < fill_months

        if is_deposit_month:
            # Update total money put in
            invested += monthly_deposit
            # Update the base capital
            capital += monthly_deposit
            deposit_count += 1

            # Track the deposit for annual interest calculation
            pac_deposits_in_year.append(monthly_deposit)

        # --- 2. Handle Annual Return (Year End) ---
        annual_gain = 0

        # Apply interest after 12, 24, 36 months, etc.
        if month % 12 == 0:

            # --- Annual Gain Calculation ---

            # 1. Interest on Capital at the start of the year (which benefited for the full 12 months)
            gain_on_initial = capital_start_of_year * annual_return

            # 2. Interest on PACs deposited during the year
            # Assumption: PAC money deposited during the year benefited for roughly half a year on average.

            total_pac_this_year = sum(pac_deposits_in_year)
            num_pac_deposits = len(pac_deposits_in_year)

            if num_pac_deposits > 0:
                # We use a simplified factor of 0.5 (representing 6 months on average) of the annual return
                average_investment_factor = num_pac_deposits / 24
                gain_on_pac = total_pac_this_year * annual_return * average_investment_factor
            else:
                gain_on_pac = 0

            # Total annual gain
            annual_gain = round(gain_on_initial + gain_on_pac, 2)

            # Update Total Capital
            capital += annual_gain

            # Prepare for the next year
            capital_start_of_year = capital  # New start-of-year capital is the current capital after gain
            pac_deposits_in_year = []  # Reset PAC deposits list

        # --- 3. Data Recording ---
        date = (pd.to_datetime(start_date_str) + pd.DateOffset(months=month)) + pd.offsets.MonthEnd(0)

        # Calculate Total Return (Net Gain)
        total_return = round(capital - invested, 2)

        records.append({
            "Month": month,
            "Date": date,
            "Invested": round(invested, 2),
            "Capital": round(capital, 2),
            "Accumulated_Capital": round(capital, 2),  # NEW COLUMN: Total Value of the Investment
            "Total_Return": total_return,
            "Annual_Gain": annual_gain,
            "is_deposit_month": is_deposit_month
        })

    return pd.DataFrame(records)

# --- MAIN EXECUTION BLOCK ---


def pac_etoro():
    # Parameters
    initial_capital = 910
    monthly_deposit = 100
    quarterly_return = 0.12
    months = 60
    fill_months= 60
    start_fill_months=0
    # Start date (ultimo giorno del mese iniziale)
    start_date = pd.to_datetime("2025-11-30")

    # Data storage
    records = []
    capital = initial_capital
    invested = initial_capital  # money put in (without gains)

    for month in range(1, months + 1):
        # deposit each month
        start_fill_months+=1
        if start_fill_months < fill_months:
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
            "Quarterly_Gain": gain,
            "filled_month": start_fill_months < fill_months
        })

    df = pd.DataFrame(records)
    print(df)

def test_picpac():
    # Define Variables for Execution (Outside the Function)
    YEARS = 10
    INITIAL_CAPITAL = 20000  # Lump Sum / PIC
    MONTHLY_DEPOSIT = 200  # Monthly Deposit / PAC
    ANNUAL_RETURN = 0.07  # Example: 8% Annual Return
    TOTAL_MONTHS = 12 * YEARS  # Total Duration (5 years)
    FILL_MONTHS = 12 * YEARS  # PAC deposited for 4 years
    START_DATE_STR = "2026-01-01"

    # Execute the Function
    df_result = pic_pac_etf_annual_compounding(
        initial_capital=INITIAL_CAPITAL,
        monthly_deposit=MONTHLY_DEPOSIT,
        annual_return=ANNUAL_RETURN,
        months=TOTAL_MONTHS,
        fill_months=FILL_MONTHS,
        start_date_str=START_DATE_STR
    )

    print(df_result)


def init_pd():
    pd.set_option('display.max_rows', None)  # Nessun limite di righe
    pd.set_option('display.max_columns', None)  # Nessun limite di colonne
    pd.set_option('display.width', 1000)  # Assicura che la console sia abbastanza larga
    pd.set_option('display.max_colwidth', 50)  # Per vedere stringhe più lunghe


if __name__ == "__main__":
    init_pd()
    test_picpac()
    #pac_etoro()
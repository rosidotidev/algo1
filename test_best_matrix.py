import pandas as pd

def build_query_and_filter(win_rate, ret, trades) -> str:
    query = (
        f"df[(df['Win Rate [%]'] > {win_rate}) & "
        f"(df['Return [%]'] > {ret}) & "
        f"(df['# Trades'] >= {trades})]"
    )
    return query

def filter_df_with_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    selected_columns = ['Ticker', 'strategy']
    try:
        filtered_df = eval(query, {"df": df, "pd": pd})
        if not isinstance(filtered_df, pd.DataFrame):
            filtered_df = pd.DataFrame(filtered_df)
        return filtered_df[selected_columns]
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def main():
    file_path = "../results/report.csv"
    df = pd.read_csv(file_path)

    filter_condition = build_query_and_filter(60, 70, 1)
    filtered_matrix = filter_df_with_query(df, filter_condition)
    print(filtered_matrix)

if __name__ == "__main__":
    main()

import warnings
import gradio as gr
import pandas as pd
import stock.dax_pattern_bt as trades
import stock.indicators_signal as ins
import stock.candle_signal as cs
import stock.ticker as ti
import backtrader_util.bu as bu

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def save_cache(stop_loss, take_profit):
    bu.cache["stop_loss"] = stop_loss
    bu.cache["take_profit"] = take_profit
    return "Values saved!"

def run_backtest(ticker,function_name):
    function_name=function_name.replace(" ","_")
    merged_functions_list = cs.candlestick_strategies + ins.indicators_strategy
    functions_dict = {func.__name__: func for func in merged_functions_list}
    func=functions_dict[function_name]
    res=trades.run_backtest_DaxPattern(f"../data/{ticker}.csv", slperc=bu.cache["stop_loss"], tpperc=bu.cache["take_profit"], capital_allocation=1, show_plot=True,
                            target_strategy=func, add_indicators=True)
    return res

def toggle_strategy_filter(full_matrix, only_valid):
    import pandas as pd
    if only_valid:
        return full_matrix[full_matrix["strategy"] != "NO_STRATEGY"].reset_index(drop=True)
    else:
        return full_matrix

def generate_best_matrix(win_rate, ret, trades):
    df = pd.read_csv("../results/report.csv")

    query = (
        f"df[(df['Win Rate [%]'] > {win_rate}) & "
        f"(df['Return [%]'] > {ret}) & "
        f"(df['# Trades'] >= {trades})]"
    )

    try:
        filtered = eval(query, {"df": df, "pd": pd})
        selected = filtered[['Ticker', 'strategy','Win Rate [%]','Return [%]','# Trades']].copy()
    except Exception as e:
        selected = pd.DataFrame({"Error": [str(e)]})

    all_tickers = set(df['Ticker'].unique())
    selected_tickers = set(selected['Ticker'].unique())
    missing = all_tickers - selected_tickers
    missing_df = pd.DataFrame([{'Ticker': t, 'strategy': 'NO_STRATEGY'} for t in missing])

    result_df = pd.concat([selected, missing_df], ignore_index=True).sort_values('Ticker')
    result_df.to_csv("../data/best_matrix.csv", index=False)
    return result_df

def run_long_process(optimize=False):
    result_string=trades.exec_analysis_and_save_results(base_path="./", slperc=bu.cache["stop_loss"], tpperc=bu.cache["take_profit"],optimize=optimize)
    updated_files = bu.get_csv_files("../results/")
    return result_string, gr.update(choices=updated_files)

def load_all_tickers():
    res=ti.fetch_and_save_ticker_data("../data/tickers.txt","../data/","3y")
    return res

def filter_dataframe(query,file_name='report.csv'):
    df = pd.read_csv(f"../results/{file_name}")
    selected_columns = [
        'Ticker',
        'Last Action',
        'Equity Final [$]',
        'Return [%]',
        'Buy & Hold Return [%]',
        '# Trades',
        'Win Rate [%]',
        '_strategy',
        'strategy'
    ]
    try:
        filtered_df = eval(query, {"df": df, "pd": pd})
        if not isinstance(filtered_df, pd.DataFrame):
            filtered_df = pd.DataFrame(filtered_df)
        return filtered_df[selected_columns]
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def top_strategies(file_name='report.csv', metric='Return [%]', top_n=10):
    df = pd.read_csv(f"../results/{file_name}")
    df = df.dropna(subset=['Ticker', 'strategy', 'Return [%]', 'Win Rate [%]', '# Trades'])

    if metric == 'aggregate':
        df['score'] = (df['Return [%]'] * df['Win Rate [%]']).round(2)
        metric = 'score'

    # Group by Ticker and strategy, calculate mean for relevant metrics
    grouped = df.groupby(['Ticker', 'strategy']).agg({
        'Return [%]': 'mean',
        'Win Rate [%]': 'mean',
        '# Trades': 'mean',
        metric: 'mean' if metric != 'score' else 'first'
    }).reset_index()

    if metric == 'score':
        grouped = grouped.rename(columns={'score': 'Score'})
        grouped['Score'] = grouped['Score'].round(2)
        sort_col = 'Score'
    else:
        sort_col = metric

    grouped[['Return [%]', 'Win Rate [%]', '# Trades']] = grouped[['Return [%]', 'Win Rate [%]', '# Trades']].round(2)

    grouped = grouped.sort_values(by=sort_col, ascending=False).head(top_n)
    return grouped



def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Stock Analysis and Ticker Loader")

        with gr.Tabs():
            with gr.TabItem("Results Inspector"):
                gr.Markdown("### Use this section to filter CSV results based on conditions")
                with gr.Row():
                    with gr.Column(scale=1):
                        files = bu.get_csv_files("../results/")
                        file_dropdown = gr.Dropdown(files, label="Select CSV File")

                        win_rate_slider = gr.Slider(minimum=0, maximum=100, value=80, label="Min Win Rate [%]")
                        return_slider = gr.Slider(minimum=0, maximum=500, value=50, label="Min Return [%]")
                        trades_slider = gr.Slider(minimum=0, maximum=100, value=1, label="Min # Trades")
                        last_action_check = gr.CheckboxGroup([1, 2,0,-1,-2], label="Last Action values to include")

                        filter_button = gr.Button("Apply Filter")

                    with gr.Column(scale=4):
                        query_input = gr.Textbox(
                            lines=4,
                            placeholder='Enter a filter, e.g., df[(df["Close"] > 0)]',
                            label="Query",
                            value="""
df[
    (df['Win Rate [%]'] > 80) &
    (df['Return [%]'] > 50) &
    (df['Last Action'].isin([1, 2,0,-1,-2])) &
    (df['# Trades'] > 0)
]"""
                        )
                        df_output = gr.Dataframe(label="Filtered DataFrame")

                        def build_query_and_filter(file_name, win_rate, ret, trades, actions):
                            if not actions:
                                actions = [1, 2]
                            query = f"""df[(df['Win Rate [%]'] > {win_rate}) & (df['Return [%]'] > {ret}) & (df['# Trades'] >= {trades}) & (df['Last Action'].isin({actions}))]"""
                            return filter_dataframe(query, file_name)

                        filter_button.click(build_query_and_filter, inputs=[file_dropdown, win_rate_slider, return_slider, trades_slider, last_action_check], outputs=df_output)

                        execute_query_button = gr.Button("Execute Query")
                        execute_query_button.click(fn=filter_dataframe, inputs=[query_input, file_dropdown], outputs=df_output)

                        file_dropdown.change(fn=filter_dataframe, inputs=[query_input, file_dropdown], outputs=df_output)


            with gr.TabItem("Backtesting"):
                gr.Markdown("### Run a backtest on a specific ticker with a selected trading strategy.")
                tickers = ti.read_tickers_from_file("../data/tickers.txt")
                with gr.Row():
                    ticker_dropdown = gr.Dropdown(choices=tickers, label="Select Ticker")
                    backtesting_functions = ins.indicators_strategy + cs.candlestick_strategies
                    algorithm_choices = [f.__name__.replace("_", " ") for f in backtesting_functions]
                    algorithm_dropdown = gr.Dropdown(choices=algorithm_choices, label="Select Algorithm")
                backtest_output = gr.Textbox(label="Backtesting Results")
                backtest_button = gr.Button("Run Backtest")
                backtest_button.click(run_backtest, inputs=[ticker_dropdown, algorithm_dropdown], outputs=backtest_output)

            with gr.TabItem("Load Tickers"):
                gr.Markdown("### Fetch and store recent historical data for all tickers.")
                tickers_output = gr.Textbox(label="Loaded Tickers")
                load_button = gr.Button("Load Tickers")
                load_button.click(load_all_tickers, outputs=tickers_output)

            with gr.TabItem("Process all strategies"):
                gr.Markdown("### Run all strategies across all tickers and generate daily reports.")
                with gr.Row():
                    with gr.Column(scale=1):
                        stop_loss_input = gr.Number(label="Stop Loss", value=bu.cache.get("stop_loss", 0))
                        take_profit_input = gr.Number(label="Take Profit", value=bu.cache.get("take_profit", 0))
                        save_button = gr.Button("Save")
                        save_output = gr.Textbox(label="Save Status")
                        save_button.click(save_cache, inputs=[stop_loss_input, take_profit_input], outputs=save_output)
                    with gr.Column(scale=5):
                        process_output = gr.Textbox(label="Process strategies")
                        optimize_checkbox = gr.Checkbox(label="Optimize", value=False)
                        long_process_button = gr.Button("Start Long Process")

                        # Modifica run_long_process per accettare un parametro optimize
                        long_process_button.click(
                            run_long_process,
                            inputs=[optimize_checkbox],
                            outputs=[process_output, file_dropdown]
                        )
            with gr.TabItem("Strategy report"):
                gr.Markdown("### Show best strategies using filters")
                with gr.Row():
                    with gr.Accordion( open=True):
                        metric_selector = gr.Dropdown(choices=['Return [%]', 'Win Rate [%]', 'aggregate'],
                                                      value='Return [%]', label="Select Metric")
                        top_button = gr.Button("Show Top Strategies")
                        top_output = gr.Dataframe(label="Top Strategies")
                        top_button.click(top_strategies, inputs=[file_dropdown, metric_selector], outputs=top_output)
            with gr.TabItem("Generate Best Matrix"):
                gr.Markdown("### Generate and Save Best Strategy Matrix")
                matrix_output = gr.State()
                with gr.Row():
                    with gr.Column(scale=1):
                        win_rate_input = gr.Slider(minimum=0, maximum=100, value=60, label="Min Win Rate [%]")
                        return_input = gr.Slider(minimum=0, maximum=500, value=70, label="Min Return [%]")
                        trades_input = gr.Slider(minimum=0, maximum=100, value=1, label="Min # Trades")
                        generate_button = gr.Button("Generate Matrix")
                        show_only_valid_strategies = gr.Checkbox(label="Show only valid strategies", value=True)
                    with gr.Column(scale=2):
                        matrix_output = gr.Dataframe(label="Best Matrix")

                generate_button.click(
                    fn=generate_best_matrix,
                    inputs=[win_rate_input, return_input, trades_input],
                    outputs=matrix_output
                ).then(
                    fn=toggle_strategy_filter,
                    inputs=[matrix_output, show_only_valid_strategies],
                    outputs=matrix_output
                )

                show_only_valid_strategies.change(
                    fn=toggle_strategy_filter,
                    inputs=[matrix_output, show_only_valid_strategies],
                    outputs=matrix_output
                )

    demo.launch(share=True)

if __name__ == '__main__':
    main()

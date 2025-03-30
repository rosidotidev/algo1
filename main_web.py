import gradio as gr
import pandas as pd
import stock.dax_pattern_bt as trades
import stock.indicators_signal as ins
import stock.candle_signal as cs
import stock.ticker as ti
import backtrader_util.bu as bu

def save_cache(stop_loss, take_profit):
    """save values on cache"""
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
# Predefined function to load all tickers
def run_long_process():
    trades.exec_analysis_and_save_results(base_path="./",slperc=bu.cache["stop_loss"], tpperc=bu.cache["take_profit"])
    return "finished"


# Predefined function to load all tickers
def load_all_tickers():
    res=ti.fetch_and_save_ticker_data("../data/tickers.txt","../data/","3y")
    return res


# Function to filter the DataFrame using the provided query
def filter_dataframe(query,file_name='report.csv'):
    # Load the DataFrame from a CSV file (modify "data.csv" with the correct path)
    df = pd.read_csv(f"../results/{file_name}")

    try:
        # Evaluate the query on the DataFrame.
        # The variables 'df' and 'pd' are available to the query
        filtered_df = eval(query, {"df": df, "pd": pd})

        # If the result is not a DataFrame, try to convert it
        if not isinstance(filtered_df, pd.DataFrame):
            filtered_df = pd.DataFrame(filtered_df)
        return filtered_df

    except Exception as e:
        # In case of error, return a DataFrame containing the error message
        return pd.DataFrame({"Error": [str(e)]})


def main():
    # Create an interface using gr.Blocks with two tabs for different functionalities
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Stock Analysis and Ticker Loader")
        gr.Markdown("Use the sections below to filter the DataFrame or to load all tickers.")

        with gr.Tabs():
            with gr.TabItem("Results Inspector"):
                with gr.Row():
                    with gr.Column(scale=1):  # Colonna per il dropdown dei file
                        files = bu.get_csv_files("../results/")
                        file_dropdown = gr.Dropdown(files, label="Select CSV File")

                    with gr.Column(scale=4):
                        df_file_name = None



                        gr.Markdown("### Filter the DataFrame")
                        query_input = gr.Textbox(
                            lines=2,
                            placeholder='Enter a filter, e.g., df[(df["Close"] > 0)]',
                            label="Query", value="""
df[(df['Win Rate [%]'] >= 50)
                & (df['Last Action']==1)
            #   & (df['Ticker'] =='CPR.MI')    
            #   & (df['Equity Final [$]'] >80000)
                & (df['# Trades'] >0 )
        ]"""

                        )
                        df_output = gr.Dataframe(label="Filtered DataFrame")
                        filter_button = gr.Button("Apply Filter")
                        filter_button.click(filter_dataframe, inputs=query_input, outputs=df_output)

                        file_dropdown.change(fn=filter_dataframe, inputs=[query_input,file_dropdown], outputs=df_output)
            with gr.TabItem("Backtesting"):
                gr.Markdown("### Run Backtesting")
                tickers = ti.read_tickers_from_file("../data/tickers.txt")  # Read tickers from file

                with gr.Row():
                    ticker_dropdown = gr.Dropdown(
                        choices=tickers,
                        label="Select Ticker"
                    )
                    backtesting_functions = ins.indicators_strategy + cs.candlestick_strategies
                    algorithm_choices = [f.__name__.replace("_", " ") for f in backtesting_functions]

                    algorithm_dropdown = gr.Dropdown(
                        choices=algorithm_choices,
                        label="Select Algorithm"
                    )

                backtest_output = gr.Textbox(label="Backtesting Results")
                backtest_button = gr.Button("Run Backtest")
                backtest_button.click(run_backtest, inputs=[ticker_dropdown, algorithm_dropdown],
                                      outputs=backtest_output)
            with gr.TabItem("Load Tickers"):
                gr.Markdown("### Load All Tickers")
                tickers_output = gr.Textbox(label="Loaded Tickers")  # Usa gr.Textbox
                load_button = gr.Button("Load Tickers")
                load_button.click(load_all_tickers, outputs=tickers_output)
            with gr.TabItem("Process all strategies"):
                gr.Markdown("### Run Long Process")
                # This textbox will display feedback after the long process completes
                with gr.Row():
                    with gr.Column(scale=1):  # Colonna per i widget di input
                        stop_loss_input = gr.Number(label="Stop Loss",
                                                    value=bu.cache.get("stop_loss", 0))  # Usa cache.get()
                        take_profit_input = gr.Number(label="Take Profit",
                                                      value=bu.cache.get("take_profit", 0))  # Usa cache.get()
                        save_button = gr.Button("Save")
                        save_output = gr.Textbox(label="Save Status")

                        save_button.click(save_cache, inputs=[stop_loss_input, take_profit_input], outputs=save_output)

                    with gr.Column(scale=5):  # Colonna per il pulsante e l'output del processo
                        process_output = gr.Textbox(label="Process strategies")
                        long_process_button = gr.Button("Start Long Process")
                        long_process_button.click(run_long_process, outputs=process_output)

    # Launch the Gradio app
    demo.launch(share=True)


if __name__ == '__main__':
    main()

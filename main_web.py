import gradio as gr
import pandas as pd
import stock.dax_pattern_bt as trades
import stock.ticker as ti

# Predefined function to load all tickers
def long_process():
    trades.exec_analysis_and_save_results()
    return "finished"


# Predefined function to load all tickers
def load_all_tickers():
    res=ti.fetch_and_save_ticker_data("../data/tickers.txt","../data/","3y")
    return res


# Function to filter the DataFrame using the provided query
def filter_dataframe(query: str):
    # Load the DataFrame from a CSV file (modify "data.csv" with the correct path)
    df = pd.read_csv("../results/report.csv")

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
            with gr.TabItem("DataFrame Filter"):
                gr.Markdown("### Filter the DataFrame")
                query_input = gr.Textbox(
                    lines=2,
                    placeholder='Enter a filter, e.g., df[(df["Close"] > 0)]',
                    label="Query"
                )
                df_output = gr.Dataframe(label="Filtered DataFrame")
                filter_button = gr.Button("Apply Filter")
                filter_button.click(filter_dataframe, inputs=query_input, outputs=df_output)

            with gr.TabItem("Load Tickers"):
                gr.Markdown("### Load All Tickers")
                tickers_output = gr.Textbox(label="Loaded Tickers")  # Usa gr.Textbox
                load_button = gr.Button("Load Tickers")
                load_button.click(load_all_tickers, outputs=tickers_output)
            with gr.TabItem("Long Process"):
                gr.Markdown("### Run Long Process")
                # This textbox will display feedback after the long process completes
                process_output = gr.Textbox(label="Process Feedback")
                long_process_button = gr.Button("Start Long Process")
                long_process_button.click(long_process, outputs=process_output)

    # Launch the Gradio app
    demo.launch()


if __name__ == '__main__':
    main()

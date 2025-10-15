import warnings
import gradio as gr
import pandas as pd
import stock.enriched_signal_vec as ins_vec
import stock.simple_signal_vec as cs_vec
import stock.ticker as ti
import backtrader_util.bu as bu
import biz.biz_logic as biz
from stock.strategy_repo import StrategyRepo
import json
import numpy as np
from strategy.strategy_optmizer import optimize_strategy


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def prefill_param_space(strategy_name):

    param_space = bu.suggest_param_space(strategy_name)

    # converte in JSON / dict leggibile
    import json
    return json.dumps(param_space, separators=(",", ":"), indent=None)


def save_cache(stop_loss, take_profit):
    return biz.save_cache(stop_loss, take_profit)

def get_strategy_names():
    return biz.get_strategy_names()

def run_backtest(ticker, function_name):
    return biz.run_backtest(ticker, function_name)

def init_repos(repo):
    repo.init_repo()
    from strategy.ticker_stategy_repo import TickerStrategyRepo
    s_repo = TickerStrategyRepo()
    s_repo.backup()
    s_repo.init_repo()
    s_repo.merge_ticker_strategies()

def toggle_strategy_filter(full_matrix, only_valid):
    if only_valid:
        return full_matrix[full_matrix["strategy"] != "NO_STRATEGY"].reset_index(drop=True)
    else:
        return full_matrix

def generate_best_matrix(win_rate, ret, trades,strategies):
    return biz.generate_best_matrix(win_rate, ret, trades,strategies)

def run_long_process(optimize=False):
    #bu.reset_df_cache()
    result_string, updated_files =biz.run_long_process(optimize=optimize)

    return result_string, gr.update(choices=updated_files)

def load_all_tickers():
    res=ti.fetch_and_save_ticker_data("../data/tickers.txt","../data/","3y")
    return res

def save_selection_callback(ticker_dropdown, strategy_dropdown, params_text):
    return biz.update_ticker_strategy(ticker_dropdown, strategy_dropdown, params_text)


def filter_dataframe(query,file_name='report.csv'):
    df = pd.read_csv(f"../results/{file_name}")
    df=ti.calculate_performance_index(df)
    selected_columns = [
        'Ticker',
        'Last Action',
        '# Trades',
        'Win Rate [%]',
        'Return [%]',
        'strategy',
        'BobIndex',
        'Equity Final [$]',
        'Buy & Hold Return [%]'


    ]
    try:
        filtered_df = eval(query, {"df": df, "pd": pd})
        if not isinstance(filtered_df, pd.DataFrame):
            filtered_df = pd.DataFrame(filtered_df)
        return filtered_df[selected_columns]
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def top_strategies(file_name='report.csv',
    metric='Return [%]',
    min_win_rate=0,
    min_return=0,
    min_trades=0,
    top_n=2000
):
    return biz.top_strategies(file_name=file_name,
    metric=metric,
    min_win_rate=min_win_rate,
    min_return=min_return,
    min_trades=min_trades,
    top_n=top_n)

def fill_ticker_count():
    return ti.count_tickers_in_best_matrix("../data/best_matrix.csv","../data/tickers.txt")


def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Stock Analysis and Ticker Loader")

        with gr.Tabs():
            with gr.TabItem("Results Inspector"):
                gr.Markdown("### Use this section to filter CSV results based on conditions")

                files = bu.get_csv_files("../results/")

                with gr.Row():
                    # --- Left column: filters + query + buttons ---
                    with gr.Column(scale=1):
                        file_dropdown = gr.Dropdown(files, label="Select CSV File")

                        win_rate_slider = gr.Slider(minimum=0, maximum=100, value=80, label="Min Win Rate [%]")
                        return_slider = gr.Slider(minimum=0, maximum=500, value=50, label="Min Return [%]")
                        trades_slider = gr.Slider(minimum=0, maximum=100, value=1, label="Min # Trades")
                        last_action_check = gr.CheckboxGroup([1, 2, 0, -1, -2], label="Last Action values to include",
                                                             value=[1, 2, -1, -2])
                        filter_button = gr.Button("Apply Filter")



                    # --- Right column: output DataFrame ---
                    with gr.Column(scale=5):
                        df_output = gr.Dataframe(label="Filtered DataFrame")
                        # Query input
                        gr.Markdown("### Run pandas query using df as Dataframe variable.")
                        query_input = gr.Textbox(
                            lines=4,
                            placeholder='Enter a filter, e.g., df[(df["Close"] > 0)]',
                            label="Query",
                            value="""df[
    (df['Win Rate [%]'] > 80) &
    (df['Return [%]'] > 50) &
    (df['Last Action'].isin([1, 2,0,-1,-2])) &
    (df['# Trades'] > 0)
]"""
                        )

                        execute_query_button = gr.Button("Execute Query")
                # --- Callbacks ---
                def build_query_and_filter(file_name, win_rate, ret, trades, actions):
                    if not actions:
                        actions = [1, 2]
                    query = f"""df[(df['Win Rate [%]'] > {win_rate}) & (df['Return [%]'] > {ret}) & (df['# Trades'] >= {trades}) & (df['Last Action'].isin({actions}))]"""
                    return filter_dataframe(query, file_name)

                filter_button.click(
                    build_query_and_filter,
                    inputs=[file_dropdown, win_rate_slider, return_slider, trades_slider, last_action_check],
                    outputs=df_output
                )

                execute_query_button.click(
                    fn=filter_dataframe,
                    inputs=[query_input, file_dropdown],
                    outputs=df_output
                )

                file_dropdown.change(
                    fn=filter_dataframe,
                    inputs=[query_input, file_dropdown],
                    outputs=df_output
                )

            with gr.TabItem("Backtesting"):
                gr.Markdown("### Run a backtest on a specific ticker with a selected trading strategy.")

                tickers = ti.read_tickers_from_file("../data/tickers.txt")
                backtesting_functions = ins_vec.indicators_strategy + cs_vec.candlestick_strategies
                algorithm_choices = [f.__name__.replace("_", " ") for f in backtesting_functions]

                with gr.Row():
                    # --- Left column: filters and button ---
                    with gr.Column(scale=1):
                        ticker_dropdown = gr.Dropdown(choices=tickers, label="Select Ticker")
                        algorithm_dropdown = gr.Dropdown(choices=algorithm_choices, label="Select Algorithm")
                        backtest_button = gr.Button("Run Backtest")

                    # --- Right column: output ---
                    with gr.Column(scale=3):
                        backtest_output = gr.Textbox(label="Backtesting Results")

                # --- Connect button ---
                backtest_button.click(
                    run_backtest,
                    inputs=[ticker_dropdown, algorithm_dropdown],
                    outputs=backtest_output
                )

            with gr.TabItem("Manage Tickers"):
                gr.Markdown("### Fetch and store recent historical data for all tickers.")
                with gr.Row():
                    with gr.Column(scale=1):
                        tickers_output = gr.Textbox(label="Loaded Tickers")
                        load_button = gr.Button("Read Tickers OHLC Data")
                        load_button.click(load_all_tickers, outputs=tickers_output)
                    with gr.Column(scale=1):
                        gr.Markdown("### Ticker Frequency in Best Matrix")
                        count_tickers_button = gr.Button("Count Ticker Frequency")
                        ticker_count_output = gr.Dataframe(label="Ticker Frequency")
                        count_tickers_button.click(fn=fill_ticker_count, outputs=ticker_count_output)


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
                        optimize_checkbox = gr.Checkbox(label="Optimize", value=True)
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
                    # Colonna sinistra: filtro e pulsante
                    with gr.Column(scale=1):
                        with gr.Accordion(open=True):
                            metric_selector = gr.Dropdown(
                                choices=['Return [%]', 'Win Rate [%]', 'aggregate'],
                                value='Return [%]',
                                label="Select Metric"
                            )
                            win_rate_slider = gr.Slider(minimum=0, maximum=100, value=80, label="Min Win Rate [%]")
                            return_slider = gr.Slider(minimum=0, maximum=500, value=50, label="Min Return [%]")
                            trades_slider = gr.Slider(minimum=0, maximum=100, value=1, label="Min # Trades")
                            top_button = gr.Button("Show Top Strategies")

                    # Colonna destra: tabella risultati
                    with gr.Column(scale=3):
                        top_output = gr.Dataframe(label="Top Strategies")

                # Callback
                top_button.click(
                    top_strategies,
                    inputs=[file_dropdown, metric_selector, win_rate_slider, return_slider, trades_slider],
                    outputs=top_output
                )
            with gr.TabItem("Generate Best Matrix"):
                gr.Markdown("### Generate and Save Best Strategy Matrix")
                matrix_output = gr.State()
                with gr.Row():
                    with gr.Column(scale=1):
                        win_rate_input = gr.Slider(minimum=0, maximum=100, value=60, label="Min Win Rate [%]")
                        return_input = gr.Slider(minimum=0, maximum=500, value=50, label="Min Return [%]")
                        trades_input = gr.Slider(minimum=0, maximum=100, value=5, label="Min # Trades")
                        generate_button = gr.Button("Generate Matrix")
                        show_only_valid_strategies = gr.Checkbox(label="Show only valid strategies", value=True)

                        strategy_choices = get_strategy_names()
                        strategy_filter = gr.CheckboxGroup(
                            label="Filter by Strategy",
                            choices=strategy_choices,
                            value=strategy_choices
                        )

                    with gr.Column(scale=2):
                        matrix_output = gr.Dataframe(label="Best Matrix")

                generate_button.click(
                    fn=generate_best_matrix,
                    inputs=[win_rate_input, return_input, trades_input,strategy_filter],
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
            with gr.TabItem("Manage Strategies"):
                with gr.Tabs():  # <-- contenitore dei sottotab
                    with gr.TabItem("Strategy Enablement"):  # <-- il tuo primo sotto-tab
                        # --- tutto il codice attuale qui dentro ---

                        repo = StrategyRepo()

                        def get_df_for_display():
                            df = repo.get_all_strategies().copy()
                            if "function_ref" in df.columns:
                                df = df.drop(columns=["function_ref"])
                            return df

                        with gr.Row():
                            # Left column: actions
                            with gr.Column(scale=1):
                                init_button = gr.Button("Initialize Repository")

                                strategy_dropdown = gr.Dropdown(
                                    choices=repo.get_all_strategies_as_list(),
                                    label="Select Strategy"
                                )

                                enable_button = gr.Button("Enable Strategy")
                                disable_button = gr.Button("Disable Strategy")
                                save_button = gr.Button("Save Changes")
                                enable_all_button = gr.Button("Enable All Strategies")
                                disable_all_button = gr.Button("Disable All Strategies")

                                message_output = gr.Textbox(label="Status Message")

                            # Right column: display current DataFrame
                            with gr.Column(scale=3):
                                strategies_df = gr.Dataframe(
                                    value=get_df_for_display(),
                                    label="Current Strategies",
                                    interactive=True
                                )

                        # --- Callback functions come prima ---
                        def enable_all_callback():
                            repo.enable_all()
                            return get_df_for_display(), "All strategies enabled."

                        def disable_all_callback():
                            repo.disable_all()
                            return get_df_for_display(), "All strategies disabled."

                        def init_repo():
                            init_repos(repo)
                            return get_df_for_display(), repo.get_all_strategies_as_list(), "Repository initialized."

                        def enable_strategy(name):
                            if not name:
                                return repo.get_all_strategies(), "No strategy selected."
                            repo.enable(name)
                            return get_df_for_display(), f"Strategy '{name}' enabled."

                        def disable_strategy(name):
                            if not name:
                                return repo.get_all_strategies(), "No strategy selected."
                            repo.disable(name)
                            return get_df_for_display(), f"Strategy '{name}' disabled."

                        def save_changes_callback(df):
                            df = df.copy()
                            if "enabled" in df.columns:
                                df["enabled"] = df["enabled"].astype(bool)
                            if "score" in df.columns:
                                df["score"] = pd.to_numeric(df["score"], errors="coerce")
                            repo._df = df
                            repo.save()
                            return get_df_for_display(), "Changes saved."

                        enable_all_button.click(
                            enable_all_callback,
                            inputs=[],
                            outputs=[strategies_df, message_output]
                        )
                        disable_all_button.click(
                            disable_all_callback,
                            inputs=[],
                            outputs=[strategies_df, message_output]
                        )
                        save_button.click(
                            save_changes_callback,
                            inputs=[strategies_df],
                            outputs=[strategies_df, message_output]
                        )
                        init_button.click(
                            init_repo,
                            inputs=[],
                            outputs=[strategies_df, strategy_dropdown, message_output]
                        )
                        enable_button.click(
                            enable_strategy,
                            inputs=[strategy_dropdown],
                            outputs=[strategies_df, message_output]
                        )
                        disable_button.click(
                            disable_strategy,
                            inputs=[strategy_dropdown],
                            outputs=[strategies_df, message_output]
                        )

                    with gr.TabItem("Optimization"):
                        gr.Markdown("### Run parameter optimization for a specific ticker and strategy.")

                        tickers = ti.read_tickers_from_file("../data/tickers.txt")
                        strategies = ins_vec.indicators_strategy + cs_vec.candlestick_strategies
                        strategy_choices = [f.__name__.replace("_", " ") for f in strategies]

                        with gr.Row():
                            # --- Left column: inputs ---
                            with gr.Column(scale=1):
                                ticker_dropdown = gr.Dropdown(
                                    choices=tickers,
                                    label="Select Ticker"
                                )
                                strategy_dropdown = gr.Dropdown(
                                    choices=strategy_choices,
                                    label="Select Strategy"
                                )

                                # Prefilled param_space text area (editable)
                                params_text = gr.Textbox(
                                    label="Parameter Space (Python dict)",
                                    lines=3,
                                    value="{}"
                                )

                                optimize_button = gr.Button("Run Optimization")

                                best_params = gr.Textbox(
                                    label="Best params",
                                    lines=3,
                                    value="{}"
                                )
                                save_selection_button = gr.Button("Save Selected as Best")
                                save_feedback = gr.Textbox(label="Save Feedback", interactive=False)
                                save_selection_button.click(
                                    save_selection_callback,
                                    inputs=[ticker_dropdown, strategy_dropdown, best_params],
                                    outputs=[save_feedback]
                                )

                            # --- Right column: results ---
                            with gr.Column(scale=3):
                                optimization_results = gr.Dataframe(
                                    label="Optimization Results",
                                    interactive=True
                                )
                                best_params_output = gr.Textbox(
                                    label="Best Parameters"
                                )

                        # --- Callback ---
                        def run_optimization_ui(ticker, strategy_name, params_text):
                            # Recupera la funzione dalla lista delle strategie
                            strategies = ins_vec.indicators_strategy + cs_vec.candlestick_strategies
                            strategy_map = {f.__name__.replace("_", " "): f for f in strategies}
                            func = strategy_map.get(strategy_name)
                            if func is None:
                                return {}, f"Strategy '{strategy_name}' not found."

                            # Legge il JSON dalla Textbox
                            try:
                                param_dict = json.loads(params_text)
                            except json.JSONDecodeError as e:
                                return {}, f"Errore nel JSON: {e}"

                            # Converte [min,max,step] in np.arange
                            param_space = {}
                            for k, v in param_dict.items():
                                if isinstance(v, list) and len(v) == 3:
                                    param_space[k] = np.arange(v[0], v[1], v[2])
                                else:
                                    return {}, f"Parametro '{k}' non valido: deve essere [min,max,step]"

                            # Chiama la funzione di ottimizzazione
                            df_results, best_portfolio = optimize_strategy(
                                ticker=ticker,
                                func=func,
                                param_space=param_space
                            )

                            return df_results, str(best_portfolio)

                        optimize_button.click(
                            run_optimization_ui,
                            inputs=[ticker_dropdown, strategy_dropdown, params_text],
                            outputs=[optimization_results, best_params_output]
                        )


                        strategy_dropdown.change(
                            prefill_param_space,  # funzione che calcola param_space
                            inputs=[strategy_dropdown],  # input: nome della strategy selezionata
                            outputs=[params_text]  # output: aggiorna la textbox
                        )

    demo.launch(share=True)

if __name__ == '__main__':
    main()

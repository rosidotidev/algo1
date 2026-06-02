from datetime import date
import warnings
import streamlit as st
import pandas as pd
import stock.enriched_signal_vec as ins_vec
import stock.portfolio_manager as pm
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Stock Analysis", layout="wide")
st.title("Stock Analysis and Ticker Loader")

# ---------------------------------------------------------------------------
# Helper functions (ported from main_web.py, no existing files modified)
# ---------------------------------------------------------------------------
def get_strategy_names():
    return biz.get_strategy_names()


def run_backtest_ui(ticker, function_name):
    return biz.run_backtest(ticker, function_name,
                            bu.cache["enabled_long"],
                            bu.cache["enabled_short"])


def init_repos(repo):
    repo.init_repo()
    from strategy.ticker_stategy_repo import TickerStrategyRepo
    s_repo = TickerStrategyRepo()
    s_repo.backup()
    s_repo.init_repo()
    s_repo.merge_ticker_strategies()


def filter_dataframe(query, file_name="report.csv"):
    df = pd.read_csv(f"../results/{file_name}")
    df = ti.calculate_performance_index(df)
    selected_columns = [
        "Ticker", "Last Action", "# Trades", "Win Rate [%]",
        "Return [%]", "strategy", "BobIndex", "Equity Final [$]",
        "Buy & Hold Return [%]",
    ]
    try:
        filtered_df = eval(query, {"df": df, "pd": pd})
        if not isinstance(filtered_df, pd.DataFrame):
            filtered_df = pd.DataFrame(filtered_df)
        return filtered_df[selected_columns]
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def load_all_tickers():
    return ti.fetch_and_save_ticker_data("../data/tickers.txt", "../data/", "6y")


def prefill_param_space(strategy_name):
    param_space = bu.suggest_param_space(strategy_name)
    return json.dumps(param_space, separators=(",", ":"), indent=None)


def toggle_strategy_filter(full_matrix, only_valid):
    if only_valid:
        return full_matrix[full_matrix["strategy"] != "NO_STRATEGY"].reset_index(drop=True)
    return full_matrix


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "report_files" not in st.session_state:
    st.session_state.report_files = bu.get_csv_files("../results/")

if "selected_report_file" not in st.session_state:
    sf = st.session_state.report_files
    st.session_state.selected_report_file = sf[0] if sf else "report.csv"

if "generate_matrix_df" not in st.session_state:
    st.session_state.generate_matrix_df = None

if "opt_params" not in st.session_state:
    st.session_state.opt_params = "{}"

if "best_params_str" not in st.session_state:
    st.session_state.best_params_str = ""

if "prev_opt_strategy" not in st.session_state:
    st.session_state.prev_opt_strategy = ""

if "last_filtered_results" not in st.session_state:
    st.session_state.last_filtered_results = None

# ---------------------------------------------------------------------------
# Shared data that all tabs can reference
# ---------------------------------------------------------------------------
tickers = ti.read_tickers_from_file("../data/tickers.txt")
all_strategy_functions = ins_vec.indicators_strategy + cs_vec.candlestick_strategies
strategy_choices = [f.__name__.replace("_", " ") for f in all_strategy_functions]

# ============================================================================
# TABS  (native Streamlit tabs  —  more compact, similar to Gradio)
# ============================================================================
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none !important; }
    .block-container { padding-top: 0.5rem !important; }
    h1 { margin: 0 !important; padding: 0 !important; font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

TAB_NAMES = [
    "Results Inspector", "Backtesting", "Manage Tickers",
    "Process all strategies", "Strategy report",
    "Generate Best Matrix", "Manage Strategies",
    "📊 Portfolio",
]
tabs = st.tabs(TAB_NAMES)

# ---------------------------------------------------------------------------
# TAB 1 — Results Inspector
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Use this section to filter CSV results based on conditions")

    left, right = st.columns([1, 5])

    with left:
        current_index = 0
        if st.session_state.selected_report_file in st.session_state.report_files:
            current_index = st.session_state.report_files.index(
                st.session_state.selected_report_file
            )
        sel_file = st.selectbox(
            "Select CSV File",
            options=st.session_state.report_files,
            index=current_index,
            key="ri_file",
        )
        st.session_state.selected_report_file = sel_file

        ri_wr = st.slider("Min Win Rate [%]", 0, 100, 50, key="ri_wr")
        ri_ret = st.slider("Min Return [%]", 0, 500, 20, key="ri_ret")
        ri_tr = st.slider("Min # Trades", 0, 100, 1, key="ri_tr")
        ri_la = st.multiselect(
            "Last Action values to include",
            [1, 2, 0, -1, -2],
            default=[1, 2, -1, -2],
            key="ri_la",
        )

        filter_btn = st.button("Apply Filter", key="ri_filter_btn")

    with right:
        # Always show filtered results if available
        if st.session_state.last_filtered_results is not None:
            st.dataframe(
                st.session_state.last_filtered_results,
                use_container_width=True,
                key="ri_results_grid",
            )

            st.divider()
            st.markdown("### ⚡ Quick Backtest")

            df_qb = st.session_state.last_filtered_results
            if not df_qb.empty:
                def make_label(row):
                    t = row.get("Ticker", "?")
                    s = row.get("strategy", "?")
                    wr = row.get("Win Rate [%]", "?")
                    rt = row.get("Return [%]", "?")
                    return f"{t} — {s}  (WR={wr}%, Ret={rt}%)"

                label_options = [make_label(df_qb.iloc[i]) for i in range(len(df_qb))]
                selected_idx = st.selectbox(
                    "Select row",
                    options=range(len(df_qb)),
                    format_func=lambda i: label_options[i],
                    key="qb_idx",
                )

                qb_row = df_qb.iloc[selected_idx]
                qb_ticker = qb_row.get("Ticker", "")
                qb_strategy = qb_row.get("strategy", "")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Ticker", qb_ticker)
                m2.metric("Strategy", qb_strategy)
                m3.metric("Win Rate [%]", qb_row.get("Win Rate [%]", "—"))
                m4.metric("Return [%]", qb_row.get("Return [%]", "—"))

                if st.button("▶ Run Backtest", key="qb_run"):
                    with st.spinner(f"Running backtest for {qb_ticker} / {qb_strategy}..."):
                        qb_result = run_backtest_ui(qb_ticker, qb_strategy)
                    with st.expander("Backtest Result", expanded=True):
                        st.text(qb_result)
            else:
                st.info("💡 Apply a filter on the left.")
        else:
            st.info("💡 Apply a filter on the left to see results and run backtests.")

    # Apply filter logic (updates session state, display is handled above)
    if filter_btn:
        actions = ri_la if ri_la else [1, 2]
        q = (
            f"df[(df['Win Rate [%]'] > {ri_wr}) & "
            f"(df['Return [%]'] > {ri_ret}) & "
            f"(df['# Trades'] >= {ri_tr}) & "
            f"(df['Last Action'].isin({actions}))]"
        )
        result = filter_dataframe(q, st.session_state.selected_report_file)
        st.session_state.last_filtered_results = result

# ---------------------------------------------------------------------------
# TAB 2 — Backtesting
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### Run a backtest on a specific ticker with a selected trading strategy.")

    col1, col2 = st.columns([1, 3])

    with col1:
        bt_ticker = st.selectbox("Select Ticker", tickers, key="bt_ticker")
        bt_algo = st.selectbox("Select Algorithm", strategy_choices, key="bt_algo")
        bt_run = st.button("Run Backtest")

    with col2:
        bt_out = st.empty()

    if bt_run:
        with st.spinner("Running backtest..."):
            result = run_backtest_ui(bt_ticker, bt_algo)
            bt_out.text(result)

# ---------------------------------------------------------------------------
# TAB 3 — Manage Tickers
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Fetch and store recent historical data for all tickers.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Read Tickers OHLC Data"):
            with st.spinner("Fetching ticker data..."):
                result = load_all_tickers()
                st.text(result)

    with col2:
        st.markdown("### Ticker Frequency in Best Matrix")
        if st.button("Count Ticker Frequency"):
            with st.spinner("Counting..."):
                df_count = ti.count_tickers_in_best_matrix(
                    "../data/best_matrix.csv", "../data/tickers.txt"
                )
                st.dataframe(df_count, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4 — Process all strategies
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Run all strategies across all tickers and generate daily reports.")

    left, right = st.columns([1, 5])

    with left:
        sl_val = st.number_input(
            "Stop Loss",
            value=bu.cache.get("stop_loss", 0.05),
            format="%.4f",
            key="pl_sl",
        )
        tp_val = st.number_input(
            "Take Profit",
            value=bu.cache.get("take_profit", 0.08),
            format="%.4f",
            key="pl_tp",
        )
        en_long = st.checkbox(
            "Enable Long",
            value=bu.cache.get("enabled_long", True),
            key="pl_long",
        )
        en_short = st.checkbox(
            "Enable Short",
            value=bu.cache.get("enabled_short", True),
            key="pl_short",
        )

        if st.button("Save", key="pl_save"):
            biz.save_cache(sl_val, tp_val, en_long, en_short)
            st.success("Values saved!")

    with right:
        optimize_val = st.checkbox("Optimize", value=True, key="pl_opt")

        if st.button("Start Long Process"):
            with st.spinner(
                "Processing all strategies across all tickers... This may take a while."
            ):
                result_string, updated_files = biz.run_long_process(
                    optimize=st.session_state.pl_opt,
                )
                st.session_state.report_files = updated_files
                st.text(result_string)
            st.success("Process completed!")

# ---------------------------------------------------------------------------
# TAB 5 — Strategy report
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### Show best strategies using filters")

    left, right = st.columns([1, 3])

    with left:
        with st.expander("Filters", expanded=True):
            sr_metric = st.selectbox(
                "Select Metric",
                ["Return [%]", "Win Rate [%]", "aggregate"],
                key="sr_metric",
            )
            sr_wr = st.slider("Min Win Rate [%]", 0, 100, 80, key="sr_wr")
            sr_ret = st.slider("Min Return [%]", 0, 500, 50, key="sr_ret")
            sr_tr = st.slider("Min # Trades", 0, 100, 1, key="sr_tr")
            sr_btn = st.button("Show Top Strategies")

    with right:
        sr_out = st.empty()

    if sr_btn:
        with st.spinner("Loading top strategies..."):
            result = biz.top_strategies(
                file_name=st.session_state.selected_report_file,
                metric=sr_metric,
                min_win_rate=sr_wr,
                min_return=sr_ret,
                min_trades=sr_tr,
            )
            sr_out.dataframe(result, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 6 — Generate Best Matrix
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("### Generate and Save Best Strategy Matrix")

    left, right = st.columns([1, 2])

    with left:
        gm_wr = st.slider("Min Win Rate [%]", 0, 100, 60, key="gm_wr")
        gm_ret = st.slider("Min Return [%]", 0, 500, 50, key="gm_ret")
        gm_tr = st.slider("Min # Trades", 0, 100, 5, key="gm_tr")

        all_strat_names = get_strategy_names()
        gm_strategies = st.multiselect(
            "Filter by Strategy",
            options=all_strat_names,
            default=all_strat_names,
            key="gm_strategies",
        )

        gm_btn = st.button("Generate Matrix")
        show_valid_only = st.checkbox(
            "Show only valid strategies", value=True, key="gm_valid"
        )

    with right:
        gm_out = st.empty()

    if gm_btn:
        with st.spinner("Generating best matrix..."):
            df_matrix = biz.generate_best_matrix(
                gm_wr, gm_ret, gm_tr, gm_strategies
            )
            st.session_state.generate_matrix_df = df_matrix

    if st.session_state.generate_matrix_df is not None:
        display_df = toggle_strategy_filter(
            st.session_state.generate_matrix_df, show_valid_only
        )
        gm_out.dataframe(display_df, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 7 — Manage Strategies (nested sub-tabs)
# ---------------------------------------------------------------------------
with tabs[6]:
    subtabs = st.tabs(["Strategy Enablement", "Optimization"])

    # --- Subtab 7-A: Strategy Enablement ---
    with subtabs[0]:
        repo = StrategyRepo()

        def get_df_for_display():
            df = repo.get_all_strategies().copy()
            if "function_ref" in df.columns:
                df = df.drop(columns=["function_ref"])
            return df

        left, right = st.columns([1, 3])

        with left:
            if st.button("Initialize Repository", key="se_init"):
                with st.spinner("Initializing..."):
                    init_repos(repo)
                    st.success("Repository initialized.")
                    st.rerun()

            strategy_names = repo.get_all_strategies_as_list()
            se_strategy = st.selectbox(
                "Select Strategy",
                options=strategy_names if strategy_names else [""],
                key="se_strategy",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Enable Strategy", key="se_enable"):
                    if se_strategy:
                        repo.enable(se_strategy)
                        st.success(f"Strategy '{se_strategy}' enabled.")
                        st.rerun()
            with col_b:
                if st.button("Disable Strategy", key="se_disable"):
                    if se_strategy:
                        repo.disable(se_strategy)
                        st.success(f"Strategy '{se_strategy}' disabled.")
                        st.rerun()

            if st.button("Enable All Strategies", key="se_enable_all"):
                repo.enable_all()
                st.success("All strategies enabled.")
                st.rerun()

            if st.button("Disable All Strategies", key="se_disable_all"):
                repo.disable_all()
                st.success("All strategies disabled.")
                st.rerun()

        with right:
            display_df = get_df_for_display()
            edited_df = st.data_editor(
                display_df, use_container_width=True, key="se_editor"
            )

            if st.button("Save Changes", key="se_save"):
                df_save = edited_df.copy()
                if "enabled" in df_save.columns:
                    df_save["enabled"] = df_save["enabled"].astype(bool)
                if "score" in df_save.columns:
                    df_save["score"] = pd.to_numeric(df_save["score"], errors="coerce")
                repo._df = df_save
                repo.save()
                st.success("Changes saved.")
                st.rerun()

    # --- Subtab 7-B: Optimization ---
    with subtabs[1]:
        st.markdown(
            "### Run parameter optimization for a specific ticker and strategy."
        )

        left, right = st.columns([1, 3])

        with left:
            opt_ticker = st.selectbox("Select Ticker", tickers, key="opt_ticker")
            opt_strategy = st.selectbox("Select Strategy", strategy_choices, key="opt_strategy")

            # Prefill param space when the user picks a different strategy
            if opt_strategy != st.session_state.prev_opt_strategy:
                st.session_state.prev_opt_strategy = opt_strategy
                try:
                    st.session_state.opt_params = prefill_param_space(opt_strategy)
                except ValueError:
                    st.session_state.opt_params = "{}"

            st.text_area(
                "Parameter Space (Python dict)",
                key="opt_params",
                height=100,
            )

            opt_run = st.button("Run Optimization")

            if st.button("Save Selected as Best", key="opt_save"):
                result = biz.update_ticker_strategy(
                    opt_ticker,
                    opt_strategy,
                    st.session_state.best_params_str,
                )
                st.success(result)

        with right:
            opt_results_out = st.empty()
            st.markdown("**Best Parameters**")
            best_params_display = st.empty()

            if opt_run:
                with st.spinner("Running optimization..."):
                    strategies_map = {
                        f.__name__.replace("_", " "): f
                        for f in all_strategy_functions
                    }
                    func = strategies_map.get(opt_strategy)

                    if func is None:
                        st.error(f"Strategy '{opt_strategy}' not found.")
                    else:
                        try:
                            param_dict = json.loads(st.session_state.opt_params)
                        except json.JSONDecodeError as e:
                            st.error(f"JSON error: {e}")
                            param_dict = None

                        if param_dict:
                            param_space = {}
                            valid = True
                            for k, v in param_dict.items():
                                if isinstance(v, list) and len(v) == 3:
                                    param_space[k] = np.arange(v[0], v[1], v[2])
                                else:
                                    st.error(
                                        f"Parameter '{k}' invalid: must be [min, max, step]"
                                    )
                                    valid = False
                                    break

                            if valid:
                                df_results, best_portfolio = optimize_strategy(
                                    ticker=opt_ticker,
                                    func=func,
                                    param_space=param_space,
                                )
                                opt_results_out.dataframe(
                                    df_results, use_container_width=True
                                )
                                best_params_str = (
                                    str(best_portfolio)
                                    if best_portfolio is not None
                                    else "{}"
                                )
                                st.session_state.best_params_str = best_params_str
                                best_params_display.text(best_params_str)

# ---------------------------------------------------------------------------
# TAB 8 — 📊 Portfolio
# ---------------------------------------------------------------------------
with tabs[7]:
    pm.init_portfolio()

    st.markdown("### Record your real trades and monitor your portfolio")

    left, right = st.columns([1.3, 3])

    with left:
        st.markdown("#### ➕ Open a New Position")
        with st.form("trade_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                pf_ticker = st.text_input("Ticker", placeholder="e.g. AAPL", key="pf_ticker")
            with col2:
                pf_amount = st.number_input("Amount (€)", min_value=0.0, step=10.0,
                                            format="%.2f", key="pf_amount",
                                            help="Total € spent (Buy=long) or received (Sell=short)")
            with col3:
                pf_action = st.selectbox("Direction", ["Buy (Long)", "Sell (Short)"], key="pf_action")

            col1b, col2b, col3b = st.columns(3)
            with col1b:
                pf_leverage = st.selectbox("Leverage", [1, 2, 3], key="pf_leverage")
            with col2b:
                pf_fee = st.number_input("Fee (€)", min_value=0.0, step=0.5,
                                         format="%.2f", key="pf_fee")
            with col3b:
                pf_date = st.date_input("Date", value=date.today(), key="pf_date")

            pf_note = st.text_input("Notes", placeholder="optional", key="pf_note")

            submitted = st.form_submit_button("➕ Open Trade", type="primary",
                                              use_container_width=True)

            if submitted:
                if not pf_ticker.strip():
                    st.error("Please enter a ticker.")
                elif pf_amount <= 0:
                    st.error("Amount must be greater than 0.")
                else:
                    # Parse "Buy (Long)" → "Buy", "Sell (Short)" → "Sell"
                    raw_action = pf_action.split()[0]
                    pm.add_trade(
                        ticker=pf_ticker.strip(),
                        action=raw_action,
                        amount=pf_amount,
                        leverage=pf_leverage,
                        entry_date=pf_date.isoformat(),
                        fee=pf_fee,
                        note=pf_note.strip(),
                    )
                    exposure = pf_amount * pf_leverage
                    st.success(f"Opened {raw_action} {pf_ticker.upper()} "
                               f"€{pf_amount:.2f} (×{pf_leverage} = €{exposure:.2f})")
                    st.rerun()

        st.divider()
        st.markdown("#### 💰 Cash Operations")
        with st.form("cash_op_form", clear_on_submit=True):
            co1, co2, co3 = st.columns(3)
            with co1:
                co_action = st.selectbox("Type", ["Deposit", "Withdrawal"], key="co_action")
            with co2:
                co_amount = st.number_input("Amount (€)", min_value=0.0, step=50.0,
                                            format="%.2f", key="co_amount",
                                            help="Positive = deposit, negative = withdrawal")
            with co3:
                co_date = st.date_input("Date", value=date.today(), key="co_date")

            co_note = st.text_input("Notes", placeholder="e.g. bank transfer", key="co_note")
            co_submitted = st.form_submit_button("💳 Record Cash Operation",
                                                 type="secondary", use_container_width=True)
            if co_submitted:
                amount = co_amount if co_action == "Deposit" else -co_amount
                pm.add_cash_operation(
                    amount=amount,
                    entry_date=co_date.isoformat(),
                    note=co_note.strip(),
                )
                st.success(f"{co_action} of €{co_amount:.2f} recorded!")
                st.rerun()

    with right:
        # --- Dashboard style ---
        st.markdown("""
        <style>
        .dash-card {
            background: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-radius: 10px;
            padding: 12px 10px 8px 10px;
            margin-bottom: 8px;
            text-align: center;
            min-height: 80px;
        }
        .dash-label {
            font-size: 0.75rem;
            color: #9e9eb8;
            margin-bottom: 4px;
        }
        .dash-value {
            font-size: 1.2rem;
            font-weight: 700;
            color: #ffffff;
        }
        .dash-positive { color: #00c853; }
        .dash-negative { color: #ff5252; }
        .dash-row {
            display: flex;
            gap: 10px;
            margin-bottom: 6px;
        }
        .dash-row > div {
            flex: 1;
            min-width: 0;
        }
        </style>
        """, unsafe_allow_html=True)

        summary = pm.get_portfolio_summary()
        pnl = summary['realized_pnl']
        pnl_class = "dash-positive" if pnl > 0 else ("dash-negative" if pnl < 0 else "")

        # Row 1 — 4 main metrics
        html_row1 = f"""
        <div class="dash-row">
            <div class="dash-card">
                <div class="dash-label">💰 Cash Balance</div>
                <div class="dash-value">€{summary['cash_balance']:,.2f}</div>
            </div>
            <div class="dash-card">
                <div class="dash-label">📈 Positions Exposure</div>
                <div class="dash-value">€{summary['positions_exposure']:,.2f}</div>
            </div>
            <div class="dash-card">
                <div class="dash-label">🏦 Total Portfolio</div>
                <div class="dash-value">€{summary['portfolio_total']:,.2f}</div>
            </div>
            <div class="dash-card">
                <div class="dash-label">💹 Realized P&L</div>
                <div class="dash-value {pnl_class}">{f"+€{pnl:,.2f}" if pnl > 0 else (f"-€{abs(pnl):,.2f}" if pnl < 0 else "€0.00")}</div>
            </div>
        </div>
        """

        exclude_deps = st.checkbox(
            "Exclude deposits/withdrawals", key="pf_exclude_deps",
            help="When checked, ROI is calculated on trades only",
        )
        stats = pm.get_advanced_stats(exclude_deposits=exclude_deps)

        # Row 2 — 4 metrics: Month / Year / 12M / Win Rate
        m_cls = "dash-positive" if stats['pnl_month'] >= 0 else "dash-negative"
        y_cls = "dash-positive" if stats['pnl_year'] >= 0 else "dash-negative"
        y12_cls = "dash-positive" if stats['pnl_12m'] >= 0 else "dash-negative"

        html_row2 = f"""
        <div class="dash-row">
            <div class="dash-card">
                <div class="dash-label">📅 P&L This Month</div>
                <div class="dash-value {m_cls}">{f"+€{stats['pnl_month']:,.2f}" if stats['pnl_month'] > 0 else (f"-€{abs(stats['pnl_month']):,.2f}" if stats['pnl_month'] < 0 else "€0.00")}</div>
            </div>
            <div class="dash-card">
                <div class="dash-label">📅 P&L This Year</div>
                <div class="dash-value {y_cls}">{f"+€{stats['pnl_year']:,.2f}" if stats['pnl_year'] > 0 else (f"-€{abs(stats['pnl_year']):,.2f}" if stats['pnl_year'] < 0 else "€0.00")}</div>
            </div>
            <div class="dash-card">
                <div class="dash-label">📅 P&L Last 12M</div>
                <div class="dash-value {y12_cls}">{f"+€{stats['pnl_12m']:,.2f}" if stats['pnl_12m'] > 0 else (f"-€{abs(stats['pnl_12m']):,.2f}" if stats['pnl_12m'] < 0 else "€0.00")}</div>
            </div>
            <div class="dash-card">
                <div class="dash-label">🏆 Win Rate</div>
                <div class="dash-value">{stats['win_rate']:.1f}%</div>
            </div>
        </div>
        """

        # Row 3 — 4 metrics: Avg Trade / Deposits / ROI / Best+Worst combined
        avg_cls = "dash-positive" if stats['avg_trade'] >= 0 else "dash-negative"
        best_cls = "dash-positive"
        worst_cls = "dash-negative"

        html_row3 = f"""
        <div class="dash-row">
            <div class="dash-card">
                <div class="dash-label">📊 Avg Trade</div>
                <div class="dash-value {avg_cls}">{f"+€{stats['avg_trade']:,.2f}" if stats['avg_trade'] > 0 else (f"-€{abs(stats['avg_trade']):,.2f}" if stats['avg_trade'] < 0 else "€0.00")}</div>
            </div>
            <div class="dash-card">
                <div class="dash-label">💰 Total Deposits</div>
                <div class="dash-value">€{stats['total_deposits']:,.2f}</div>
            </div>
            <div class="dash-card">
                <div class="dash-label">📈 ROI</div>
                <div class="dash-value">{f"+{stats['roi']:.1f}%" if stats['roi'] > 0 else (f"{stats['roi']:.1f}%" if stats['roi'] < 0 else "0.0%")}</div>
            </div>
            <div class="dash-card">
                <div class="dash-label">⭐ Best · 🔥 Worst</div>
                <div class="dash-value" style="font-size:0.95rem">
                    <span class="dash-positive">{f"+€{stats['best_trade']:,.2f}" if stats['best_trade'] > 0 else (f"-€{abs(stats['best_trade']):,.2f}" if stats['best_trade'] < 0 else "€0.00")}</span>
                    &nbsp;|&nbsp;
                    <span class="dash-negative">{f"+€{stats['worst_trade']:,.2f}" if stats['worst_trade'] > 0 else (f"-€{abs(stats['worst_trade']):,.2f}" if stats['worst_trade'] < 0 else "€0.00")}</span>
                </div>
            </div>
        </div>
        """

        st.markdown(html_row1 + html_row2 + html_row3, unsafe_allow_html=True)
        st.divider()

        # --- Holdings ---
        st.markdown("#### 📋 Open Positions")
        holdings = pm.get_holdings()
        if not holdings.empty:
            st.dataframe(holdings, use_container_width=True, hide_index=True)

            # Close position — leverage is inherited from the open position
            st.markdown("##### 🔒 Close a Position")

            # Get unique (ticker, direction) pairs from holdings
            close_options = sorted(set(
                (r["Ticker"], r["Direction"]) for _, r in holdings.iterrows()
            ))
            close_labels = [f"{t} ({d})" for t, d in close_options]

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                close_ticker = st.selectbox(
                    "Position", close_labels, key="pf_close_label",
                )
            with cc2:
                close_amount = st.number_input(
                    "Amount received (€)", min_value=0.0, step=10.0,
                    format="%.2f", key="pf_close_amount",
                    help="For LONG: total € received. For SHORT: total € spent to buy back.",
                )
            with cc3:
                close_fee = st.number_input("Fee (€)", min_value=0.0, step=0.5,
                                            format="%.2f", key="pf_close_fee")

            # Parse back ticker and direction
            close_ticker_val, direction = close_ticker.split(" (")
            direction = direction.rstrip(")")

            cc_note1, cc_note2 = st.columns([2, 1])
            with cc_note1:
                close_note = st.text_input("Notes", key="pf_close_note", placeholder="optional")
            with cc_note2:
                if st.button("🔒 Close Position", type="primary", use_container_width=True):
                    if close_amount <= 0:
                        st.error("Enter a valid amount.")
                    else:
                        ok = pm.close_trades(
                            ticker=close_ticker_val,
                            direction=direction,
                            amount_received=close_amount,
                            fee=close_fee,
                            note=close_note or "closed via UI",
                        )
                        if ok:
                            st.success(f"Closed {close_ticker_val} — received €{close_amount:.2f}")
                            st.rerun()
                        else:
                            st.error("No open positions found for that ticker.")
                if close_amount <= 0:
                    st.error("Enter a valid amount.")
                else:
                    ok = pm.close_trades(
                        ticker=close_ticker,
                        direction=direction,
                        amount_received=close_amount,
                        fee=close_fee,
                        note=close_note or "closed via UI",
                    )
                    if ok:
                        st.success(f"Closed {close_ticker} — received €{close_amount:.2f}")
                        st.rerun()
                    else:
                        st.error("No open positions found for that ticker.")
        else:
            st.info("No open positions.")

        st.divider()

        # --- Trade History ---
        st.markdown("#### 📜 Trade History")
        history = pm.get_trade_history()
        if not history.empty:
            st.dataframe(history, use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded yet.")

if __name__ == "__main__":
    pass  # Streamlit runs the script directly on import
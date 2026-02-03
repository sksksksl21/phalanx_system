import streamlit as st
import pandas as pd
import plotly.express as px
import os


st.set_page_config(page_title="Phalanx Backtest Dashboard", layout="wide")

DEFAULT_INITIAL_EQUITY = 10000.0

REQUIRED_TRADE_COLUMNS = ["Datetime", "Symbol", "Side", "Type", "Price", "Amount", "PnL", "Balance", "Reason"]
REQUIRED_EQUITY_COLUMNS = ["Datetime", "Equity"]


def _abs_path(filename: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, filename)


@st.cache_data(show_spinner=False)
def load_trade_log(path: str):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_TRADE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Trade CSV missing columns: {missing}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    for col in ["Price", "Amount", "PnL", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Datetime", "Type", "Symbol"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_mtm_curve(path: str):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_EQUITY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Equity CSV missing columns: {missing}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Equity"] = pd.to_numeric(df["Equity"], errors="coerce")

    df = df.dropna(subset=["Datetime", "Equity"]).sort_values("Datetime").reset_index(drop=True)
    return df


def compute_mdd_from_equity_series(df_equity: pd.DataFrame, equity_col: str):
    if df_equity is None or df_equity.empty:
        return None

    temp = df_equity[["Datetime", equity_col]].dropna().copy()
    if temp.empty:
        return None

    temp["peak"] = temp[equity_col].cummax()
    temp["dd"] = (temp[equity_col] - temp["peak"]) / temp["peak"] * 100.0
    return float(temp["dd"].min())


def compute_exit_kpis(trade_df: pd.DataFrame):
    if trade_df is None or trade_df.empty:
        return None

    exits = trade_df[trade_df["Type"] == "EXIT"].copy()
    if exits.empty:
        return None

    wins = exits[exits["PnL"] > 0]
    losses = exits[exits["PnL"] < 0]

    total_trades = len(exits)
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = float(exits["PnL"].sum())

    pos_sum = float(wins["PnL"].sum()) if not wins.empty else 0.0
    neg_sum = float(losses["PnL"].sum()) if not losses.empty else 0.0

    if neg_sum == 0.0:
        profit_factor = float("inf") if pos_sum > 0 else 0.0
    else:
        profit_factor = abs(pos_sum / neg_sum)

    return {
        "total_trades": int(total_trades),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "profit_factor": float(profit_factor),
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
    }


def main():
    st.title("🛡️ Phalanx Backtest Dashboard (Event + MTM)")

    trade_path = _abs_path("backtest_history.csv")
    mtm_path = _abs_path("backtest_equity_curve.csv")

    # Load
    try:
        df_trade = load_trade_log(trade_path)
    except Exception as e:
        st.error(f"❌ Trade log load/validation failed: {e}")
        df_trade = None

    try:
        df_mtm = load_mtm_curve(mtm_path)
    except Exception as e:
        st.warning(f"⚠️ MTM curve load failed (없으면 무시 가능): {e}")
        df_mtm = None

    # Existence checks
    if df_trade is None:
        st.error("❌ backtest_history.csv 가 없습니다/깨졌습니다. 먼저 백테스트를 실행하세요.")
        st.caption(f"Expected: {trade_path}")
        return

    if df_trade.empty:
        st.warning("Trade log 데이터가 비어있습니다.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")
    symbol_list = ["All"] + sorted(df_trade["Symbol"].dropna().unique().tolist())
    selected_symbol = st.sidebar.selectbox("Select Symbol (Event KPI/Log)", symbol_list)

    df_view = df_trade.copy()
    if selected_symbol != "All":
        df_view = df_view[df_view["Symbol"] == selected_symbol].copy()

    # KPIs
    st.subheader("📌 Portfolio KPI (Overall)")

    # Portfolio final equity from trade log (event-balance)
    bal = df_trade["Balance"].dropna()
    start_equity = float(bal.iloc[0]) if not bal.empty else DEFAULT_INITIAL_EQUITY
    final_equity_event = float(bal.iloc[-1]) if not bal.empty else DEFAULT_INITIAL_EQUITY
    roi_event = (final_equity_event - start_equity) / start_equity * 100.0 if start_equity > 0 else 0.0

    # MTM final equity
    final_equity_mtm = None
    roi_mtm = None
    if df_mtm is not None and not df_mtm.empty:
        final_equity_mtm = float(df_mtm["Equity"].iloc[-1])
        roi_mtm = (final_equity_mtm - start_equity) / start_equity * 100.0 if start_equity > 0 else 0.0

    kpi_all = compute_exit_kpis(df_trade)
    mdd_event = compute_mdd_from_equity_series(df_trade.rename(columns={"Balance": "Equity"}), "Equity")
    mdd_mtm = compute_mdd_from_equity_series(df_mtm, "Equity") if df_mtm is not None else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Final Equity (Event)", f"${final_equity_event:,.2f}", f"{roi_event:.2f}%")

    if final_equity_mtm is not None:
        c2.metric("💹 Final Equity (MTM)", f"${final_equity_mtm:,.2f}", f"{roi_mtm:.2f}%")
    else:
        c2.metric("💹 Final Equity (MTM)", "N/A")

    if mdd_event is not None:
        c3.metric("🌊 MDD (Event)", f"{mdd_event:.2f}%")
    else:
        c3.metric("🌊 MDD (Event)", "N/A")

    if mdd_mtm is not None:
        c4.metric("🌊 MDD (MTM)", f"{mdd_mtm:.2f}%")
    else:
        c4.metric("🌊 MDD (MTM)", "N/A")

    if kpi_all:
        pf = kpi_all["profit_factor"]
        c5.metric("⚖️ Profit Factor (EXIT)", "∞" if pf == float("inf") else f"{pf:.2f}")
        c6.metric("✅ Win Rate (EXIT)", f"{kpi_all['win_rate']:.1f}%  ({kpi_all['win_count']}W/{kpi_all['loss_count']}L)")
    else:
        c5.metric("⚖️ Profit Factor (EXIT)", "N/A")
        c6.metric("✅ Win Rate (EXIT)", "N/A")

    st.caption(
        "Event=거래 이벤트 기록(Balance), MTM=캔들마다 시가평가 Equity. "
        "두 곡선은 로그 방식이 달라서 MDD/ROI가 다르게 나오는 게 정상입니다."
    )

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Equity & Drawdown", "📊 PnL Analysis", "📝 Trade Log"])

    with tab1:
        st.subheader("Equity Curves (Overlay)")

        # Build overlay data
        # Event equity series
        eq_event = df_trade[["Datetime", "Balance"]].dropna().copy()
        eq_event.rename(columns={"Balance": "Equity"}, inplace=True)
        eq_event["Series"] = "Event"

        overlay = eq_event

        # MTM series (if exists)
        if df_mtm is not None and not df_mtm.empty:
            eq_mtm = df_mtm[["Datetime", "Equity"]].copy()
            eq_mtm["Series"] = "MTM"
            overlay = pd.concat([eq_event, eq_mtm], ignore_index=True)

        fig_eq = px.line(
            overlay,
            x="Datetime",
            y="Equity",
            color="Series",
            title="Equity Curve (Event vs MTM)",
        )
        st.plotly_chart(fig_eq, width='stretch')

        # Drawdown overlay (computed per series)
        st.subheader("Drawdown Curves (Overlay)")
        dd_frames = []

        def _dd(df_in, label):
            temp = df_in[["Datetime", "Equity"]].dropna().copy()
            if temp.empty:
                return None
            temp = temp.sort_values("Datetime")
            temp["peak"] = temp["Equity"].cummax()
            temp["dd"] = (temp["Equity"] - temp["peak"]) / temp["peak"] * 100.0
            temp["Series"] = label
            return temp[["Datetime", "dd", "Series"]]

        dd_event = _dd(eq_event, "Event")
        if dd_event is not None:
            dd_frames.append(dd_event)

        if df_mtm is not None and not df_mtm.empty:
            dd_mtm = _dd(df_mtm.rename(columns={"Equity": "Equity"}), "MTM")
            if dd_mtm is not None:
                dd_frames.append(dd_mtm)

        if dd_frames:
            dd_all = pd.concat(dd_frames, ignore_index=True)
            fig_dd = px.line(dd_all, x="Datetime", y="dd", color="Series", title="Drawdown (%) (Event vs MTM)")
            st.plotly_chart(fig_dd, width='stretch')
        else:
            st.info("Drawdown 표시를 위한 데이터가 부족합니다.")

    with tab2:
        exits = df_trade[df_trade["Type"] == "EXIT"].copy()
        if exits.empty:
            st.info("EXIT 거래가 없습니다.")
        else:
            # PnL by Symbol
            sym_pnl = exits.groupby("Symbol")["PnL"].sum().sort_values()
            bar_df = sym_pnl.reset_index()
            bar_df.columns = ["Symbol", "PnL"]
            fig_bar = px.bar(bar_df, x="PnL", y="Symbol", orientation="h", title="PnL by Symbol (EXIT Sum)")
            st.plotly_chart(fig_bar, width='stretch')

            # Trade PnL scatter
            scat = exits.copy()
            scat["abs_pnl"] = scat["PnL"].abs()
            fig_scatter = px.scatter(
                scat,
                x="Datetime",
                y="PnL",
                color="Symbol",
                size="abs_pnl",
                title="Trade PnL Distribution (EXIT)",
            )
            st.plotly_chart(fig_scatter, width='stretch')

    with tab3:
        st.subheader("Trade Log (Filtered)")
        st.dataframe(df_view.sort_values(by="Datetime", ascending=False), width='stretch')

        st.caption(f"Trade CSV: {trade_path}")
        if os.path.exists(mtm_path):
            st.caption(f"MTM CSV: {mtm_path}")
        else:
            st.caption("MTM CSV: (not found) — backtest_engine.py에서 backtest_equity_curve.csv 저장 패치를 적용하세요.")


if __name__ == "__main__":
    main()

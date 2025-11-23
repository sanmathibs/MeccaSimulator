import json
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model.state_based_rostering import QuantumForecast


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

STORE_MAP_INPUTS = {
    "Castle Towers": "Mecca Castle Towers",
    "Double Bay": "Mecca Double Bay",
    "Parramatta": "Mecca Parramatta",
}

FORECAST_DATA_FOLDER = "data/forecast/"
FORECAST_DATA_FILE_NAMES = {
    "Castle Towers": "Castle_Towers.json",
    "Double Bay": "Double_Bay.json",
    "Parramatta": "Parramatta.json",
}

FORECAST_INPUT_REQUIRED_COLUMNS = [
    "Date",
    "SalesActual",
    "SalesForecast",
    "HoursActual",
    "HoursForecast",
]


# -------------------------------------------------------------------
# SMALL HELPERS
# -------------------------------------------------------------------

def _info_card(title: str, value: str, subtitle: str = "") -> str:
    """Simple, elegant KPI card HTML."""
    return f"""
    <div style="
        padding: 16px 18px;
        border-radius: 12px;
        background: #0F766E10;
        border: 1px solid #0F766E40;
        color: #0F172A;
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.08);
        margin-bottom: 12px;
        min-height: 80px;
    ">
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; opacity: 0.7;">
            {title}
        </div>
        <div style="font-size: 22px; margin-top: 4px; font-weight: 700;">
            {value}
        </div>
        <div style="font-size: 11px; margin-top: 4px; opacity: 0.7;">
            {subtitle}
        </div>
    </div>
    """


def save_forecast_output(store_name: str, output_data: dict, postfix: str = "") -> Path:
    """
    Save forecast output to the designated JSON file, keeping your existing
    naming convention compatible with other tabs.
    """
    file_name = FORECAST_DATA_FILE_NAMES[store_name]
    if postfix:
        file_name = file_name.replace(".json", f"_{postfix}.json")
    output_path = Path(FORECAST_DATA_FOLDER, file_name)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4, default=str)

    return output_path


@st.cache_data(show_spinner=False)
def load_store_history(store_name: str) -> pd.DataFrame:
    """
    Load historical data for a given store from the master Excel file.
    Ensures continuous calendar (fills missing dates with zeros).
    """
    df_all = pd.read_excel("data/all_stores.xlsx")
    store_column_value = STORE_MAP_INPUTS[store_name]

    df_store = df_all[df_all["Store"] == store_column_value].copy()
    if df_store.empty:
        return pd.DataFrame()

    # Keep only required columns for QuantumForecast
    df_store = df_store[FORECAST_INPUT_REQUIRED_COLUMNS].copy()
    df_store["Date"] = pd.to_datetime(df_store["Date"])

    # Fill missing dates with zeros
    min_date = df_store["Date"].min()
    max_date = df_store["Date"].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    df_store = df_store.set_index("Date").reindex(all_dates)
    df_store.index.name = "Date"
    df_store.reset_index(inplace=True)

    for col in ["SalesActual", "SalesForecast", "HoursActual", "HoursForecast"]:
        df_store[col] = pd.to_numeric(df_store[col], errors="coerce").fillna(0)

    return df_store


@st.cache_data(show_spinner=True)
def run_quantum_forecast_cached(
    store_name: str,
    forecast_start_str: str,
    forecast_end_str: str,
    quantum_hours: float,
    blend_weight: float,
) -> dict:
    """
    Cached wrapper around QuantumForecast so repeated runs with the same
    settings are fast.
    """
    df_store = load_store_history(store_name)
    if df_store.empty:
        raise ValueError(f"No historical data found for {store_name}")

    qf = QuantumForecast(
        intput_df=df_store,
        forecast_start=forecast_start_str,
        forecast_end=forecast_end_str,
        quantum_hours=quantum_hours,
        blend_weight=blend_weight,
        name=STORE_MAP_INPUTS[store_name],
    )
    qf.run()
    return qf.output_data


def _compute_default_horizon(df_store: pd.DataFrame, weeks_forward: int = 8) -> tuple[date, date]:
    """
    Default: forecast starts the day after last actual and runs N weeks.
    """
    last_date = pd.to_datetime(df_store["Date"]).max().date()
    start = last_date + timedelta(days=1)
    end = start + timedelta(weeks=weeks_forward) - timedelta(days=1)
    return start, end


def _prepare_forecast_frames(output_data: dict, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    From QuantumForecast output JSON, return:
    - df_full: full history + forecast as DataFrame
    - df_future: only rows in forecast window
    - df_tail_hist: recent 12 weeks of history before forecast_start
    """
    df_full = pd.DataFrame(output_data["Data"])
    df_full["Date"] = pd.to_datetime(df_full["Date"])
    forecast_start = pd.to_datetime(start)
    forecast_end = pd.to_datetime(end)

    df_future = df_full[(df_full["Date"] >= forecast_start) & (df_full["Date"] <= forecast_end)].copy()

    # Tail history for chart (12 weeks back)
    tail_start = forecast_start - timedelta(weeks=12)
    df_tail_hist = df_full[
        (df_full["Date"] < forecast_start) & (df_full["Date"] >= tail_start)
    ].copy()

    return df_full, df_future, df_tail_hist


def _simple_future_kpis(df_future: pd.DataFrame) -> dict:
    """
    Compute high-level KPIs over the forecast window.
    """
    if df_future.empty:
        return {
            "days": 0,
            "total_sales": 0.0,
            "total_hours": 0.0,
            "avg_sales": 0.0,
            "peak_date": None,
            "peak_sales": 0.0,
        }

    days = df_future["Date"].nunique()
    total_sales = float(df_future["Hybrid_Forecast_Sales"].sum())
    total_hours = float(df_future["Hybrid_Forecast_Labour"].sum())
    avg_sales = float(df_future["Hybrid_Forecast_Sales"].mean())

    idx_peak = df_future["Hybrid_Forecast_Sales"].idxmax()
    peak_row = df_future.loc[idx_peak]
    peak_date = peak_row["Date"].date()
    peak_sales = float(peak_row["Hybrid_Forecast_Sales"])

    return {
        "days": days,
        "total_sales": total_sales,
        "total_hours": total_hours,
        "avg_sales": avg_sales,
        "peak_date": peak_date,
        "peak_sales": peak_sales,
    }


# -------------------------------------------------------------------
# MAIN RENDER
# -------------------------------------------------------------------

def render():
    st.header("⚡ Live Dynamic Forecast Lab")

    # --------------------------------------------------
    # Store selection + history context
    # --------------------------------------------------
    list_store_names = list(STORE_MAP_INPUTS.keys())
    col_store, col_info = st.columns([1.0, 2.0])

    with col_store:
        store_name = st.selectbox(
            "Select store",
            options=list_store_names,
            index=list_store_names.index("Parramatta"),
            key="live_forecast_store_select",
        )

    df_store = load_store_history(store_name)
    if df_store.empty:
        st.error(f"No historical data found for {store_name}.")
        return

    last_hist_date = pd.to_datetime(df_store["Date"]).max().date()
    first_hist_date = pd.to_datetime(df_store["Date"]).min().date()
    n_days = df_store["Date"].nunique()

    with col_info:
        st.caption(
            f"Historical window for **{store_name}**: "
            f"{first_hist_date:%d %b %Y} → {last_hist_date:%d %b %Y} "
            f"({n_days:,} days)"
        )

    st.markdown("---")

    # --------------------------------------------------
    # Forecast controls: horizon + hyperparameters
    # --------------------------------------------------
    st.subheader("1️⃣ Configure forecast")

    col_horizon, col_dates = st.columns([1.1, 1.6])
    col_qh, col_blend, col_save = st.columns([1.0, 1.0, 1.0])

    with col_horizon:
        preset = st.selectbox(
            "Forecast horizon",
            [
                "Next 4 weeks",
                "Next 8 weeks",
                "Next 12 weeks",
                "Custom date range",
            ],
            index=1,
            key="live_forecast_horizon_select",
            help="How far into the future you want to forecast.",
        )

    # Compute default start/end based on preset
    default_start, default_end = _compute_default_horizon(
        df_store,
        weeks_forward={"Next 4 weeks": 4, "Next 8 weeks": 8, "Next 12 weeks": 12}.get(
            preset, 8
        ),
    )

    with col_dates:
        if preset == "Custom date range":
            min_start = last_hist_date + timedelta(days=1)
            max_end = last_hist_date + timedelta(days=365)
            start_date, end_date = st.date_input(
                "Forecast period",
                value=(default_start, default_end),
                min_value=min_start,
                max_value=max_end,
                help="Pick a custom start/end for the forecast.",
            )
        else:
            # lock to the computed default
            start_date, end_date = default_start, default_end
            st.write(
                f"Forecast period: **{start_date:%d %b %Y} → {end_date:%d %b %Y}**"
            )

    # Hyperparameters (simple, non-technical wording)
    with col_qh:
        quantum_hours = st.slider(
            "Labour granularity (hours per quantum)",
            min_value=3.0,
            max_value=8.0,
            step=0.5,
            value=6.0,
            help=(
                "Controls how chunky your labour blocks are. "
                "Smaller numbers = finer-grained scheduling; "
                "larger = bigger blocks."
            ),
        )

    with col_blend:
        blend_pct = st.slider(
            "Forecast style: intuitive ↔ ML-driven",
            min_value=0,
            max_value=100,
            value=30,
            help=(
                "0% = rely fully on the RandomForest model. "
                "100% = rely fully on the intuitive seasonal/event model. "
                "We blend the two under the hood."
            ),
        )
        blend_weight = blend_pct / 100.0

    with col_save:
        save_output = st.checkbox(
            "Save this run to JSON",
            value=True,
            help="Saves under data/forecast/… so other tabs can re-use it.",
        )

    # Safety check
    if isinstance(start_date, (list, tuple)):
        start_date, end_date = start_date[0], start_date[1]
    if start_date <= last_hist_date:
        st.warning(
            "Forecast start date is on or before the last historical date. "
            "The model still runs, but forecasts are most meaningful **after** "
            "your history ends."
        )

    st.markdown("")

    # --------------------------------------------------
    # Run button + background-like status
    # --------------------------------------------------
    run_col1, run_col2, _ = st.columns([1.2, 1.2, 2.0])

    with run_col1:
        run_clicked = st.button("🚀 Run forecast", type="primary", width='stretch')

    with run_col2:
        st.caption(
            "Cached by store + horizon + settings so repeated runs are fast."
        )

    if run_clicked:
        # Validate date range
        if start_date > end_date:
            st.error("Forecast start date must be before the end date.")
            return

        forecast_start_str = start_date.strftime("%Y-%m-%d")
        forecast_end_str = end_date.strftime("%Y-%m-%d")

        try:
            # Nice status panel while the forecast is running
            if hasattr(st, "status"):
                with st.status(
                    f"Running QuantumForecast for {store_name}…",
                    expanded=True,
                ) as status:
                    st.write(
                        f"Forecasting from **{forecast_start_str}** to "
                        f"**{forecast_end_str}** with "
                        f"quantum={quantum_hours}h and blend={blend_pct}% intuitive."
                    )
                    output_data = run_quantum_forecast_cached(
                        store_name,
                        forecast_start_str,
                        forecast_end_str,
                        quantum_hours,
                        blend_weight,
                    )
                    status.update(
                        label="✅ Forecast completed",
                        state="complete",
                        expanded=False,
                    )
            else:
                # Fallback for older Streamlit
                with st.spinner("Running QuantumForecast…"):
                    output_data = run_quantum_forecast_cached(
                        store_name,
                        forecast_start_str,
                        forecast_end_str,
                        quantum_hours,
                        blend_weight,
                    )

            # Toast notification
            if hasattr(st, "toast"):
                st.toast(
                    f"Forecast completed for {store_name} "
                    f"({forecast_start_str} → {forecast_end_str}) 🎉"
                )

            # Optionally save to JSON (with a descriptive postfix)
            saved_path = None
            if save_output:
                postfix = f"{forecast_start_str}_to_{forecast_end_str}".replace("-", "")
                saved_path = save_forecast_output(store_name, output_data, postfix=postfix)

            # Attach to session_state for easy reuse in this session
            st.session_state.setdefault("live_forecast_runs", {})
            st.session_state["live_forecast_runs"][store_name] = {
                "params": {
                    "forecast_start": forecast_start_str,
                    "forecast_end": forecast_end_str,
                    "quantum_hours": quantum_hours,
                    "blend_weight": blend_weight,
                },
                "output": output_data,
                "saved_path": str(saved_path) if saved_path else None,
            }

        except Exception as e:
            st.error(f"Error running forecast: {e}")
            return

    # --------------------------------------------------
    # If we have a run in session state, show results
    # --------------------------------------------------
    run_state = (
        st.session_state.get("live_forecast_runs", {}).get(store_name)
        if "live_forecast_runs" in st.session_state
        else None
    )

    if not run_state:
        st.info("Run a forecast above to see the results.")
        return

    output_data = run_state["output"]
    forecast_start_run = datetime.strptime(
        run_state["params"]["forecast_start"], "%Y-%m-%d"
    ).date()
    forecast_end_run = datetime.strptime(
        run_state["params"]["forecast_end"], "%Y-%m-%d"
    ).date()

    st.markdown("---")
    st.subheader("2️⃣ Forecast snapshot & KPIs")

    df_full, df_future, df_tail_hist = _prepare_forecast_frames(
        output_data, forecast_start_run, forecast_end_run
    )
    kpis = _simple_future_kpis(df_future)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            _info_card(
                "Days in forecast",
                f"{kpis['days']}",
                f"{forecast_start_run:%d %b} → {forecast_end_run:%d %b}",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _info_card(
                "Total forecast sales",
                f"${kpis['total_sales']:,.0f}",
                "Hybrid forecast (sales)",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _info_card(
                "Total forecast labour",
                f"{kpis['total_hours']:,.1f} hrs",
                "Hybrid forecast (labour)",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        peak_sub = (
            f"Peak day: {kpis['peak_date']:%d %b}" if kpis["peak_date"] else "—"
        )
        st.markdown(
            _info_card(
                "Avg daily sales",
                f"${kpis['avg_sales']:,.0f}",
                peak_sub,
            ),
            unsafe_allow_html=True,
        )

    # --------------------------------------------------
    # Sample forecast snippet
    # --------------------------------------------------
    st.markdown("### 🔎 Forecast sample (first 21 days)")

    if df_future.empty:
        st.info("No rows found in the forecast window. Check your dates.")
    else:
        snippet_cols = [
            "Date",
            "DayName",
            "Hybrid_Forecast_Sales",
            "Hybrid_Forecast_Labour",
            "Intuitive_Forecast_Sales",
            "RandomForest_Forecast",
        ]
        # Some columns only exist in the future block; be defensive
        snippet_cols = [c for c in snippet_cols if c in df_future.columns]

        df_snippet = df_future.copy()
        df_snippet["Date"] = df_snippet["Date"].dt.date
        df_snippet = df_snippet[snippet_cols].head(21)

        st.dataframe(df_snippet, width='stretch')

        # Download as CSV
        csv_bytes = df_future[snippet_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full forecast window as CSV",
            data=csv_bytes,
            file_name=f"{store_name}_forecast_{forecast_start_run}_to_{forecast_end_run}.csv",
            mime="text/csv",
        )

    # --------------------------------------------------
    # Chart: recent history + new forecast
    # --------------------------------------------------
    st.markdown("### 📈 Recent actuals vs new hybrid forecast")

    if not df_tail_hist.empty:
        df_tail_hist_plot = df_tail_hist.copy()
        df_future_plot = df_future.copy()

        df_tail_hist_plot["Series"] = "Actual sales"
        df_tail_hist_plot["Value"] = df_tail_hist_plot["SalesActual"]

        df_future_plot["Series"] = "Hybrid forecast"
        df_future_plot["Value"] = df_future_plot["Hybrid_Forecast_Sales"]

        df_chart = pd.concat(
            [
                df_tail_hist_plot[["Date", "Series", "Value"]],
                df_future_plot[["Date", "Series", "Value"]],
            ],
            ignore_index=True,
        )

        fig = px.line(
            df_chart,
            x="Date",
            y="Value",
            color="Series",
            labels={"Value": "Sales", "Date": ""},
            markers=False,
        )

        # --- Draw "Forecast start" as a vertical scatter line instead of add_vline ---
        y_min = df_chart["Value"].min()
        y_max = df_chart["Value"].max()

        fig.add_scatter(
            x=[forecast_start_run, forecast_start_run],
            y=[y_min, y_max],
            mode="lines",
            line=dict(dash="dash", width=1, color="#666666"),
            name="Forecast start",
            showlegend=True,
        )

        fig.add_annotation(
            x=forecast_start_run,
            y=y_max,
            text="Forecast start",
            showarrow=False,
            yshift=10,
            font=dict(size=10),
        )

        fig.update_layout(legend_title_text="")

        st.plotly_chart(fig, width='stretch')

    else:
        st.info(
            "Not enough tail history to show a comparison chart. "
            "Try a shorter horizon or ensure you have at least ~12 weeks of history."
        )

    # --------------------------------------------------
    # Metadata & debug info
    # --------------------------------------------------
    with st.expander("⚙️ Run details & debug info"):
        st.write("**Run parameters**")
        st.json(run_state["params"])

        if run_state.get("saved_path"):
            st.write("**Saved to JSON**")
            st.code(run_state["saved_path"])
        else:
            st.write("This run was not saved to disk.")

        st.write("**Model metadata**")
        st.json(
            {
                "FoundationState": output_data.get("FoundationState"),
                "SalesBin": output_data.get("SalesBin"),
                "SmoothSlope": output_data.get("SmoothSlope"),
                "ForecastStart": output_data.get("ForecastStart"),
                "ForecastEnd": output_data.get("ForecastEnd"),
            }
        )

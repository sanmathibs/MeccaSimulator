import streamlit as st
import pandas as pd
import json
from datetime import timedelta
import plotly.express as px
import numpy as np


# ---------- DATA LOADING ----------

@st.cache_data(show_spinner=False)
def load_forecast_data():
    """Load JSON forecast files for all stores and return a combined DataFrame."""
    DATA_FILE = {
        "Castle Towers": "data/forecast/Castle_Towers.json",
        "Double Bay": "data/forecast/Double_Bay.json",
        "Parramatta": "data/forecast/Parramatta.json",
    }

    dfs = []
    numeric_cols = [
        "SalesActual", "SalesForecast", "HoursActual", "HoursForecast",
        "TF_Forecast_RawSales", "TF_Forecast_Sales",
        "SE_Forecast_State", "SE_Forecast_Sales",
        "Intuitive_Forecast_Sales", "RandomForest_Forecast",
        "Hybrid_Forecast_Sales", "Hybrid_State",
        "Hybrid_Sales_Rounded", "Hybrid_Forecast_Labour",
    ]

    for store_name, file_path in DATA_FILE.items():
        with open(file_path, "r") as f:
            data = json.load(f)

        df_store = pd.DataFrame(data.get("Data", []))
        if df_store.empty:
            continue

        df_store["Store"] = store_name
        df_store["Date"] = pd.to_datetime(df_store["Date"], errors="coerce")
        df_store[numeric_cols] = df_store[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        # History vs forecast flag
        df_store["IsHistory"] = df_store["SalesActual"].notna()

        dfs.append(df_store)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def _apply_scenario_filter(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Filter dataframe for high-level scenario choices."""
    if scenario == "All days":
        return df

    if scenario == "Special events only":
        return df[df["EventName"].notna()]

    if scenario == "Non-event days":
        return df[df["EventName"].isna()]

    if scenario == "Black Friday window":
        mask = df["EventName"].fillna("").str.contains("Black Friday")
        return df[mask]

    if scenario == "Christmas & New Year window":
        mask = df["EventName"].fillna("").str.contains("Christmas|New Year")
        return df[mask]

    if scenario == "12-week peak (Weeks 41–53 2025)":
        return df[(df["Year"] == 2025) & (df["WoY"].between(41, 53))]

    return df


# ---------- FILTER BAR ----------

def render_global_filters(df: pd.DataFrame):
    """Render filters and return the filtered dataframe + selections."""
    st.header("Filters")

    if df.empty:
        st.warning("No data loaded.")
        return pd.DataFrame(), [], None, None

    col1, col2, col3, col4 = st.columns([1.2, 1.4, 1.0, 1.3])

    # Store selector
    stores = sorted(df["Store"].dropna().unique().tolist())
    with col1:
        selected_stores = st.multiselect(
            "Store(s)",
            options=stores,
            default=stores,
        )

    # Date presets
    min_date = df["Date"].min()
    max_date = df["Date"].max()

    with col2:
        preset_options = {
            "Last 12 weeks": (max_date - timedelta(weeks=12), max_date),
            "Last 52 weeks": (max_date - timedelta(weeks=52), max_date),
            "Full history": (min_date, max_date),
        }
        preset_choice = st.selectbox(
            "Period preset",
            options=list(preset_options.keys()),
            index=2,
        )
        preset_start, preset_end = preset_options[preset_choice]

    # Model selector
    with col3:
        model_options = ["Mecca", "Intuitive", "RandomForest", "Hybrid"]
        selected_model = st.selectbox(
            "Forecast model", options=model_options, index=3
        )

    # Scenario selector
    with col4:
        scenario_options = [
            "All days",
            "Special events only",
            "Non-event days",
            "Black Friday window",
            "Christmas & New Year window",
            "12-week peak (Weeks 41–53 2025)",
        ]
        selected_scenario = st.selectbox(
            "Scenario focus", options=scenario_options, index=0
        )

    # Date range picker
    start_date, end_date = st.date_input(
        "Date range",
        value=(preset_start.date(), preset_end.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Apply filters
    df_filtered = df.copy()

    if selected_stores:
        df_filtered = df_filtered[df_filtered["Store"].isin(selected_stores)]

    df_filtered = df_filtered[
        (df_filtered["Date"] >= start_date)
        & (df_filtered["Date"] <= end_date)
    ]

    df_filtered = _apply_scenario_filter(df_filtered, selected_scenario)

    # Map selected forecast model to column
    model_col_map = {
        "Mecca": "SalesForecast",
        "Intuitive": "Intuitive_Forecast_Sales",
        "RandomForest": "RandomForest_Forecast",
        "Hybrid": "Hybrid_Forecast_Sales",
    }
    df_filtered["Forecast_Selected_Model"] = df_filtered[model_col_map[selected_model]]

    return df_filtered, selected_stores, selected_model, selected_scenario


# ---------- METRICS & IMPACT ----------

def compute_overview_metrics(df_filtered: pd.DataFrame, selected_model: str):
    df_hist = df_filtered[df_filtered["SalesActual"].notna()].copy()
    if df_hist.empty:
        return None

    total_sales = df_hist["SalesActual"].sum()
    total_forecast = df_hist["Forecast_Selected_Model"].sum()
    total_hours = df_hist["HoursActual"].sum()
    sales_per_hour = total_sales / total_hours if total_hours else np.nan

    abs_err_model = (df_hist["SalesActual"] - df_hist["Forecast_Selected_Model"]).abs()
    mae_model = abs_err_model.mean()
    rmse = np.sqrt((abs_err_model ** 2).mean())
    mape = (
        abs_err_model
        .div(df_hist["SalesActual"].replace(0, np.nan))
        .mean()
        * 100
    )

    abs_err_mecca = (df_hist["SalesActual"] - df_hist["SalesForecast"]).abs()
    mae_mecca = abs_err_mecca.mean()

    if selected_model == "Mecca" or mae_mecca == 0 or np.isnan(mae_mecca):
        improvement_pct = 0.0
    else:
        improvement_pct = (mae_mecca - mae_model) / mae_mecca * 100

    return {
        "df_hist": df_hist,
        "total_sales": total_sales,
        "total_forecast": total_forecast,
        "total_hours": total_hours,
        "sales_per_hour": sales_per_hour,
        "mae_model": mae_model,
        "mae_mecca": mae_mecca,
        "mape": mape,
        "rmse": rmse,
        "improvement_pct": improvement_pct,
    }


def render_impact_banner(metrics: dict, selected_model: str):
    if metrics is None:
        return

    improvement = metrics["improvement_pct"]

    if selected_model == "Mecca":
        headline = "Viewing MECCA baseline forecast"
        sub = "Switch to Intuitive, RandomForest or Hybrid to see impact vs MECCA."
    else:
        if improvement >= 0:
            headline = (
                f"{selected_model} reduces average forecast error "
                f"by {improvement:.1f}% vs MECCA"
            )
        else:
            headline = (
                f"{selected_model} increases average forecast error "
                f"by {abs(improvement):.1f}% vs MECCA"
            )
        sub = "Based on historical days in the selected period and stores."

    st.markdown(
        f"""
        <div style="
            margin-top: 10px;
            margin-bottom: 10px;
            padding: 18px 22px;
            border-radius: 12px;
            background: linear-gradient(90deg, #f3f6ff, #ffffff);
            border: 1px solid #e2e6ff;
        ">
            <div style="font-size:18px; font-weight:600; margin-bottom:4px;">
                {headline}
            </div>
            <div style="font-size:13px; color:#555;">
                {sub}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(metrics: dict, selected_model: str):
    if metrics is None:
        st.warning("No historical actuals in the selected period.")
        return

    total_sales = metrics["total_sales"]
    total_forecast = metrics["total_forecast"]
    total_hours = metrics["total_hours"]
    sales_per_hour = metrics["sales_per_hour"]
    mape = metrics["mape"]
    rmse = metrics["rmse"]
    improvement_pct = metrics["improvement_pct"]

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    def kpi_card(col, title, value, subtitle=None):
        subtitle_html = (
            f"<div style='font-size:12px;color:#666;'>{subtitle}</div>"
            if subtitle else ""
        )
        col.markdown(
            f"""
            <div style="
                background-color: #ffffff;
                padding: 16px;
                border-radius: 10px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
                text-align: center;
                min-height: 110px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                width: 100%;
            ">
                <div style="font-size:14px; margin-bottom:6px;">{title}</div>
                <div style="font-size:22px; font-weight:700; margin-bottom:4px;">{value}</div>
                {subtitle_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    kpi_card(col1, "Total actual sales", f"${total_sales:,.0f}")
    kpi_card(col2, f"Total forecast ({selected_model})", f"${total_forecast:,.0f}")
    kpi_card(
        col3,
        "MAPE / RMSE (Based on days with actual)",
        f"{mape:,.2f}% / {rmse:,.0f}",
        " lesser the better",
    )
    kpi_card(col4, "Labour hours (actual)", f"{total_hours:,.0f}")
    kpi_card(
        col5,
        "Sales per labour hour",
        f"${sales_per_hour:,.0f}" if not np.isnan(sales_per_hour) else "–",
    )

    if selected_model == "Mecca":
        kpi_card(
            col6,
            "Error improvement vs MECCA",
            "0.0%",
            "You are viewing the MECCA forecast",
        )
    else:
        sign = "+" if improvement_pct >= 0 else ""
        kpi_card(
            col6,
            "Error improvement vs MECCA",
            f"{sign}{improvement_pct:,.1f}%",
            "Positive = lower MAE than MECCA",
        )


# ---------- HERO CHART ----------

def render_event_bar_chart(df_plot: pd.DataFrame, selected_model: str):
    """For special-event scenarios: per-store total sales vs forecast."""
    df_events = df_plot[df_plot["IsHistory"]].copy()
    if df_events.empty:
        st.warning("No historical event days in this selection.")
        return

    df_agg = (
        df_events.groupby("Store")
        .agg(
            SalesActual=("SalesActual", "sum"),
            Forecast_Selected_Model=("Forecast_Selected_Model", "sum"),
        )
        .reset_index()
    )

    df_long = df_agg.melt(
        id_vars="Store",
        value_vars=["SalesActual", "Forecast_Selected_Model"],
        var_name="Series",
        value_name="Sales",
    )
    df_long["Series"] = df_long["Series"].map(
        {"SalesActual": "Actual", "Forecast_Selected_Model": selected_model}
    )

    fig = px.bar(
        df_long,
        x="Store",
        y="Sales",
        color="Series",
        barmode="group",
        title=f"Special-event sales vs {selected_model} forecast (total for selected period)",
    )
    fig.update_layout(
        xaxis_title="Store",
        yaxis_title="Total sales",
        legend_title="",
        title_x=0.5,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_network_trend_chart(
    df_filtered: pd.DataFrame,
    selected_model: str,
    selected_stores,
    scenario_focus: str,
):
    """
    Hero chart:
      - For event-focused scenarios -> per-store bar chart
      - Otherwise -> clean aggregated time series
    """
    if df_filtered.empty:
        st.warning("No data to display chart.")
        return

    df_plot = df_filtered.copy()
    if selected_stores:
        df_plot = df_plot[df_plot["Store"].isin(selected_stores)]

    if df_plot.empty:
        st.warning("No data for the selected store(s).")
        return

    event_focus = scenario_focus in [
        "Special events only",
        "Black Friday window",
        "Christmas & New Year window",
    ]

    if event_focus:
        render_event_bar_chart(df_plot, selected_model)
        return

    agg_choice = st.radio(
        "Aggregation level", options=["Daily", "Weekly"], horizontal=True
    )

    if agg_choice == "Weekly":
        df_plot["WeekStart"] = df_plot["Date"] - pd.to_timedelta(
            df_plot["Date"].dt.weekday, unit="d"
        )
        group_col = "WeekStart"
    else:
        group_col = "Date"

    df_agg = (
        df_plot.groupby(group_col)
        .agg(
            SalesActual=("SalesActual", "sum"),
            Forecast_Selected_Model=("Forecast_Selected_Model", "sum"),
        )
        .reset_index()
    )
    legend_actual = "Actual (all stores)"
    legend_model = f"{selected_model} (all stores)"

    df_agg = df_agg.rename(
        columns={
            "SalesActual": legend_actual,
            "Forecast_Selected_Model": legend_model,
        }
    )

    fig = px.line(
        df_agg,
        x=group_col,
        y=[legend_actual, legend_model],
        labels={group_col: "Date", "value": "Sales", "variable": "Series"},
        title=f"Actual sales vs {selected_model} forecast (network total)",
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        legend_title="",
        title_x=0.5,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_forecasting_insights(metrics: dict, selected_model: str):
    """Short, network-level forecasting insights shown under the main chart."""
    if metrics is None:
        return

    df_hist = metrics["df_hist"]
    n_days = df_hist["Date"].nunique()
    n_stores = df_hist["Store"].nunique()

    total_sales = metrics["total_sales"]
    mae_mecca = metrics["mae_mecca"]
    mae_model = metrics["mae_model"]
    mape = metrics["mape"]
    improvement = metrics["improvement_pct"]

    st.markdown(
        """
        <div style="font-weight:650; margin-top:18px; margin-bottom:4px; font-size:24px;">
            Forecasting insights (network view)
        </div>
        """,
        unsafe_allow_html=True,
    )

    html = f"""
    <ul style="margin-top:0; font-size:16px;">
        <li>
            Across <strong>{n_stores}</strong> stores and
            <strong>{n_days}</strong> trading days, the
            <strong>{selected_model}</strong> model achieves an average MAPE of
            <strong>{mape:,.1f}%</strong>.
        </li>
        <li>
            Mean absolute error falls from <strong>${mae_mecca:,.0f}</strong> under the
            MECCA forecast to <strong>${mae_model:,.0f}</strong>, a
            <strong>{improvement:,.1f}% reduction</strong> in forecast error.
        </li>
        <li>
            Over this period the network generated
            <strong>${total_sales:,.0f}</strong> of actual sales, so even single-digit
            percentage improvements represent meaningful dollars for labour and stock
            decisions.
        </li>
    </ul>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------- STORE-WISE ANALYSIS ----------

def compute_store_stats(df_hist: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    """Per-store MAPE for MECCA, Hybrid, and optionally selected model + improvements."""
    rows = []
    sel_col_map = {
        "Mecca": "SalesForecast",
        "Intuitive": "Intuitive_Forecast_Sales",
        "RandomForest": "RandomForest_Forecast",
        "Hybrid": "Hybrid_Forecast_Sales",
    }

    for store in sorted(df_hist["Store"].unique()):
        d = df_hist[df_hist["Store"] == store]
        if d.empty:
            continue

        abs_err_mecca = (d["SalesActual"] - d["SalesForecast"]).abs()
        abs_err_hybrid = (d["SalesActual"] - d["Hybrid_Forecast_Sales"]).abs()

        mape_mecca = (
            abs_err_mecca
            .div(d["SalesActual"].replace(0, np.nan))
            .mean()
            * 100
        )
        mape_hybrid = (
            abs_err_hybrid
            .div(d["SalesActual"].replace(0, np.nan))
            .mean()
            * 100
        )
        imp_hybrid = (
            (mape_mecca - mape_hybrid) / mape_mecca * 100 if mape_mecca else 0.0
        )

        row = {
            "Store": store,
            "MECCA MAPE %": mape_mecca,
            "Hybrid MAPE %": mape_hybrid,
            "Hybrid improvement vs MECCA %": imp_hybrid,
        }

        # Add selected model if it is different from Hybrid and Mecca
        if selected_model not in ["Hybrid", "Mecca"]:
            abs_err_sel = (d["SalesActual"] - d[sel_col_map[selected_model]]).abs()
            mape_sel = (
                abs_err_sel
                .div(d["SalesActual"].replace(0, np.nan))
                .mean()
                * 100
            )
            imp_sel = (
                (mape_mecca - mape_sel) / mape_mecca * 100 if mape_mecca else 0.0
            )
            row[f"{selected_model} MAPE %"] = mape_sel
            row[f"{selected_model} improvement vs MECCA %"] = imp_sel

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        by="Hybrid improvement vs MECCA %", ascending=False
    )


def render_store_analysis(df_hist: pd.DataFrame, selected_model: str):
    """Store-wise accuracy chart + overall summary table."""
    if df_hist.empty:
        return

    df_stats = compute_store_stats(df_hist, selected_model)
    if df_stats.empty:
        return

    st.subheader("Store-wise forecast accuracy")

    # Build long format for grouped bar chart
    series_map = [
        ("MECCA MAPE %", "MECCA forecast"),
        ("Hybrid MAPE %", "Hybrid forecast"),
    ]
    if f"{selected_model} MAPE %" in df_stats.columns:
        series_map.append((f"{selected_model} MAPE %", f"{selected_model} forecast"))

    long_parts = []
    for col_name, label in series_map:
        temp = df_stats[["Store", col_name]].copy()
        temp.rename(columns={col_name: "MAPE"}, inplace=True)
        temp["Series"] = label
        long_parts.append(temp)

    df_long = pd.concat(long_parts, ignore_index=True)

    fig = px.bar(
        df_long,
        x="Store",
        y="MAPE",
        color="Series",
        barmode="group",
        text=df_long["MAPE"].map(lambda x: f"{x:,.1f}%"),
        labels={"MAPE": "MAPE (%)"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Store",
        yaxis_title="MAPE (%)",
        legend_title="",
        title="MECCA vs Hybrid (and selected model) by store",
        title_x=0.5,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Overall storewise summary table (no fancy Styler logic)
    st.markdown("**Store-wise summary (MAPE and improvements vs MECCA)**")
    display_cols = [
        "Store",
        "MECCA MAPE %",
        "Hybrid MAPE %",
        "Hybrid improvement vs MECCA %",
    ]

    # Only add extra columns if the selected model is NOT already Hybrid or Mecca
    if selected_model not in ["Hybrid", "Mecca"]:
        if f"{selected_model} MAPE %" in df_stats.columns:
            display_cols.append(f"{selected_model} MAPE %")
        if f"{selected_model} improvement vs MECCA %" in df_stats.columns:
            display_cols.append(f"{selected_model} improvement vs MECCA %")

    df_show = df_stats[display_cols].copy()
    st.dataframe(
        df_show.style.format("{:.2f}", subset=df_show.columns.difference(["Store"])),
        use_container_width=True,
    )

    return df_stats  # so we can reuse it in the detail section


# ---------- STORE DETAIL ANALYSIS ----------

def render_store_detail(df_hist: pd.DataFrame, df_stats: pd.DataFrame):
    """Detailed analysis for a single store (default Parramatta)."""
    if df_hist.empty or df_stats.empty:
        return

    st.subheader("Store detail analysis")

    stores = sorted(df_hist["Store"].unique().tolist())
    default_index = stores.index("Parramatta") if "Parramatta" in stores else 0
    store_sel = st.selectbox("Select store", stores, index=default_index)

    d = df_hist[df_hist["Store"] == store_sel].copy()
    if d.empty:
        st.info("No data for selected store.")
        return

    # Volume metrics
    n_days = d["Date"].nunique()
    n_event_days = d[d["EventName"].notna()]["Date"].nunique()
    total_sales = d["SalesActual"].sum()
    total_hours = d["HoursActual"].sum()
    sales_per_hour = total_sales / total_hours if total_hours else np.nan
    event_sales = d[d["EventName"].notna()]["SalesActual"].sum()
    event_share = event_sales / total_sales * 100 if total_sales else np.nan

    # Summary table for this store
    summary_rows = [
        ["Store", store_sel],
        ["Days with actuals", n_days],
        ["Event days", n_event_days],
        ["Total actual sales", f"${total_sales:,.0f}"],
        ["Total labour hours", f"{total_hours:,.0f}"],
        [
            "Sales per labour hour",
            f"${sales_per_hour:,.0f}" if not np.isnan(sales_per_hour) else "–",
        ],
        [
            "Share of sales from event days",
            f"{event_share:,.1f}%" if not np.isnan(event_share) else "–",
        ],
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    # Model performance for this store
    model_cols = {
        "MECCA": "SalesForecast",
        "Intuitive": "Intuitive_Forecast_Sales",
        "RandomForest": "RandomForest_Forecast",
        "Hybrid": "Hybrid_Forecast_Sales",
    }
    perf_rows = []
    for model_name, col in model_cols.items():
        abs_err = (d["SalesActual"] - d[col]).abs()
        mae = abs_err.mean()
        mape = (
            abs_err
            .div(d["SalesActual"].replace(0, np.nan))
            .mean()
            * 100
        )
        total_forecast = d[col].sum()
        perf_rows.append(
            {
                "Model": model_name,
                "Total forecast sales": total_forecast,
                "MAPE %": mape,
                "MAE": mae,
            }
        )
    df_perf = pd.DataFrame(perf_rows)

    # Layout: summary on left, model performance on right
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("**Store snapshot**")
        st.table(df_summary)
    with col_right:
        st.markdown("**Model performance for this store**")
        st.dataframe(
            df_perf.style.format(
                {
                    "Total forecast sales": "${:,.0f}",
                    "MAPE %": "{:.2f}",
                    "MAE": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )

    # 3 store-specific insight statements
    # Use stats from df_stats for Hybrid vs MECCA
    row_stats = df_stats[df_stats["Store"] == store_sel].iloc[0]
    hybrid_imp = row_stats["Hybrid improvement vs MECCA %"]
    mape_mecca = row_stats["MECCA MAPE %"]
    mape_hybrid = row_stats["Hybrid MAPE %"]

    # Event vs non-event MAPE (Hybrid)
    d_events = d[d["EventName"].notna()]
    d_non = d[d["EventName"].isna()]

    def mape_for(df_local: pd.DataFrame, col: str):
        if df_local.empty:
            return np.nan
        abs_err = (df_local["SalesActual"] - df_local[col]).abs()
        return (
            abs_err
            .div(df_local["SalesActual"].replace(0, np.nan))
            .mean()
            * 100
        )

    mape_mecca_event = mape_for(d_events, "SalesForecast")
    mape_hybrid_event = mape_for(d_events, "Hybrid_Forecast_Sales")
    mape_mecca_non = mape_for(d_non, "SalesForecast")
    mape_hybrid_non = mape_for(d_non, "Hybrid_Forecast_Sales")

    # --- Store insights (clean HTML, no stray ** characters) ---
    st.markdown(
        """
        <div style="font-weight:600; margin-top:12px; margin-bottom:6px; font-size:24px;">
            Store insights
        </div>
        """,
        unsafe_allow_html=True,
    )

    store_insights_html = f"""
    <ul style="margin-top:0; font-size:15px;">
        <li>
            Over the selected period, <strong>{store_sel}</strong> generated
            <strong>${total_sales:,.0f}</strong> of actual sales from
            <strong>{total_hours:,.0f}</strong> labour hours – around
            <strong>${sales_per_hour:,.0f} per labour hour</strong>.
        </li>
        <li>
            The <strong>Hybrid</strong> forecast improves accuracy from
            <strong>{mape_mecca:,.1f}% MAPE (MECCA)</strong> to
            <strong>{mape_hybrid:,.1f}%</strong>, a
            <strong>{hybrid_imp:,.1f}% reduction in error</strong> for this store.
        </li>
        <li>
            Event days make up <strong>{event_share:,.1f}%</strong> of sales; on those days
            Hybrid MAPE improves from <strong>{mape_mecca_event:,.1f}% (MECCA)</strong>
            to <strong>{mape_hybrid_event:,.1f}%</strong>, while on non-event days it
            improves from <strong>{mape_mecca_non:,.1f}%</strong> to
            <strong>{mape_hybrid_non:,.1f}%</strong>, giving planners a more reliable
            signal for both peaks and BAU.
        </li>
    </ul>
    """
    st.markdown(store_insights_html, unsafe_allow_html=True)


# ---------- ENTRY POINT ----------

def render():
    df = load_forecast_data()
    df_filtered, stores, model, scenario = render_global_filters(df)

    if df_filtered.empty:
        return

    metrics = compute_overview_metrics(df_filtered, model)

    render_impact_banner(metrics, model)
    render_kpi_cards(metrics, model)

    st.markdown("---")

    render_network_trend_chart(df_filtered, model, stores, scenario)
    render_forecasting_insights(metrics, model)

    st.markdown("---")

    if metrics is not None:
        df_stats = render_store_analysis(metrics["df_hist"], model)
        st.markdown("---")
        render_store_detail(metrics["df_hist"], df_stats)

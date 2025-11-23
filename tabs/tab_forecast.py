import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.holiday_utils import get_holiday  # optional, kept for future extension

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_FILE = {
    "Castle Towers": "data/forecast/Castle_Towers.json",
    "Double Bay": "data/forecast/Double_Bay.json",
    "Parramatta": "data/forecast/Parramatta.json",
}

# Used by Plotly charts
COLOR_MAP = {
    # Sales
    "Actual Sales": "#ff0000",
    "Mecca Forecast": "#5700b3",
    "Intuitive Forecast": "#0059ff",
    "RandomForest Forecast": "#ffa500",
    "Hybrid Forecast": "#00E2AD",
    # Labour hours
    "Actual Labour Hours": "#ff0000",
    "Mecca Forecast Labour Hours": "#5700b3",
    "Hybrid Forecast Labour Hours": "#00E2AD",
    # Errors
    "Mecca Forecast Error": "#5700b3",
    "Intuitive Forecast Error": "#0059ff",
    "RandomForest Forecast Error": "#ffa500",
    "Hybrid Forecast Error": "#00E2AD",
}

MODEL_COL_MAP = {
    "Mecca": "SalesForecast",
    "Intuitive": "Intuitive_Forecast_Sales",
    "RandomForest": "RandomForest_Forecast",
    "Hybrid": "Hybrid_Forecast_Sales",
}


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_location_data(location: str):
    """Load JSON for a store and return dataframe + meta."""
    file_path = DATA_FILE[location]
    with open(file_path, "r") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["Data"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df, raw.get("SalesBin", None)


def apply_scenario_filter(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Filter dataframe for All days / Special events / Non-event days."""
    if scenario == "All days":
        return df

    if scenario == "Special events only":
        return df[df["EventName"].notna()]

    if scenario == "Non-event days":
        return df[df["EventName"].isna()]

    return df


def compute_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAE, RMSE, MAPE, Avg error & improvement vs Mecca
    for each forecast model on the currently filtered window.
    """
    df_hist = df.dropna(subset=["SalesActual"]).copy()
    metrics = []

    for model, col in MODEL_COL_MAP.items():
        if col not in df_hist.columns:
            continue

        d = df_hist[["SalesActual", col]].dropna()
        if d.empty:
            continue

        err = d["SalesActual"] - d[col]
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        # standard MAPE: mean(|e|/|actual|)
        mape = (
            np.mean(np.abs(err) / d["SalesActual"].replace(0, np.nan)) * 100
        )
        avg_err = np.mean(err)

        metrics.append(
            {
                "Model": model,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "Avg Error": avg_err,
            }
        )

    if not metrics:
        return pd.DataFrame()

    df_metrics = pd.DataFrame(metrics).set_index("Model")

    # Improvement vs Mecca (higher is better)
    if "Mecca" in df_metrics.index:
        mecca_mae = df_metrics.loc["Mecca", "MAE"]
        if mecca_mae != 0:
            df_metrics["Improvement vs Mecca (%)"] = (
                (mecca_mae - df_metrics["MAE"]) / mecca_mae * 100
            )
        else:
            df_metrics["Improvement vs Mecca (%)"] = 0.0
        df_metrics.loc["Mecca", "Improvement vs Mecca (%)"] = 0.0

    return df_metrics


def make_sales_timeseries(
    df: pd.DataFrame, forecast_models: list, aggregation: str = "Daily"
) -> pd.DataFrame:
    """Prepare long-form dataframe for Actual + selected models."""
    cols = ["SalesActual"]
    for m in forecast_models:
        col = MODEL_COL_MAP.get(m)
        if col and col in df.columns:
            cols.append(col)

    if len(cols) == 1:
        # only actual
        df_agg = df[["Date", "SalesActual"]].copy()
    else:
        if aggregation == "Weekly":
            df = df.copy()
            df["WeekStart"] = df["Date"] - pd.to_timedelta(
                df["Date"].dt.weekday, unit="d"
            )
            group_col = "WeekStart"
        else:
            group_col = "Date"

        df_agg = (
            df.groupby(group_col)[cols]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={group_col: "Date"})
        )

    # Rename for display
    rename_map = {
        "SalesActual": "Actual Sales",
        "SalesForecast": "Mecca Forecast",
        "Intuitive_Forecast_Sales": "Intuitive Forecast",
        "RandomForest_Forecast": "RandomForest Forecast",
        "Hybrid_Forecast_Sales": "Hybrid Forecast",
    }
    df_agg = df_agg.rename(columns=rename_map)

    long_df = df_agg.melt(
        id_vars="Date",
        var_name="Series",
        value_name="Sales",
    )
    return long_df


def make_error_timeseries(
    df: pd.DataFrame, forecast_models: list, aggregation: str = "Daily"
) -> pd.DataFrame:
    """Prepare long-form dataframe for forecast error over time."""
    df_hist = df.dropna(subset=["SalesActual"]).copy()
    if df_hist.empty:
        return pd.DataFrame(columns=["Date", "Series", "Error"])

    for m in forecast_models:
        col = MODEL_COL_MAP.get(m)
        if col and col in df_hist.columns:
            display = f"{m} Forecast Error" if m != "Mecca" else "Mecca Forecast Error"
            df_hist[display] = df_hist[col] - df_hist["SalesActual"]

    error_cols = [c for c in df_hist.columns if "Forecast Error" in c]
    if not error_cols:
        return pd.DataFrame(columns=["Date", "Series", "Error"])

    if aggregation == "Weekly":
        df_hist["WeekStart"] = df_hist["Date"] - pd.to_timedelta(
            df_hist["Date"].dt.weekday, unit="d"
        )
        group_col = "WeekStart"
    else:
        group_col = "Date"

    df_agg = (
        df_hist.groupby(group_col)[error_cols]
        .mean()
        .reset_index()
        .rename(columns={group_col: "Date"})
    )

    long_df = df_agg.melt(
        id_vars="Date",
        value_vars=error_cols,
        var_name="Series",
        value_name="Error",
    )
    return long_df


# -------------------------------------------------------------------
# MAIN RENDER
# -------------------------------------------------------------------

def render():
    st.header("Forecast Analysis")

    # --------------------------------------------------
    # Control bar with synced Period & Date range
    # --------------------------------------------------
    col_store, col_period, col_date, col_scenario, col_models = st.columns(
        [1.0, 1.0, 1.7, 1.3, 1.4]
    )

    with col_store:
        location = st.selectbox("Store", list(DATA_FILE.keys()), index=2)

    # load data for this store
    df_raw, sales_bin = load_location_data(location)
    min_date = df_raw["Date"].min().normalize()
    max_date = df_raw["Date"].max().normalize()

    # --- helpers to compute ranges & keep state ---

    def compute_range(label: str):
        """Return (start_date, end_date) for a given period label."""
        if label == "Last 12 weeks":
            start = max_date - pd.Timedelta(weeks=12)
        elif label == "Last 52 weeks":
            start = max_date - pd.Timedelta(weeks=52)
        elif label == "Full history":
            start = min_date
        else:  # "Custom" or unknown
            start = min_date

        if start < min_date:
            start = min_date

        return start.date(), max_date.date()

    # initialise session_state keys on first load
    if "forecast_period" not in st.session_state:
        st.session_state["forecast_period"] = "Last 52 weeks"
    if "forecast_date_range" not in st.session_state:
        st.session_state["forecast_date_range"] = compute_range(
            st.session_state["forecast_period"]
        )

    # callbacks that will be triggered when widgets change
    def on_period_change():
        start, end = compute_range(st.session_state["forecast_period"])
        st.session_state["forecast_date_range"] = (start, end)

    def on_date_change():
        # user is manually overriding the dates -> mark as custom period
        st.session_state["forecast_period"] = "Custom"

    with col_period:
        period_preset = st.selectbox(
            "Period",
            ["Last 12 weeks", "Last 52 weeks", "Full history", "Custom"],
            key="forecast_period",
            on_change=on_period_change,
        )

    with col_date:
        date_range = st.date_input(
            "Date range",
            value=st.session_state["forecast_date_range"],
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="forecast_date_range",
            on_change=on_date_change,
        )

    start_date, end_date = [pd.to_datetime(d) for d in st.session_state["forecast_date_range"]]

    with col_scenario:
        scenario = st.selectbox(
            "Scenario focus",
            ["All days", "Special events only", "Non-event days"],
            index=0,
        )

    with col_models:
        forecast_models = st.multiselect(
            "Forecast models",
            ["Mecca", "Intuitive", "RandomForest", "Hybrid"],
            default=["Mecca", "Intuitive", "RandomForest", "Hybrid"],
        )

    st.markdown("---")

    # --------------------------------------------------
    # Filter data
    # --------------------------------------------------
    df = df_raw[(df_raw["Date"] >= start_date) & (df_raw["Date"] <= end_date)].copy()
    df = apply_scenario_filter(df, scenario)

    if df.empty:
        st.info("No data for the selected filters.")
        return

    # Aggregation for time-series
    agg_level = st.radio(
        "Aggregation level",
        options=["Daily", "Weekly"],
        index=0,
        horizontal=True,
        key="forecast_agg_level",
    )

    if not forecast_models:
        st.warning("Select at least one forecast model to explore.")
        return

    # --------------------------------------------------
    # Model metrics & high-level summary
    # --------------------------------------------------
    metrics_df = compute_model_metrics(df)

    if metrics_df.empty:
        st.info("No historical actuals available for error metrics in this window.")
        return

    # choose primary model = best MAPE (lower is better)
    best_model = metrics_df["MAPE (%)"].idxmin()
    best_row = metrics_df.loc[best_model]
    mecca_mape = (
        metrics_df.loc["Mecca", "MAPE (%)"] if "Mecca" in metrics_df.index else np.nan
    )
    best_imp = (
        (mecca_mape - best_row["MAPE (%)"]) / mecca_mape * 100
        if "Mecca" in metrics_df.index and mecca_mape
        else 0.0
    )

    st.markdown(
        f"""
        <div style="
            margin-top:4px;
            margin-bottom:10px;
            padding:14px 18px;
            border-radius:12px;
            background:linear-gradient(90deg,#f4f7ff,#ffffff);
            border:1px solid #e1e5ff;">
            <div style="font-size:15px;font-weight:600;margin-bottom:4px;">
                {best_model} is the most accurate model for {location} in this window
            </div>
            <div style="font-size:13px;color:#555;">
                It achieves a MAPE of <strong>{best_row['MAPE (%)']:.1f}%</strong>.
        """
        + (
            f" Compared with the MECCA forecast ({mecca_mape:.1f}%), "
            f"this is a <strong>{best_imp:.1f}% reduction</strong> in average error."
            if best_model != "Mecca" and not np.isnan(mecca_mape)
            else ""
        )
        + """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # KPI cards for a selected model
    # --------------------------------------------------
    primary_model = st.selectbox(
        "Primary model for KPIs",
        options=list(MODEL_COL_MAP.keys()),
        index=list(MODEL_COL_MAP.keys()).index(best_model),
        help="Choose which model to highlight in the stats below.",
    )

    df_hist = df.dropna(subset=["SalesActual"]).copy()
    primary_col = MODEL_COL_MAP[primary_model]
    d = df_hist[["SalesActual", primary_col]].dropna()
    err = d["SalesActual"] - d[primary_col]

    total_sales = df_hist["SalesActual"].sum()
    mae = np.mean(np.abs(err))
    mape = metrics_df.loc[primary_model, "MAPE (%)"]
    avg_err = np.mean(err)
    days = d.shape[0]
    event_share = (
        df_hist[df_hist["EventName"].notna()]["SalesActual"].sum() / total_sales * 100
        if total_sales
        else np.nan
    )

    col_a, col_b, col_c, col_d = st.columns(4)

    def kpi_card(col, title, value, subtitle=None):
        subtitle_html = (
            f"<div style='font-size:12px;color:#666;margin-top:2px;'>{subtitle}</div>"
            if subtitle
            else ""
        )
        col.markdown(
            f"""
            <div style="
                background-color:#ffffff;
                padding:14px 16px;
                border-radius:10px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.05);
                min-height:90px;">
                <div style="font-size:13px;color:#555;margin-bottom:4px;">{title}</div>
                <div style="font-size:20px;font-weight:700;">{value}</div>
                {subtitle_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    kpi_card(col_a, "Total actual sales", f"${total_sales:,.0f}", f"{location}")
    kpi_card(
        col_b,
        f"{primary_model} MAPE",
        f"{mape:.1f}%",
        f"Based on {days} days with actuals",
    )
    kpi_card(
        col_c,
        "Mean absolute error",
        f"${mae:,.0f}",
        "Average daily $ error",
    )
    kpi_card(
        col_d,
        "Sales from event days",
        f"{event_share:.1f}%" if not np.isnan(event_share) else "–",
        "Share of sales on special events",
    )

    st.markdown("---")

    # --------------------------------------------------
    # Sales forecast chart
    # --------------------------------------------------
    st.subheader("Sales forecast vs actual")

    sales_long = make_sales_timeseries(df, forecast_models, aggregation=agg_level)
    if not sales_long.empty:
        fig_sales = px.line(
            sales_long,
            x="Date",
            y="Sales",
            color="Series",
            color_discrete_map=COLOR_MAP,
        )
        fig_sales.update_layout(
            xaxis_title="",
            yaxis_title="Sales",
            legend_title="",
            margin=dict(l=10, r=10, t=30, b=10),
            title="",
        )
        st.plotly_chart(fig_sales, width='stretch')
    else:
        st.info("No data to display for sales chart.")

    # --------------------------------------------------
    # Error over time
    # --------------------------------------------------
    st.subheader("Forecast error over time")

    error_long = make_error_timeseries(df, forecast_models, aggregation=agg_level)
    if not error_long.empty:
        fig_err = px.line(
            error_long,
            x="Date",
            y="Error",
            color="Series",
            color_discrete_map=COLOR_MAP,
        )
        fig_err.update_layout(
            xaxis_title="",
            yaxis_title="Forecast – Actual ($)",
            legend_title="",
            margin=dict(l=10, r=10, t=30, b=10),
            title="",
        )
        st.plotly_chart(fig_err, width='stretch')
    else:
        st.info("No historical actuals to show error trend for this window.")

    # --------------------------------------------------
    # Labour hours
    # --------------------------------------------------
    st.subheader("Labour hours: actual vs forecast")

    labour_cols = ["HoursActual", "HoursForecast", "Hybrid_Forecast_Labour"]
    df_lab = df[["Date"] + labour_cols].copy()

    rename_lab = {
        "HoursActual": "Actual Labour Hours",
        "HoursForecast": "Mecca Forecast Labour Hours",
        "Hybrid_Forecast_Labour": "Hybrid Forecast Labour Hours",
    }
    df_lab = df_lab.rename(columns=rename_lab)

    lab_long = df_lab.melt(
        id_vars="Date", var_name="Series", value_name="Hours"
    )
    fig_lab = px.line(
        lab_long,
        x="Date",
        y="Hours",
        color="Series",
        color_discrete_map=COLOR_MAP,
    )
    fig_lab.update_layout(
        xaxis_title="",
        yaxis_title="Hours",
        legend_title="",
        margin=dict(l=10, r=10, t=30, b=10),
        title="",
    )
    st.plotly_chart(fig_lab, width='stretch')

    st.markdown("---")

    # --------------------------------------------------
    # Model comparison bar chart + metrics table
    # --------------------------------------------------
    st.subheader("Model comparison – error metrics")

    # Bar chart for MAPE
    fig_mape = px.bar(
        metrics_df.reset_index(),
        x="Model",
        y="MAPE (%)",
        text=metrics_df.reset_index()["MAPE (%)"].map(lambda x: f"{x:.1f}%"),
    )
    fig_mape.update_traces(textposition="outside")
    fig_mape.update_layout(
        yaxis_title="MAPE (%)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_mape, width='stretch')

    st.markdown(
        "**Note:** Positive values in 'Improvement vs Mecca (%)' indicate lower MAE than the MECCA forecast."
    )
    st.dataframe(
        metrics_df.style.format(
            {
                "MAE": "${:,.0f}",
                "RMSE": "${:,.0f}",
                "MAPE (%)": "{:.1f}",
                "Avg Error": "${:,.0f}",
                "Improvement vs Mecca (%)": "{:.1f}",
            }
        ),
        width='content',
    )

    # --------------------------------------------------
    # Narrative: why this matters
    # --------------------------------------------------
    uplift_text = ""
    if primary_model != "Mecca" and "Mecca" in metrics_df.index:
        primary_mae = metrics_df.loc[primary_model, "MAE"]
        mecca_mae = metrics_df.loc["Mecca", "MAE"]
        total_days = df_hist["Date"].nunique()
        if mecca_mae and not np.isnan(mecca_mae):
            uplift_pct = (mecca_mae - primary_mae) / mecca_mae * 100
            uplift_dollars = (mecca_mae - primary_mae) * total_days
            uplift_text = (
                f"Over {total_days} trading days, this reduction in daily error "
                f"equates to around **${uplift_dollars:,.0f}** less uncertainty "
                f"in total sales – a clearer signal for staffing and inventory decisions."
            )

    st.markdown(
        f"""
        <div style="font-weight:600; margin-top:18px; margin-bottom:6px; font-size:24px;">
            Forecasting insights for {location}
        </div>
        """,
        unsafe_allow_html=True,
    )

    insight_md = f"""
- With the current filters, **{primary_model}** delivers a MAPE of **{mape:.1f}%** and an average absolute error of **${mae:,.0f}** per day.
- Special events contribute about **{event_share:.1f}%** of total sales, so accurate event forecasting is critical to avoid over- or under-staffing on those peak days.
"""
    if uplift_text:
        insight_md += f"- Compared with the MECCA forecast, {uplift_text}\n"

    st.markdown(insight_md)

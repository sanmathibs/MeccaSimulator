import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import datetime
from utils.holiday_utils import get_holiday


# ---------- LOAD DATA ----------

def load_store(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    cols = ["SalesActual", "HoursActual", "SalesForecast", "HoursForecast"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


# ---------- KPI CALCULATION ----------

def compute_kpis(df: pd.DataFrame) -> dict:
    """Core KPI metrics for a filtered store dataframe."""
    if df.empty:
        return {
            "Total Sales": 0.0,
            "Total Labour Hours": 0.0,
            "Sales per Labour Hour": np.nan,
            "Forecast MAPE %": np.nan,
            "Forecast Bias %": np.nan,
            "Average Daily Sales": 0.0,
        }

    total_sales = df["SalesActual"].sum()
    total_hours = df["HoursActual"].sum()
    sales_per_hour = total_sales / total_hours if total_hours else np.nan

    df_valid = df[df["SalesActual"] > 0].copy()
    abs_err = (df_valid["SalesActual"] - df_valid["SalesForecast"]).abs()
    mape = (abs_err / df_valid["SalesActual"]).mean() * 100

    signed_err = df_valid["SalesForecast"] - df_valid["SalesActual"]
    bias_pct = (signed_err / df_valid["SalesActual"]).mean() * 100

    avg_daily_sales = df.groupby("Date")["SalesActual"].sum().mean()

    return {
        "Total Sales": total_sales,
        "Total Labour Hours": total_hours,
        "Sales per Labour Hour": sales_per_hour,
        "Forecast MAPE %": mape,
        "Forecast Bias %": bias_pct,
        "Average Daily Sales": avg_daily_sales,
    }


def display_kpis(store_a_name, df_a, store_b_name=None, df_b=None):
    """Elegant KPI cards for single store or comparison."""
    st.subheader("KPI summary")

    kpis_a = compute_kpis(df_a)

    def format_money(v):
        return "–" if np.isnan(v) else f"${v:,.0f}"

    def format_pct(v):
        return "–" if np.isnan(v) else f"{v:,.1f}%"

    if df_b is None or store_b_name is None:
        # ------- Single-store cards -------
        col1, col2, col3, col4 = st.columns(4)

        def card(col, title, value, subtitle=None):
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
                    box-shadow:2px 2px 8px rgba(0,0,0,0.06);
                    min-height:90px;
                ">
                    <div style="font-size:13px;color:#555;margin-bottom:4px;">{title}</div>
                    <div style="font-size:20px;font-weight:700;">{value}</div>
                    {subtitle_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        card(
            col1,
            "Total sales",
            format_money(kpis_a["Total Sales"]),
            store_a_name,
        )
        card(
            col2,
            "Sales per labour hour",
            format_money(kpis_a["Sales per Labour Hour"]),
        )
        card(
            col3,
            "Forecast MAPE",
            format_pct(kpis_a["Forecast MAPE %"]),
        )
        card(
            col4,
            "Forecast bias",
            format_pct(kpis_a["Forecast Bias %"]),
            "Positive = over-forecast",
        )

    else:
        # ------- Comparison cards (A vs B) -------
        kpis_b = compute_kpis(df_b)

        metrics = [
            ("Total sales", "Total Sales", format_money),
            ("Sales per labour hour", "Sales per Labour Hour", format_money),
            ("Forecast MAPE", "Forecast MAPE %", format_pct),
            ("Forecast bias", "Forecast Bias %", format_pct),
        ]

        cols = st.columns(4)

        for (title, key, fmt), col in zip(metrics, cols):
            a_val = kpis_a[key]
            b_val = kpis_b[key]

            diff = None
            if not (np.isnan(a_val) or np.isnan(b_val) or b_val == 0):
                if "MAPE" in title or "bias" in title:
                    diff = (a_val - b_val)  # lower is better
                else:
                    diff = (a_val / b_val - 1) * 100

            diff_str = ""
            if diff is not None and not np.isnan(diff):
                if "MAPE" in title or "bias" in title:
                    # show absolute difference in percentage points
                    sign = "+" if diff >= 0 else ""
                    diff_str = f"{sign}{diff:,.1f} pts vs {store_b_name}"
                else:
                    sign = "+" if diff >= 0 else ""
                    diff_str = f"{sign}{diff:,.1f}% vs {store_b_name}"

            col.markdown(
                f"""
                <div style="
                    background-color:#ffffff;
                    padding:14px 16px;
                    border-radius:10px;
                    box-shadow:2px 2px 8px rgba(0,0,0,0.06);
                    min-height:110px;
                ">
                    <div style="font-size:13px;color:#555;margin-bottom:4px;">{title}</div>
                    <div style="font-size:20px;font-weight:700;margin-bottom:4px;">
                        {store_a_name}: {fmt(a_val)}
                    </div>
                    <div style="font-size:13px;color:#555;">
                        {store_b_name}: {fmt(b_val)}<br/>
                        <span style="color:#777;">{diff_str}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------- CHARTS ----------

def plot_actual_vs_forecast(
    store_a,
    store_b=None,
    store_a_name="Store A",
    store_b_name="Store B",
    start_date=None,
    end_date=None,
):
    st.subheader("Actual sales vs forecast (daily)")

    def prepare_df(df):
        """Build a long-format dataframe, scaling the axis to the filtered data window."""
        if df.empty:
            return pd.DataFrame(columns=["Date", "Type", "Amount"])

        # Limit the visual range to where we actually have data
        start = df["Date"].min()
        end = df["Date"].max()

        all_dates = pd.date_range(start=start, end=end)
        df_full = pd.DataFrame({"Date": all_dates})
        df_merged = df_full.merge(df, on="Date", how="left")

        df_long = df_merged.melt(
            id_vars="Date",
            value_vars=["SalesActual", "SalesForecast"],
            var_name="Type",
            value_name="Amount",
        )
        return df_long

    df_a_long = prepare_df(store_a)
    fig_a = px.line(
        df_a_long,
        x="Date",
        y="Amount",
        color="Type",
        labels={"Amount": "", "Type": "", "Date": ""},
        title=f"{store_a_name}",
    )
    fig_a.update_traces(connectgaps=False)
    fig_a.update_layout(
        title=dict(text=f"{store_a_name}", x=0.5, xanchor="center", yanchor="top")
    )

    if store_b is not None:
        df_b_long = prepare_df(store_b)
        fig_b = px.line(
            df_b_long,
            x="Date",
            y="Amount",
            color="Type",
            labels={"Amount": "", "Type": "", "Date": ""},
            title=f"{store_b_name}",
        )
        fig_b.update_traces(connectgaps=False)
        fig_b.update_layout(
            title=dict(text=f"{store_b_name}", x=0.5, xanchor="center", yanchor="top")
        )

        col1, col2 = st.columns(2)
        col1.plotly_chart(
            fig_a,
            use_container_width=True,
            key=f"line_{store_a_name}_{np.random.randint(1e6)}",
        )
        col2.plotly_chart(
            fig_b,
            use_container_width=True,
            key=f"line_{store_b_name}_{np.random.randint(1e6)}",
        )
    else:
        st.plotly_chart(
            fig_a,
            use_container_width=True,
            key=f"line_{store_a_name}_{np.random.randint(1e6)}",
        )


def plot_avg_sales_by_dow(
    store_a, store_b=None, store_a_name="Store A", store_b_name="Store B"
):
    st.subheader("Average sales by day of week")

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    a = store_a.copy()
    a["DayOfWeek"] = a["Date"].dt.day_name()
    avg_a = (
        a.groupby("DayOfWeek")["SalesActual"]
        .mean()
        .reindex(order)
    )

    fig_a = px.bar(
        x=avg_a.index,
        y=avg_a.values,
        labels={"x": "", "y": ""},
        title=f"{store_a_name}",
    )
    fig_a.update_layout(
        title=dict(text=f"{store_a_name}", x=0.5, xanchor="center", yanchor="top")
    )

    if store_b is not None:
        b = store_b.copy()
        b["DayOfWeek"] = b["Date"].dt.day_name()
        avg_b = (
            b.groupby("DayOfWeek")["SalesActual"]
            .mean()
            .reindex(order)
        )
        fig_b = px.bar(
            x=avg_b.index,
            y=avg_b.values,
            labels={"x": "", "y": ""},
            title=f"{store_b_name}",
        )
        fig_b.update_layout(
            title=dict(
                text=f"{store_b_name}",
                x=0.5,
                xanchor="center",
                yanchor="top",
            )
        )
        col1, col2 = st.columns(2)
        col1.plotly_chart(
            fig_a,
            use_container_width=True,
            key=f"bar_{store_a_name}_{np.random.randint(1e6)}",
        )
        col2.plotly_chart(
            fig_b,
            use_container_width=True,
            key=f"bar_{store_b_name}_{np.random.randint(1e6)}",
        )
    else:
        st.plotly_chart(
            fig_a,
            use_container_width=True,
            key=f"bar_{store_a_name}_{np.random.randint(1e6)}",
        )


def plot_error_heatmap(
    store_a, store_b=None, store_a_name="Store A", store_b_name="Store B"
):
    st.subheader("Forecast error heatmap (Actual – Forecast)")

    def build_heatmap_df(df):
        d = df.copy()
        d["Error"] = d["SalesActual"] - d["SalesForecast"]
        d["WeekOfYear"] = d["Date"].dt.isocalendar().week
        d["DayOfWeek"] = d["Date"].dt.day_name()

        pivot = d.pivot_table(
            index="WeekOfYear", columns="DayOfWeek", values="Error", aggfunc="mean"
        ).reindex(
            columns=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        return pivot

    pivot_a = build_heatmap_df(store_a)
    fig_a = px.imshow(
        pivot_a,
        labels=dict(x="", y="Week of year", color=""),
        aspect="auto",
        text_auto=False,
        title=f"{store_a_name}",
    )
    fig_a.update_layout(
        title=dict(text=f"{store_a_name}", x=0.5, xanchor="center", yanchor="top")
    )

    if store_b is not None:
        pivot_b = build_heatmap_df(store_b)
        fig_b = px.imshow(
            pivot_b,
            labels=dict(x="", y="Week of year", color=""),
            aspect="auto",
            text_auto=False,
            title=f"{store_b_name}",
        )
        fig_b.update_layout(
            title=dict(
                text=f"{store_b_name}",
                x=0.5,
                xanchor="center",
                yanchor="top",
            )
        )
        col1, col2 = st.columns(2)
        col1.plotly_chart(
            fig_a,
            use_container_width=True,
            key=f"heatmap_{store_a_name}_{np.random.randint(1e6)}",
        )
        col2.plotly_chart(
            fig_b,
            use_container_width=True,
            key=f"heatmap_{store_b_name}_{np.random.randint(1e6)}",
        )
    else:
        st.plotly_chart(
            fig_a,
            use_container_width=True,
            key=f"heatmap_{store_a_name}_{np.random.randint(1e6)}",
        )


def plot_actual_vs_forecast_event(
    store_a, store_b=None, store_a_name="Store A", store_b_name="Store B"
):
    st.subheader("Actual vs forecast on special-event days")

    df_a_long = store_a.melt(
        id_vars="Date",
        value_vars=["SalesActual", "SalesForecast"],
        var_name="Type",
        value_name="Amount",
    )

    fig_a = px.bar(
        df_a_long,
        x="Date",
        y="Amount",
        color="Type",
        barmode="stack",
        labels={"Amount": "", "Type": "", "Date": ""},
        title=f"{store_a_name}",
    )
    fig_a.update_layout(
        title=dict(text=f"{store_a_name}", x=0.5, xanchor="center", yanchor="top")
    )

    if store_b is not None:
        df_b_long = store_b.melt(
            id_vars="Date",
            value_vars=["SalesActual", "SalesForecast"],
            var_name="Type",
            value_name="Amount",
        )
        fig_b = px.bar(
            df_b_long,
            x="Date",
            y="Amount",
            color="Type",
            barmode="stack",
            labels={"Amount": "", "Type": "", "Date": ""},
            title=f"{store_b_name}",
        )
        fig_b.update_layout(
            title=dict(text=f"{store_b_name}", x=0.5, xanchor="center", yanchor="top")
        )

        col1, col2 = st.columns(2)
        col1.plotly_chart(
            fig_a,
            use_container_width=True,
            key=f"bar_event_{store_a_name}_{np.random.randint(1e6)}",
        )
        col2.plotly_chart(
            fig_b,
            use_container_width=True,
            key=f"bar_event_{store_b_name}_{np.random.randint(1e6)}",
        )
    else:
        st.plotly_chart(
            fig_a,
            use_container_width=True,
            key=f"bar_event_{store_a_name}_{np.random.randint(1e6)}",
        )


def plot_sales_vs_hours_scatter(
    store_a, store_b=None, store_a_name="Store A", store_b_name="Store B"
):
    """New: sales vs labour hours scatter to highlight productivity and outliers."""
    st.subheader("Sales vs labour hours (per day)")

    a = store_a.copy()
    a["StoreLabel"] = store_a_name

    frames = [a]
    if store_b is not None:
        b = store_b.copy()
        b["StoreLabel"] = store_b_name
        frames.append(b)

    df_plot = pd.concat(frames, ignore_index=True)

    fig = px.scatter(
        df_plot,
        x="HoursActual",
        y="SalesActual",
        color="StoreLabel",
        labels={"HoursActual": "Labour hours", "SalesActual": "Sales"},
    )
    fig.update_layout(
        legend_title="Store",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"scatter_{store_a_name}_{np.random.randint(1e6)}",
    )


# ---------- INSIGHTS ----------

def weekend_uplift(df: pd.DataFrame) -> float:
    """Weekend vs weekday uplift in sales (%)."""
    if df.empty:
        return np.nan
    d = df.copy()
    d["weekday"] = d["Date"].dt.weekday
    weekend = d[d["weekday"] >= 5]["SalesActual"].mean()
    weekday = d[d["weekday"] < 5]["SalesActual"].mean()
    if np.isnan(weekend) or np.isnan(weekday) or weekday == 0:
        return np.nan
    return (weekend / weekday - 1) * 100


def render_store_insights(store_a, store_b, store_a_name, store_b_name=None):
    """Short narrative tying store performance back to forecasting."""
    kpi_a = compute_kpis(store_a)
    uplift_a = weekend_uplift(store_a)

    if store_b is not None and not store_b.empty and store_b_name is not None:
        kpi_b = compute_kpis(store_b)
        uplift_b = weekend_uplift(store_b)

        st.markdown(
            """
            <div style="font-weight:600; margin-top:18px; margin-bottom:6px; font-size:14px;">
                Store insights
            </div>
            """,
            unsafe_allow_html=True,
        )

        html = f"""
        <ul style="margin-top:0; font-size:13px;">
            <li>
                Over the selected period, <strong>{store_a_name}</strong> generated
                <strong>${kpi_a['Total Sales']:,.0f}</strong> of sales from
                <strong>{kpi_a['Total Labour Hours']:,.0f}</strong> labour hours
                (≈ <strong>${kpi_a['Sales per Labour Hour']:,.0f}</strong> per hour),
                compared with <strong>${kpi_b['Total Sales']:,.0f}</strong> and
                <strong>{kpi_b['Total Labour Hours']:,.0f}</strong> hours at
                <strong>{store_b_name}</strong>.
            </li>
            <li>
                Forecast accuracy (MAPE) is <strong>{kpi_a['Forecast MAPE %']:,.1f}%</strong>
                for {store_a_name} vs <strong>{kpi_b['Forecast MAPE %']:,.1f}%</strong> for {store_b_name},
                indicating where better forecasting can drive rostering decisions.
            </li>
            <li>
                Weekend trading is about <strong>{uplift_a:,.1f}%</strong> higher than weekdays at {store_a_name}
                and <strong>{uplift_b:,.1f}%</strong> at {store_b_name}, highlighting the need to flex team
                rosters to match weekend peaks.
            </li>
        </ul>
        """
        st.markdown(html, unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div style="font-weight:650; margin-top:18px; margin-bottom:6px; font-size:24px;">
                Insights
            </div>
            """,
            unsafe_allow_html=True,
        )

        html = f"""
        <ul style="margin-top:0; font-size:15px;">
            <li>
                Over the selected period, <strong>{store_a_name}</strong> generated
                <strong>${kpi_a['Total Sales']:,.0f}</strong> of sales from
                <strong>{kpi_a['Total Labour Hours']:,.0f}</strong> labour hours – around
                <strong>${kpi_a['Sales per Labour Hour']:,.0f} per labour hour</strong>.
            </li>
            <li>
                The MECCA forecast achieves a MAPE of
                <strong>{kpi_a['Forecast MAPE %']:,.1f}%</strong> at this store with a bias of
                <strong>{kpi_a['Forecast Bias %']:,.1f}%</strong>
                (positive values indicate over-forecasting).
            </li>
            <li>
                Weekend trading is roughly
                <strong>{uplift_a:,.1f}%</strong> higher than weekdays, reinforcing the need
                for higher staffing and careful forecasting on Saturdays and Sundays.
            </li>
        </ul>
        """
        st.markdown(html, unsafe_allow_html=True)


# ---------- MAIN RENDER ----------

def render():
    df = load_store("data/all_stores.xlsx")
    st.header("Store explorer")

    # ============================================================
    # FILTERS
    # ============================================================

    st.subheader("Filters")

    col_date1, col_date2, col_months, col_event = st.columns([1, 1, 1, 1])

    default_start = datetime.date(2025, 9, 25)
    default_end = datetime.date(2025, 10, 25)

    with col_date1:
        start_date_select = st.date_input(
            "Start date",
            value=default_start,
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date(),
        )

    with col_date2:
        end_date_select = st.date_input(
            "End date",
            value=default_end,
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date(),
        )

    with col_months:
        months_selected = st.multiselect("Months", list(range(1, 13)), key="months")

    with col_event:
        event_filter = st.selectbox(
            "Event type",
            ["All days", "Only special events", "Non-event days"],
            index=0,
        )

    st.markdown("---")

    col_mode, col_storeA, col_storeB = st.columns([1, 1, 1])

    with col_mode:
        compare_mode = st.radio(
            "Mode",
            ["Single store", "Compare stores"],
            index=0,
            horizontal=True,
        )

    stores = df["Store"].unique()

    with col_storeA:
        store_a_name = st.selectbox("Store A", stores, key="store_a_main")

    with col_storeB:
        if compare_mode == "Compare stores":
            store_b_name = st.selectbox("Store B", stores, key="store_b_main")
        else:
            store_b_name = None

    st.markdown("---")

    # ============================================================
    #  FILTER FUNCTION
    # ============================================================

    def filter_store(df_store, store_name):
        temp = df_store[df_store["Store"] == store_name].copy()

        temp = temp[
            (temp["Date"].dt.date >= start_date_select)
            & (temp["Date"].dt.date <= end_date_select)
        ]

        if months_selected:
            temp = temp[temp["Date"].dt.month.isin(months_selected)]

        if event_filter != "All days":
            temp["holiday_name"] = temp["Date"].apply(get_holiday)

            if event_filter == "Only special events":
                temp = temp[temp["holiday_name"].notna()]
            elif event_filter == "Non-event days":
                temp = temp[temp["holiday_name"].isna()]

            temp = temp.drop(columns=["holiday_name"])

        return temp

    # ============================================================
    #  APPLY FILTERS
    # ============================================================

    store_a = filter_store(df, store_a_name)
    store_b = filter_store(df, store_b_name) if store_b_name else None

    # ============================================================
    #  SPECIAL-EVENT TABLE
    # ============================================================

    if event_filter == "Only special events":
        st.subheader(f"Special-event days for {store_a_name}")
        st.dataframe(store_a[["Date", "SalesActual", "SalesForecast"]])

        if store_b is not None:
            st.subheader(f"Special-event days for {store_b_name}")
            st.dataframe(store_b[["Date", "SalesActual", "SalesForecast"]])

    # ============================================================
    #  KPI + CHARTS + INSIGHTS
    # ============================================================

    display_kpis(store_a_name, store_a, store_b_name, store_b)

    if event_filter == "Only special events":
        plot_actual_vs_forecast_event(store_a, store_b, store_a_name, store_b_name)
    else:
        plot_actual_vs_forecast(
            store_a,
            store_b,
            store_a_name,
            store_b_name,
            start_date_select,
            end_date_select,
        )

    plot_avg_sales_by_dow(store_a, store_b, store_a_name, store_b_name)
    plot_error_heatmap(store_a, store_b, store_a_name, store_b_name)
    plot_sales_vs_hours_scatter(store_a, store_b, store_a_name, store_b_name)

    render_store_insights(store_a, store_b, store_a_name, store_b_name)

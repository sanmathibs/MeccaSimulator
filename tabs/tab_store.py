import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime
from utils.holiday_utils import get_holiday


# ---------- LOAD & CORE METRICS ----------

def load_store(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    cols = ["SalesActual", "HoursActual", "SalesForecast", "HoursForecast"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


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


# ---------- SMALL HELPERS ----------

def fmt_money(v, zero_as_dash=False):
    if np.isnan(v):
        return "–"
    if zero_as_dash and v == 0:
        return "–"
    return f"${v:,.0f}"


def fmt_pct(v, zero_as_dash=False, decimals=1):
    if np.isnan(v):
        return "–"
    if zero_as_dash and abs(v) < 1e-9:
        return "–"
    return f"{v:,.{decimals}f}%"


def describe_days_filter(days_selected):
    all_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    if not days_selected or len(days_selected) == 7:
        return "all days"
    if set(days_selected) == {"Saturday", "Sunday"}:
        return "weekends only"
    if set(days_selected) == set(all_days) - {"Saturday", "Sunday"}:
        return "weekdays only (Mon–Fri)"
    return ", ".join(days_selected)


# ---------- KPI CARDS ----------

def display_kpis(store_a_name, df_a, store_b_name=None, df_b=None):
    """Elegant KPI cards for single store or comparison."""
    st.subheader("KPI summary")

    kpis_a = compute_kpis(df_a)

    def kpi_card(col, title, value, subtitle=None, tone: str = "neutral"):
        tone_styles = {
            "neutral": {"bg": "#ffffff", "border": "#e5e7eb"},
            "good": {"bg": "#e8f5e9", "border": "#c8e6c9"},
            "warn": {"bg": "#fff8e1", "border": "#ffe0b2"},
            "bad": {"bg": "#ffebee", "border": "#ffcdd2"},
        }
        style = tone_styles.get(tone, tone_styles["neutral"])

        subtitle_html = (
            f"<div style='font-size:12px;color:#6b7280;margin-top:2px;'>{subtitle}</div>"
            if subtitle
            else ""
        )
        col.markdown(
            f"""
            <div style="
                background-color:{style['bg']};
                border:1px solid {style['border']};
                padding:14px 16px;
                border-radius:12px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.04);
                min-height:100px;
            ">
                <div style="font-size:13px;color:#4b5563;margin-bottom:4px;">{title}</div>
                <div style="font-size:20px;font-weight:700;">{value}</div>
                {subtitle_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Single-store ----------
    if df_b is None or store_b_name is None:
        col1, col2, col3, col4 = st.columns(4)

        # Determine tones
        mape = kpis_a["Forecast MAPE %"]
        bias = kpis_a["Forecast Bias %"]

        if np.isnan(mape):
            mape_tone = "neutral"
        elif mape <= 15:
            mape_tone = "good"
        elif mape <= 25:
            mape_tone = "warn"
        else:
            mape_tone = "bad"

        if np.isnan(bias):
            bias_tone = "neutral"
        elif abs(bias) <= 5:
            bias_tone = "good"
        elif abs(bias) <= 15:
            bias_tone = "warn"
        else:
            bias_tone = "bad"

        kpi_card(
            col1,
            "Total sales",
            fmt_money(kpis_a["Total Sales"]),
            subtitle=store_a_name,
        )
        kpi_card(
            col2,
            "Sales per labour hour",
            fmt_money(kpis_a["Sales per Labour Hour"]),
        )
        kpi_card(
            col3,
            "Forecast MAPE",
            fmt_pct(kpis_a["Forecast MAPE %"]),
            tone=mape_tone,
        )
        kpi_card(
            col4,
            "Forecast bias",
            fmt_pct(kpis_a["Forecast Bias %"]),
            subtitle="Positive = over-forecast",
            tone=bias_tone,
        )

    # ---------- Comparison mode ----------
    else:
        kpis_b = compute_kpis(df_b)
        metrics = [
            ("Total sales", "Total Sales", fmt_money, False),
            ("Sales per labour hour", "Sales per Labour Hour", fmt_money, False),
            ("Forecast MAPE", "Forecast MAPE %", fmt_pct, True),
            ("Forecast bias", "Forecast Bias %", fmt_pct, True),
        ]
        cols = st.columns(4)

        for (title, key, fmt_fn, lower_is_better), col in zip(metrics, cols):
            a_val = kpis_a[key]
            b_val = kpis_b[key]

            diff = None
            if not (np.isnan(a_val) or np.isnan(b_val) or b_val == 0):
                if lower_is_better:
                    diff = a_val - b_val  # negative is better for A
                else:
                    diff = (a_val / b_val - 1) * 100

            # Diff text & tone
            diff_str = ""
            tone = "neutral"
            if diff is not None and not np.isnan(diff):
                if lower_is_better:
                    # absolute difference in pts
                    sign = "+" if diff >= 0 else ""
                    diff_str = f"{sign}{diff:,.1f} pts vs {store_b_name}"
                    tone = "good" if diff < 0 else "bad" if abs(diff) > 5 else "warn"
                else:
                    sign = "+" if diff >= 0 else ""
                    diff_str = f"{sign}{diff:,.1f}% vs {store_b_name}"
                    tone = "good" if diff > 0 else "bad" if diff < -5 else "warn"

            kpi_card(
                col,
                title,
                f"{store_a_name}: {fmt_fn(a_val)}",
                subtitle=f"{store_b_name}: {fmt_fn(b_val)}<br/><span style='color:#6b7280;'>{diff_str}</span>",
                tone=tone,
            )


# ---------- CHARTS ----------

def plot_actual_vs_forecast(
    store_a,
    store_b=None,
    store_a_name="Store A",
    store_b_name="Store B",
):
    st.subheader("Actual sales vs forecast (daily)")

    def prepare_df(df):
        if df.empty:
            return pd.DataFrame(columns=["Date", "Type", "Amount"])
        df_long = df.melt(
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
            key=f"line_actual_vs_forecast_{store_a_name}_A",
        )
        col2.plotly_chart(
            fig_b,
            use_container_width=True,
            key=f"line_actual_vs_forecast_{store_b_name}_B",
        )
    else:
        st.plotly_chart(
            fig_a,
            use_container_width=True,
            key=f"line_actual_vs_forecast_{store_a_name}_single",
        )


def plot_avg_sales_by_dow(
    store_a, store_b=None, store_a_name="Store A", store_b_name="Store B"
):
    st.subheader("Average sales by day of week")

    order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    def build_fig(df, title):
        if df.empty:
            return None
        tmp = df.copy()
        tmp["DayOfWeek"] = tmp["Date"].dt.day_name()
        avg = (
            tmp.groupby("DayOfWeek")["SalesActual"]
            .mean()
            .reindex(order)
            .fillna(0)
        )
        df_avg = pd.DataFrame(
            {"DayOfWeek": avg.index, "AvgSales": avg.values}
        )
        fig = px.bar(
            df_avg,
            x="DayOfWeek",
            y="AvgSales",
            color="AvgSales",
            color_continuous_scale=["#e0f2fe", "#1d4ed8"],
            labels={"DayOfWeek": "", "AvgSales": ""},
            title=title,
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", yanchor="top"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    fig_a = build_fig(store_a, store_a_name)

    if store_b is not None:
        fig_b = build_fig(store_b, store_b_name)
        col1, col2 = st.columns(2)
        if fig_a is not None:
            col1.plotly_chart(
                fig_a,
                use_container_width=True,
                key=f"dow_avg_{store_a_name}_A",
            )
        if fig_b is not None:
            col2.plotly_chart(
                fig_b,
                use_container_width=True,
                key=f"dow_avg_{store_b_name}_B",
            )
    else:
        if fig_a is not None:
            st.plotly_chart(
                fig_a,
                use_container_width=True,
                key=f"dow_avg_{store_a_name}_single",
            )


def plot_actual_vs_forecast_event(
    store_a, store_b=None, store_a_name="Store A", store_b_name="Store B"
):
    st.subheader("Actual vs forecast on special-event days")

    def build_fig(df, title):
        if df.empty:
            return None
        df_long = df.melt(
            id_vars="Date",
            value_vars=["SalesActual", "SalesForecast"],
            var_name="Type",
            value_name="Amount",
        )
        fig = px.bar(
            df_long,
            x="Date",
            y="Amount",
            color="Type",
            barmode="stack",
            labels={"Amount": "", "Type": "", "Date": ""},
            title=title,
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", yanchor="top"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    fig_a = build_fig(store_a, store_a_name)

    if store_b is not None:
        fig_b = build_fig(store_b, store_b_name)
        col1, col2 = st.columns(2)
        if fig_a is not None:
            col1.plotly_chart(
                fig_a,
                use_container_width=True,
                key=f"event_af_{store_a_name}_A",
            )
        if fig_b is not None:
            col2.plotly_chart(
                fig_b,
                use_container_width=True,
                key=f"event_af_{store_b_name}_B",
            )
    else:
        if fig_a is not None:
            st.plotly_chart(
                fig_a,
                use_container_width=True,
                key=f"event_af_{store_a_name}_single",
            )


def plot_sales_vs_hours_scatter(
    store_a, store_b=None, store_a_name="Store A", store_b_name="Store B"
):
    """Sales vs labour hours scatter with per-store trend lines."""
    st.subheader("Sales vs labour hours (per day)")

    frames = []
    if not store_a.empty:
        a = store_a.copy()
        a["StoreLabel"] = store_a_name
        frames.append(a)
    if store_b is not None and not store_b.empty:
        b = store_b.copy()
        b["StoreLabel"] = store_b_name
        frames.append(b)

    if not frames:
        st.info("No data to display for this chart.")
        return

    df_plot = pd.concat(frames, ignore_index=True)
    df_plot = df_plot[df_plot["HoursActual"] > 0].copy()
    df_plot["SalesPerHour"] = df_plot["SalesActual"] / df_plot["HoursActual"]
    df_plot["DayType"] = np.where(df_plot["Date"].dt.weekday >= 5, "Weekend", "Weekday")

    color_map = {
        store_a_name: "#1d4ed8",  # blue
        store_b_name: "#f97316" if store_b_name else "#f97316",  # orange
    }

    fig = px.scatter(
        df_plot,
        x="HoursActual",
        y="SalesActual",
        color="StoreLabel",
        symbol="DayType",
        color_discrete_map=color_map,
        labels={"HoursActual": "Labour hours", "SalesActual": "Sales"},
        hover_data={
            "Date": True,
            "SalesPerHour": ":.0f",
            "DayType": True,
            "StoreLabel": False,
        },
    )

    # Add simple trend line per store
    for store_label, color in color_map.items():
        d = df_plot[df_plot["StoreLabel"] == store_label]
        if d.empty or d["HoursActual"].nunique() < 2:
            continue
        x = d["HoursActual"].values
        y = d["SalesActual"].values
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = coef[0] * x_line + coef[1]
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=color, width=2, dash="dot"),
                name=f"{store_label} trend",
                showlegend=True,
            )
        )

    fig.update_layout(
        legend_title="Store",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"scatter_sales_hours_{store_a_name}_{store_b_name or 'single'}",
    )

    # Small caption with correlation
    lines = []
    for store_label in df_plot["StoreLabel"].unique():
        d = df_plot[df_plot["StoreLabel"] == store_label]
        if len(d) >= 2:
            corr = d["HoursActual"].corr(d["SalesActual"])
            if not np.isnan(corr):
                lines.append(f"{store_label}: corr(hours, sales) = {corr:,.2f}")
    if lines:
        st.caption("Relationship between labour and sales · " + " · ".join(lines))


# ---------- INSIGHTS ----------

def render_store_insights(
    store_a,
    store_b,
    store_a_name,
    store_b_name=None,
    days_filter=None,
    event_filter=None,
):
    """Narrative insights, aligned with filters."""
    kpi_a = compute_kpis(store_a)
    uplift_a = weekend_uplift(store_a)

    if store_a.empty:
        return

    period_text = (
        f"{store_a['Date'].min().date()} – {store_a['Date'].max().date()}"
    )
    days_text = describe_days_filter(days_filter)
    event_text = ""
    if event_filter == "Only special events":
        event_text = " on special-event days"
    elif event_filter == "Non-event days":
        event_text = " on non-event days"

    st.markdown(
        """
        <div style="font-weight:600; margin-top:18px; margin-bottom:6px; font-size:16px;">
            Store insights
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Comparison mode ----------
    if store_b is not None and not store_b.empty and store_b_name is not None:
        kpi_b = compute_kpis(store_b)
        uplift_b = weekend_uplift(store_b)

        if not np.isnan(uplift_a) and not np.isnan(uplift_b):
            bullet3 = (
                f"<li>Weekend trading is about <strong>{uplift_a:,.1f}%</strong> higher "
                f"than weekdays at {store_a_name} and "
                f"<strong>{uplift_b:,.1f}%</strong> at {store_b_name}, reinforcing the "
                "need to flex team rosters to match weekend peaks in both locations.</li>"
            )
        else:
            bullet3 = (
                "<li>The current filters do not cover both weekdays and weekends, so "
                "weekend uplift cannot be reliably compared here.</li>"
            )

        html = f"""
        <ul style="margin-top:0; font-size:14px;">
            <li>
                Between <strong>{period_text}</strong>, focusing on
                <strong>{days_text}{event_text}</strong>, <strong>{store_a_name}</strong>
                generated <strong>{fmt_money(kpi_a['Total Sales'])}</strong> of sales from
                <strong>{kpi_a['Total Labour Hours']:,.0f}</strong> labour hours
                (≈ <strong>{fmt_money(kpi_a['Sales per Labour Hour'])}</strong> per hour),
                compared with <strong>{fmt_money(kpi_b['Total Sales'])}</strong> from
                <strong>{kpi_b['Total Labour Hours']:,.0f}</strong> hours at
                <strong>{store_b_name}</strong>.
            </li>
            <li>
                Forecast accuracy (MAPE) is
                <strong>{fmt_pct(kpi_a['Forecast MAPE %'])}</strong> for {store_a_name}
                vs <strong>{fmt_pct(kpi_b['Forecast MAPE %'])}</strong> for {store_b_name},
                giving planners a clear view of where forecast quality supports rostering decisions.
            </li>
            {bullet3}
        </ul>
        """
        st.markdown(html, unsafe_allow_html=True)

    # ---------- Single store ----------
    else:
        html = f"""
        <ul style="margin-top:0; font-size:14px;">
            <li>
                Between <strong>{period_text}</strong>, focusing on
                <strong>{days_text}{event_text}</strong>, <strong>{store_a_name}</strong>
                generated <strong>{fmt_money(kpi_a['Total Sales'])}</strong> of sales from
                <strong>{kpi_a['Total Labour Hours']:,.0f}</strong> labour hours – around
                <strong>{fmt_money(kpi_a['Sales per Labour Hour'])} per labour hour</strong>.
            </li>
            <li>
                The MECCA forecast achieves a MAPE of
                <strong>{fmt_pct(kpi_a['Forecast MAPE %'])}</strong> at this store with a bias of
                <strong>{fmt_pct(kpi_a['Forecast Bias %'])}</strong>
                (positive values indicate over-forecasting), providing a useful benchmark for any
                improved models.
            </li>
        """

        if not np.isnan(uplift_a):
            html += (
                f"<li>Weekend trading is roughly <strong>{uplift_a:,.1f}%</strong> higher "
                "than weekdays, underscoring the need for higher staffing and careful "
                "forecasting on Saturdays and Sundays.</li>"
            )

        html += "</ul>"
        st.markdown(html, unsafe_allow_html=True)



# ---------- FILTER SUMMARY PILLS ----------

def render_selection_pills_store(
    store_a_name,
    store_b_name,
    start_date,
    end_date,
    days_selected,
    event_filter,
):
    days_text = describe_days_filter(days_selected)
    date_label = f"{start_date.strftime('%Y/%m/%d')} – {end_date.strftime('%Y/%m/%d')}"
    event_label = (
        "All days" if event_filter == "All days"
        else "Special-event days"
        if event_filter == "Only special events"
        else "Non-event days"
    )

    if store_b_name:
        primary_label = f"{store_a_name} vs {store_b_name}"
    else:
        primary_label = store_a_name

    html = f"""
    <div style="
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin-top:8px;
        margin-bottom:6px;
    ">
        <div style="
            padding:6px 12px;
            border-radius:999px;
            background:#f2f4ff;
            font-size:12px;
            color:#111827;
        ">
            <strong>Stores</strong> · {primary_label}
        </div>
        <div style="
            padding:6px 12px;
            border-radius:999px;
            background:#f5f5f5;
            font-size:12px;
            color:#111827;
        ">
            <strong>Dates</strong> · {date_label}
        </div>
        <div style="
            padding:6px 12px;
            border-radius:999px;
            background:#f0fbff;
            font-size:12px;
            color:#111827;
        ">
            <strong>Day(s) of week</strong> · {days_text}
        </div>
        <div style="
            padding:6px 12px;
            border-radius:999px;
            background:#e8f5e9;
            font-size:12px;
            color:#166534;
        ">
            <strong>Day type</strong> · {event_label}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------- MAIN RENDER ----------

def render():
    df = load_store("data/all_stores.xlsx")
    st.header("Store explorer")

    # ============================================================
    # SCENARIO CONTROL PANEL
    # ============================================================

    st.subheader("Scenario control panel")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    col_period, col_dow, col_event = st.columns([1, 1, 1])

    with col_period:
        period_options = ["Last 12 weeks", "Last 52 weeks", "Full history", "Custom range"]
        period_choice = st.selectbox("Period", period_options, index=1)

    with col_dow:
        dow_options = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        days_selected = st.multiselect(
            "Day(s) of week",
            options=dow_options,
            default=dow_options,
        )

    with col_event:
        event_filter = st.selectbox(
            "Day type",
            ["All days", "Only special events", "Non-event days"],
            index=0,
        )

    # Date range (driven by period, but user can fine-tune)
    if period_choice == "Last 12 weeks":
        default_start = max_date - datetime.timedelta(weeks=12)
        default_end = max_date
    elif period_choice == "Last 52 weeks":
        default_start = max_date - datetime.timedelta(weeks=52)
        default_end = max_date
    elif period_choice == "Full history":
        default_start = min_date
        default_end = max_date
    else:  # Custom
        default_start = max_date - datetime.timedelta(weeks=4)
        default_end = max_date

    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date_select = st.date_input(
            "Start date",
            value=default_start,
            min_value=min_date,
            max_value=max_date,
        )

    with col_date2:
        end_date_select = st.date_input(
            "End date",
            value=default_end,
            min_value=min_date,
            max_value=max_date,
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
    # FILTER FUNCTION
    # ============================================================

    def filter_store(df_store, store_name):
        if store_name is None:
            return pd.DataFrame(columns=df_store.columns)

        temp = df_store[df_store["Store"] == store_name].copy()

        temp = temp[
            (temp["Date"].dt.date >= start_date_select)
            & (temp["Date"].dt.date <= end_date_select)
        ]

        # Day-of-week filter
        if days_selected and len(days_selected) < 7:
            temp["DayName"] = temp["Date"].dt.day_name()
            temp = temp[temp["DayName"].isin(days_selected)]
            temp = temp.drop(columns=["DayName"])

        # Event / non-event
        if event_filter != "All days":
            temp["holiday_name"] = temp["Date"].apply(get_holiday)

            if event_filter == "Only special events":
                temp = temp[temp["holiday_name"].notna()]
            elif event_filter == "Non-event days":
                temp = temp[temp["holiday_name"].isna()]

            temp = temp.drop(columns=["holiday_name"])

        return temp

    # ============================================================
    # APPLY FILTERS
    # ============================================================

    store_a = filter_store(df, store_a_name)
    store_b = filter_store(df, store_b_name) if store_b_name else None

    # Selection summary pills
    render_selection_pills_store(
        store_a_name,
        store_b_name,
        start_date_select,
        end_date_select,
        days_selected,
        event_filter,
    )

    # ============================================================
    # SPECIAL-EVENT TABLE
    # ============================================================

    if event_filter == "Only special events":
        st.subheader(f"Special-event days")
        st.write(f"{store_a_name}")
        st.dataframe(store_a[["Date", "SalesActual", "SalesForecast"]])

        if store_b is not None:
            st.write(f"{store_b_name}")
            st.dataframe(store_b[["Date", "SalesActual", "SalesForecast"]])

    # ============================================================
    # KPI + CHARTS + INSIGHTS
    # ============================================================

    display_kpis(store_a_name, store_a, store_b_name, store_b)

    if event_filter == "Only special events":
        plot_actual_vs_forecast_event(store_a, store_b, store_a_name, store_b_name)
    else:
        plot_actual_vs_forecast(store_a, store_b, store_a_name, store_b_name)

    plot_avg_sales_by_dow(store_a, store_b, store_a_name, store_b_name)
    plot_sales_vs_hours_scatter(store_a, store_b, store_a_name, store_b_name)

    with st.expander("Store insights", expanded=True):
        render_store_insights(
            store_a,
            store_b,
            store_a_name,
            store_b_name,
            days_selected,
            event_filter,
        )

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.holiday_utils import get_holiday


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

# ---------------------------------------------------------
# Paid holiday helpers
# ---------------------------------------------------------
PAID_HOLIDAY_LABELS = {"Christmas", "New Years"}


def normalise_paid_holiday(event_name: str | None) -> str | None:
    """
    Map raw holiday names from get_holiday(...) to the labels we want to display
    for *paid* shutdown days only.

    Only returns a value for paid holidays (e.g. Christmas, New Years).
    All other events (e.g. Black Friday) return None.
    """
    if not event_name:
        return None

    name = event_name.strip().lower()
    # normalise separators
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split())

    if "christmas" in name:
        return "Christmas"
    if "new year" in name:
        return "New Years"

    # Not a paid holiday
    return None

def load_roster(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    # Ensure types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Hours Worked"] = pd.to_numeric(df["Hours Worked"], errors="coerce").fillna(0)
    df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0)

    df["DoW"] = df["DoW"].astype(str)
    return df


# ---------------------------------------------------------
# Selected-range metrics
# ---------------------------------------------------------
def compute_selected_metrics(df_role_filtered: pd.DataFrame) -> dict:
    """
    Metrics for the currently selected week range (slider).
    """
    if df_role_filtered.empty:
        return {
            "weeks_in_view": 0,
            "weeks_label": "No weeks selected",
            "total_hours_sel": 0.0,
            "total_wages_sel": 0.0,
        }

    df_tmp = df_role_filtered.copy()
    weeks = sorted(df_tmp["WoY"].unique())
    n_weeks = len(weeks)

    total_hours_sel = df_tmp["Hours Worked"].sum()
    total_wages_sel = df_tmp["Dollars"].sum()

    if n_weeks == 1:
        weeks_label = f"Week {weeks[0]}"
    else:
        weeks_label = f"Weeks {weeks[0]}–{weeks[-1]} (n={n_weeks})"

    return {
        "weeks_in_view": n_weeks,
        "weeks_label": weeks_label,
        "total_hours_sel": total_hours_sel,
        "total_wages_sel": total_wages_sel,
    }


# ---------------------------------------------------------
# KPI card helper
# ---------------------------------------------------------
def kpi_card_html(title: str, value: str, subtitle: str | None = None, tone: str = "neutral") -> str:
    """Shared HTML card component with soft tone colours."""
    tone_styles = {
        "neutral": {"bg": "#ffffff", "border": "#e5e7eb"},
        "good": {"bg": "#e8f5e9", "border": "#c8e6c9"},
        "warn": {"bg": "#fff8e1", "border": "#ffe0b2"},
        "bad": {"bg": "#ffebee", "border": "#ffcdd2"},
    }
    style = tone_styles.get(tone, tone_styles["neutral"])

    subtitle_html = (
        f"<div style='font-size:12px;color:#6b7280;margin-top:4px;'>{subtitle}</div>"
        if subtitle
        else ""
    )

    return f"""
    <div style="
        padding: 16px 18px;
        border-radius: 12px;
        background: {style['bg']};
        border: 1px solid {style['border']};
        color: #0f172a;
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.06);
        margin-bottom: 12px;
        text-align: left;
    ">
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; opacity: 0.7;">
            {title}
        </div>
        <div style="font-size: 22px; margin-top: 4px; font-weight: 700;">
            {value}
        </div>
        {subtitle_html}
    </div>
    """


# ---------------------------------------------------------
# Selected weeks KPI cards
# ---------------------------------------------------------
def render_selected_kpis(sel_metrics: dict, day_focus: str = "All days"):
    st.markdown("#### Selected weeks summary")

    col1, col2, col3 = st.columns(3)

    weeks_tone = "bad" if sel_metrics["weeks_in_view"] == 0 else "good"
    hours_tone = "neutral" if sel_metrics["total_hours_sel"] > 0 else "bad"
    wages_tone = hours_tone

    with col1:
        st.markdown(
            kpi_card_html(
                "Weeks in view",
                sel_metrics["weeks_label"],
                subtitle=f"Day focus: {day_focus.lower()}",
                tone=weeks_tone,
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            kpi_card_html(
                "Hours (selected weeks)",
                f"{sel_metrics['total_hours_sel']:.1f}",
                tone=hours_tone,
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            kpi_card_html(
                "Wages (selected weeks)",
                f"${sel_metrics['total_wages_sel']:,.2f}",
                tone=wages_tone,
            ),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# Filters
# ---------------------------------------------------------
def render_filter_section(df: pd.DataFrame):
    st.markdown("### TM selection & week range")

    all_roles = sorted(df["Role"].unique())
    all_weeks = sorted(df["WoY"].unique())
    min_week, max_week = int(min(all_weeks)), int(max(all_weeks))
    min_date = df["Date"].min()
    max_date = df["Date"].max()

    # Default: last 12 weeks in the dataset
    if len(all_weeks) >= 12:
        default_start = int(all_weeks[-12])
        default_end = int(all_weeks[-1])
    else:
        default_start, default_end = min_week, max_week

    col_role, col_weeks, col_period = st.columns([1.2, 1.4, 1.6])

    with col_role:
        role_selected = st.selectbox("Choose team member", all_roles)

    with col_weeks:
        week_range = st.slider(
            "Weeks of year",
            min_value=min_week,
            max_value=max_week,
            value=(default_start, default_end),
            step=1,
        )

    with col_period:
        st.caption(
            f"Roster period: **week {min_week}–{max_week}** "
            f"({len(all_weeks)} weeks), "
            f"{min_date:%d %b %Y} – {max_date:%d %b %Y}. "
            f"Default view shows the last 12 weeks for the selected role."
        )

    return role_selected, week_range



def render_selection_pills_roster(role_selected: str, week_range, day_focus: str):
    """Small pills summarising current roster scenario."""
    if week_range[0] == week_range[1]:
        weeks_label = f"Week {week_range[0]}"
    else:
        n_weeks = week_range[1] - week_range[0] + 1
        weeks_label = f"Weeks {week_range[0]}–{week_range[1]} (n={n_weeks})"

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
            <strong>Team member</strong> · {role_selected}
        </div>
        <div style="
            padding:6px 12px;
            border-radius:999px;
            background:#f5f5f5;
            font-size:12px;
            color:#111827;
        ">
            <strong>Weeks</strong> · {weeks_label}
        </div>
        <div style="
            padding:6px 12px;
            border-radius:999px;
            background:#e0f2fe;
            font-size:12px;
            color:#111827;
        ">
            <strong>Day focus</strong> · {day_focus}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------
# Build weekly roster table for a single role
# ---------------------------------------------------------
def generate_role_dashboard(df_role: pd.DataFrame) -> pd.DataFrame:
    """
    df_role: already filtered to a single Role (and desired week range).
    Returns one row per week with Sunday–Saturday columns + Hours + Wages.

    Date column is always the Sunday for that week.

    Rules for special days:
      - Only show a yellow holiday label when:
          * The store is shut (no actual shift times), AND
          * The event is a *paid* holiday (Christmas / New Years).
      - Unpaid events (e.g. Black Friday) just show as a normal "Off" day.
    """
    if df_role.empty:
        return pd.DataFrame()

    df_role = df_role.copy()
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    weeks = sorted(df_role["WoY"].unique())
    dashboard_rows = []

    for w in weeks:
        week_data = df_role[df_role["WoY"] == w].copy()
        if week_data.empty:
            continue

        # Use any date in the week to find the canonical Sunday for that week
        example_date = week_data["Date"].iloc[0]
        week_start = example_date.to_period("W-SAT").start_time  # Sunday start

        row = {
            "Week of year": w,
            "Date": week_start,  # will be formatted later
        }

        hours_sum = 0.0
        wages_sum = 0.0

        for i, day in enumerate(days):
            day_date = week_start + pd.Timedelta(days=i)

            # All roster entries on this calendar day
            day_data = week_data[week_data["Date"].dt.date == day_date.date()]

            # Normalise only PAID holidays (Christmas / New Years)
            raw_event = get_holiday(day_date)
            paid_label = normalise_paid_holiday(raw_event)

            if not day_data.empty:
                # Always include wages / hours even if it's a paid shutdown
                hours_sum += day_data["Hours Worked"].sum()
                wages_sum += day_data["Dollars"].sum()

                # Decide if there is any "real" shift (operation not shut)
                has_shift = False
                if {"Start Time", "End Time"}.issubset(day_data.columns):
                    has_shift = (
                        day_data["Start Time"].notna().any()
                        or day_data["End Time"].notna().any()
                    )
                else:
                    # Fallback: treat as shift if there are positive hours
                    has_shift = day_data["Hours Worked"].fillna(0).gt(0).any()

                if has_shift:
                    # Normal worked day – show shift times (green)
                    shifts = day_data.apply(
                        lambda r: f"{r['Start Time']} to {r['End Time']}",
                        axis=1,
                    )
                    cell_text = " / ".join(shifts.tolist())
                else:
                    # No actual shift times => shutdown day
                    # Only show a label if this is a PAID holiday
                    cell_text = paid_label if paid_label else "Off"
            else:
                # No roster row at all for this date
                cell_text = paid_label if paid_label else "Off"

            row[day] = cell_text

        row["Hours"] = hours_sum
        row["Wages"] = wages_sum
        dashboard_rows.append(row)

    df_dashboard = pd.DataFrame(dashboard_rows).sort_values("Week of year")
    return df_dashboard.reset_index(drop=True)




# ---------------------------------------------------------
# Metrics for full roster period
# ---------------------------------------------------------
def compute_role_metrics(df_role_full: pd.DataFrame) -> dict:
    """
    df_role_full: all roster data for the selected role (full roster period).
    """
    if df_role_full.empty:
        return {
            "type_role": "",
            "category_role": "",
            "hours_per_week": 0.0,
            "total_wages": 0.0,
            "wages_12_weeks": 0.0,
            "annualised_wages": 0.0,
            "weeks_text": "",
        }

    df_role_full = df_role_full.copy()
    weeks_all = sorted(df_role_full["WoY"].unique())
    n_weeks = len(weeks_all)

    type_role = df_role_full["Type"].iloc[0]
    category_role = df_role_full["Category"].iloc[0]

    total_hours = df_role_full["Hours Worked"].sum()
    hours_per_week = total_hours / n_weeks if n_weeks else 0.0

    total_wages = df_role_full["Dollars"].sum()

    # “12-week wages”: last 12 weeks available for that role
    last_12_weeks = weeks_all[-12:] if n_weeks > 12 else weeks_all
    wages_12_weeks = df_role_full[df_role_full["WoY"].isin(last_12_weeks)][
        "Dollars"
    ].sum()

    annualised_wages = wages_12_weeks * 4  # matches your Excel logic

    weeks_text = f"Weeks {weeks_all[0]}–{weeks_all[-1]} (n={n_weeks})"

    return {
        "type_role": type_role,
        "category_role": category_role,
        "hours_per_week": hours_per_week,
        "total_wages": total_wages,
        "wages_12_weeks": wages_12_weeks,
        "annualised_wages": annualised_wages,
        "weeks_text": weeks_text,
    }


# ---------------------------------------------------------
# Role overview cards
# ---------------------------------------------------------
def render_info_cards(metrics: dict):
    col1, col2, col3 = st.columns(3)

    # Tone for hours per week (very rough bands)
    hpw = metrics["hours_per_week"]
    if hpw == 0:
        hpw_tone = "bad"
    elif hpw < 10:
        hpw_tone = "warn"
    elif hpw <= 40:
        hpw_tone = "good"
    else:
        hpw_tone = "warn"

    with col1:
        st.markdown(
            kpi_card_html("Type", metrics["type_role"], tone="neutral"),
            unsafe_allow_html=True,
        )
        st.markdown(
            kpi_card_html("Category", metrics["category_role"], tone="neutral"),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            kpi_card_html(
                "Hours per week",
                f"{metrics['hours_per_week']:.2f}",
                tone=hpw_tone,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            kpi_card_html(
                "Total wages (roster period)",
                f"${metrics['total_wages']:,.2f}",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            kpi_card_html(
                "12-week wages",
                f"${metrics['wages_12_weeks']:,.2f}",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            kpi_card_html(
                "Annualised wages",
                f"${metrics['annualised_wages']:,.2f}",
            ),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# Table & chart rendering
# ---------------------------------------------------------
def render_roster_table(df_dashboard: pd.DataFrame):
    st.markdown(
        "<h3 style='text-align:center; margin-top: 12px;'>Roster details (weekly)</h3>",
        unsafe_allow_html=True,
    )

    if df_dashboard.empty:
        st.info("No roster data for this selection.")
        return

    df_display = df_dashboard.copy()

    # Pretty date format – always the Sunday of that week
    df_display["Date"] = df_display["Date"].dt.strftime("%A, %d %B %Y")

    # Numeric formatting
    df_display["Hours"] = df_display["Hours"].round(1)
    df_display["Wages"] = df_display["Wages"].round(2)

    day_columns = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    def style_shift_cell(val):
        if isinstance(val, str):
            text = val.strip()

            # Paid shutdown holiday (Christmas / New Years)
            if text in PAID_HOLIDAY_LABELS:
                return "background-color:#FEF3C7; color:#92400E; font-weight:600;"

            # Standard off day
            if text.lower().startswith("off"):
                return "background-color:#FEF2F2; color:#B91C1C;"

            # Worked shift
            if "to" in text:
                return "background-color:#ECFDF5; color:#065F46;"

        return ""

    styler = (
        df_display.style
        .format(
            {
                "Hours": "{:.1f}",
                "Wages": "${:,.2f}",
            }
        )
        .map(style_shift_cell, subset=day_columns)
    )

    st.dataframe(styler, width='stretch')



def render_weekly_summary_chart(df_dashboard: pd.DataFrame):
    """Combo chart: hours (bars, left axis) vs wages (line, right axis)."""
    if df_dashboard.empty:
        return

    df_chart = df_dashboard[["Week of year", "Hours", "Wages"]].copy()
    df_chart["Week of year"] = df_chart["Week of year"].astype(int)

    fig = go.Figure()

    # Bars for hours
    fig.add_trace(
        go.Bar(
            x=df_chart["Week of year"],
            y=df_chart["Hours"],
            name="Hours",
            marker_color="#2563eb",
            yaxis="y1",
        )
    )

    # Line for wages
    fig.add_trace(
        go.Scatter(
            x=df_chart["Week of year"],
            y=df_chart["Wages"],
            name="Wages",
            mode="lines+markers",
            marker=dict(size=6),
            line=dict(width=2),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Weekly hours and wages",
        xaxis=dict(title="Week of year"),
        yaxis=dict(
            title="Hours",
            rangemode="tozero",
        ),
        yaxis2=dict(
            title="Wages",
            overlaying="y",
            side="right",
            rangemode="tozero",
        ),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    st.plotly_chart(fig, width="stretch", key="roster_weekly_combo")


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def render():
    df = load_roster("data/roster_clean_data.xlsx")

    st.header("Roster explorer")

    # Filters (role + week range)
    role_selected, week_range = render_filter_section(df)

    # Full data for this role (for metrics / 12-week calc)
    df_role_full = df[df["Role"] == role_selected].copy()

    # Filtered data for table/chart based on slider range
    df_role_filtered = df_role_full[
        df_role_full["WoY"].between(week_range[0], week_range[1])
    ].copy()

    # Build weekly dashboard data and metrics
    df_dashboard = generate_role_dashboard(df_role_filtered)
    metrics = compute_role_metrics(df_role_full)
    selected_metrics = compute_selected_metrics(df_role_filtered)

    # --------------------------------------------------
    # 1. SHOW ROSTER TABLE FIRST – “When am I working?”
    # --------------------------------------------------
    st.markdown("---")
    render_roster_table(df_dashboard)

    # --------------------------------------------------
    # 2. SUMMARY CARDS + SELECTED-WEEKS METRICS
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("### Role overview")
    render_info_cards(metrics)

    if metrics["weeks_text"]:
        st.markdown(
            f"<p style='font-size: 13px; opacity: 0.7;'>"
            f"{metrics['weeks_text']} – based on roster data for {role_selected}."
            f"</p>",
            unsafe_allow_html=True,
        )

    render_selected_kpis(selected_metrics)

    # --------------------------------------------------
    # 3. Weekly hours & wages combo chart (optional)
    # --------------------------------------------------
    st.markdown("---")
    render_weekly_summary_chart(df_dashboard)


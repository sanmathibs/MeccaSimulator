import streamlit as st
import pandas as pd
import plotly.express as px

WEEK_DAYS = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------
def kpi_card(title: str, value: str, subtitle: str = "") -> str:
    return f"""
    <div style="
        padding: 18px;
        border-radius: 12px;
        background: #0F766E10;
        border: 1px solid #0F766E40;
        color: #0F172A;
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.08);
        margin-bottom: 12px;
    ">
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; opacity: 0.7;">
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


def highlight_positive(val, color="#90EE90"):
    """Highlight cells with values > 0 (for foundation / flex tables)."""
    try:
        if float(val) > 0:
            return f"background-color: {color}"
    except (ValueError, TypeError):
        pass
    return ""


# ---------------------------------------------------------
# Main render
# ---------------------------------------------------------
def render():
    st.header("Weekly roster coverage & scheduling")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    excel_main_scheduling = pd.ExcelFile("data/main_scheduling.xlsx")
    df_state = excel_main_scheduling.parse("State")
    df_hour_matrix = excel_main_scheduling.parse("HourMatrix")
    df_foundation_schedule = excel_main_scheduling.parse("FoundationSchedule")

    df_hour_matrix = df_hour_matrix[
        ["States"] + WEEK_DAYS
    ]

    df_state = df_state[
        ["WoY"] + WEEK_DAYS
    ]

    df_foundation_schedule = df_foundation_schedule[
        ["Available", "Type", "Category", "Role"] + WEEK_DAYS
    ]
    df_foundation_schedule[WEEK_DAYS] = (
        df_foundation_schedule[WEEK_DAYS].fillna(0).astype(int, errors="ignore")
    )

    df_roster = pd.read_excel("data/roster_clean_data.xlsx")
    df_roster = df_roster[
        [
            "WoY",
            "DoW",
            "Role",
            "Start",
            "End",
            "Foundation or Flex",
            "Date",
            "Unique",
            "Type",
            "Category",
            "Rate",
            "Hours Worked",
            "Dollars",
            "Start Time",
            "End Time",
            "Conc",
            "Weekday",
        ]
    ]
    df_roster["Date"] = pd.to_datetime(df_roster["Date"]).dt.date

    week_num_list = sorted(df_roster["WoY"].unique().tolist())

    # -----------------------------------------------------
    # Week selector
    # -----------------------------------------------------
    col_week, col_info = st.columns([1, 2])

    with col_week:
        selected_week_num = st.selectbox(
            "Select week of year (WoY)", week_num_list, index=0
        )

    with col_info:
        st.caption(
            "View coverage, rostered hours and a visual schedule for the selected week."
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Filter for selected week & derive required / supplied
    # -----------------------------------------------------
    df_roster_selected = df_roster[df_roster["WoY"] == selected_week_num].copy()

    # Get states for each day in the selected week from df_state
    state_row = df_state[df_state["WoY"] == selected_week_num]
    if state_row.empty:
        st.warning("No 'State' configuration found for this week.")
        return

    states = (
        state_row.drop(columns=["WoY"])
        .iloc[0]
        .to_dict()
    )
    states = {day: int(state) for day, state in states.items()}

    # Required hours by day from HourMatrix
    day_hours_required = {}
    for day in WEEK_DAYS:
        state = states.get(day)
        hours = int(
            df_hour_matrix.loc[df_hour_matrix["States"] == state, [day]].values[0]
        )
        day_hours_required[day] = hours

    # Supplied hours by day from roster
    total_hours_supplied = {}
    for day in WEEK_DAYS:
        total_hours = int(
            df_roster_selected[df_roster_selected["DoW"] == day]["Hours Worked"].sum()
        )
        total_hours_supplied[day] = total_hours

    # Foundation vs flexible totals
    foundation_hours = df_roster_selected[
        df_roster_selected["Foundation or Flex"] == "Foundation"
    ]["Hours Worked"].sum()
    flex_hours = df_roster_selected[
        df_roster_selected["Foundation or Flex"] == "Flexible"
    ]["Hours Worked"].sum()

    # Weekly KPIs
    week_required = sum(day_hours_required.values())
    week_supplied = sum(total_hours_supplied.values())
    coverage_pct = (
        week_supplied / week_required * 100 if week_required else 0.0
    )

    daily_gap = {
        day: total_hours_supplied[day] - day_hours_required[day] for day in WEEK_DAYS
    }
    # Largest shortage (most negative gap) and largest surplus
    shortage_day = min(daily_gap, key=daily_gap.get)
    surplus_day = max(daily_gap, key=daily_gap.get)

    flex_share = (
        flex_hours / (foundation_hours + flex_hours) * 100
        if (foundation_hours + flex_hours) > 0
        else 0.0
    )

    # -----------------------------------------------------
    # KPI cards
    # -----------------------------------------------------
    st.subheader("Week summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card(
                "Total required hours",
                f"{week_required:,.0f}",
                f"Week {selected_week_num}",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card(
                "Total rostered hours",
                f"{week_supplied:,.0f}",
                "Foundation + flexible",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card(
                "Coverage",
                f"{coverage_pct:,.1f}%",
                "100% = perfectly matched",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card(
                "Flexible share",
                f"{flex_share:,.1f}%",
                "Flexible as % of rostered hours",
            ),
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # Clean weekly schedule board (all team members)
    # -----------------------------------------------------
    st.subheader("Weekly schedule view (all team members)")

    if df_roster_selected.empty:
        st.info("No roster entries for this week.")
    else:
        df_sched = df_roster_selected.copy()

        # Build proper datetimes from Date + Start/End Time
        df_sched["StartDT"] = pd.to_datetime(
            df_sched["Date"].astype(str) + " " + df_sched["Start Time"].astype(str),
            errors="coerce",
        )
        df_sched["EndDT"] = pd.to_datetime(
            df_sched["Date"].astype(str) + " " + df_sched["End Time"].astype(str),
            errors="coerce",
        )

        # Keep only rows with valid times
        df_sched = df_sched.dropna(subset=["StartDT", "EndDT"])
        if df_sched.empty:
            st.info("No valid shift times for this week.")
        else:
            # Use Role as the display name (e.g. "#1 Host")
            df_sched["Employee"] = df_sched["Role"]

            # Order team members by total hours in the week (most hours at top)
            emp_order = (
                df_sched.groupby("Employee")["Hours Worked"]
                .sum()
                .sort_values(ascending=False)
                .index.tolist()
            )
            df_sched["Employee"] = pd.Categorical(
                df_sched["Employee"], categories=emp_order, ordered=True
            )

            # One timeline for the whole week
            fig_tl = px.timeline(
                df_sched,
                x_start="StartDT",
                x_end="EndDT",
                y="Employee",
                color="Foundation or Flex",
                color_discrete_map={
                    "Foundation": "#22c55e",  # green
                    "Flexible": "#0ea5e9",  # blue
                },
            )

            # Neat daily ticks: Mon, Tue, ... with one tick per day
            week_start = df_sched["StartDT"].dt.normalize().min()
            week_end = week_start + pd.Timedelta(days=7)

            fig_tl.update_xaxes(
                title="",
                tickformat="%a\n%d %b",  # Mon / 06 Oct
                dtick=24 * 60 * 60 * 1000,  # 1 day in ms
                range=[week_start, week_end],
                showgrid=True,
                gridcolor="rgba(148,163,184,0.2)",
            )
            fig_tl.update_yaxes(title="Team member")

            fig_tl.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=40, b=10),
                legend_title="",
                bargap=0.2,
                hovermode="closest",
            )

            # Thinner, clean bars
            fig_tl.update_traces(marker_line_width=0, opacity=0.95)

            st.plotly_chart(fig_tl, use_container_width=True)

    # -----------------------------------------------------
    # Coverage bar chart (required vs supplied by day)
    # -----------------------------------------------------
    st.subheader("Daily coverage – required vs rostered hours")

    coverage_df = pd.DataFrame(
        {
            "Day": WEEK_DAYS,
            "Required Hours": [day_hours_required.get(day, 0) for day in WEEK_DAYS],
            "Supplied Hours": [total_hours_supplied.get(day, 0) for day in WEEK_DAYS],
        }
    )
    coverage_df["Day"] = pd.Categorical(
        coverage_df["Day"], categories=WEEK_DAYS, ordered=True
    )

    coverage_long = coverage_df.melt(
        id_vars="Day", var_name="Type", value_name="Hours"
    )
    fig_cov = px.bar(
        coverage_long,
        x="Day",
        y="Hours",
        color="Type",
        barmode="group",
        category_orders={"Day": WEEK_DAYS},
        color_discrete_map={
            "Required Hours": "#004D6E",
            "Supplied Hours": "#EA7500",
        },
        labels={"Hours": "Hours", "Day": ""},
        title="",
    )
    fig_cov.update_layout(
        legend_title="",
        margin=dict(l=10, r=10, t=10, b=0),
    )
    st.plotly_chart(fig_cov, use_container_width=True)

    # -----------------------------------------------------
    # Per-day KPI table
    # -----------------------------------------------------
    st.subheader("Daily KPI table")

    kpi_data = {
        "Day": WEEK_DAYS,
        "State": [states.get(day, 0) for day in WEEK_DAYS],
        "Required Hours": [day_hours_required.get(day, 0) for day in WEEK_DAYS],
        "Supplied Hours": [total_hours_supplied.get(day, 0) for day in WEEK_DAYS],
        "Gap (Supplied - Required)": [
            total_hours_supplied.get(day, 0) - day_hours_required.get(day, 0)
            for day in WEEK_DAYS
        ],
    }
    kpi_df = pd.DataFrame(kpi_data).set_index("Day")
    st.dataframe(kpi_df, use_container_width=True)

    # -----------------------------------------------------
    # Foundation & flexible schedule tables
    # -----------------------------------------------------
    st.subheader("Foundation schedule (hours per TM per day)")
    styled_df = df_foundation_schedule.style.applymap(
        highlight_positive, subset=WEEK_DAYS
    )
    st.dataframe(styled_df, use_container_width=True)

    st.subheader("Flexible schedule (hours per TM per day)")
    df_flex_schedule = df_foundation_schedule[["Role"] + WEEK_DAYS].copy()
    df_flex_schedule.loc[:, WEEK_DAYS] = 0  # start with zeros

    df_flex_roster = df_roster_selected[
        df_roster_selected["Foundation or Flex"] == "Flexible"
    ]

    for _, row in df_flex_roster.iterrows():
        role = row["Role"]
        dow = row["DoW"]
        hours_worked = row["Hours Worked"]
        if role in df_flex_schedule["Role"].values and dow in WEEK_DAYS:
            df_flex_schedule.loc[df_flex_schedule["Role"] == role, dow] += hours_worked

    styled_flex_df = df_flex_schedule.style.applymap(
        lambda v: highlight_positive(v, color="#FFD000"), subset=WEEK_DAYS
    )
    st.dataframe(styled_flex_df, use_container_width=True)

    # -----------------------------------------------------
    # Narrative insights
    # -----------------------------------------------------
    st.subheader("Week insights")

    shortage_val = daily_gap[shortage_day]
    surplus_val = daily_gap[surplus_day]

    st.markdown(
        f"""
        - **Overall coverage:** {coverage_pct:,.1f}% of required hours are rostered this week  
        - **Largest shortage:** **{shortage_day}** with {shortage_val:+,.0f} hours vs requirement  
        - **Largest surplus:** **{surplus_day}** with {surplus_val:+,.0f} hours vs requirement  
        - **Mix of hours:** {foundation_hours:,.0f} foundation hours and {flex_hours:,.0f} flexible hours  
          (flexible makes up **{flex_share:,.1f}%** of the rostered hours)
        """
    )

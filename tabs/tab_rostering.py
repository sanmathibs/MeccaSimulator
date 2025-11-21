import streamlit as st
import pandas as pd
import plotly.express as px


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
def load_roster(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    # Ensure types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Hours Worked"] = pd.to_numeric(df["Hours Worked"], errors="coerce").fillna(0)
    df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0)

    df["DoW"] = df["DoW"].astype(str)
    return df

def compute_selected_metrics(df_role_filtered: pd.DataFrame) -> dict:
    """
    Metrics for the currently selected week range (slider).
    """
    if df_role_filtered.empty:
        return {
            "weeks_in_view": 0,
            "weeks_label": "No data",
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

def render_selected_kpis(sel_metrics: dict):
    st.markdown("#### Selected weeks summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            info_card("Weeks in view", sel_metrics["weeks_label"]),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            info_card(
                "Hours (selected weeks)",
                f"{sel_metrics['total_hours_sel']:.1f}",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            info_card(
                "Wages (selected weeks)",
                f"${sel_metrics['total_wages_sel']:,.2f}",
            ),
            unsafe_allow_html=True,
        )



# ---------------------------------------------------------
# Filters
# ---------------------------------------------------------
def render_filter_section(df: pd.DataFrame):
    st.markdown("### Filters")

    all_roles = sorted(df["Role"].unique())
    all_weeks = sorted(df["WoY"].unique())
    min_week, max_week = min(all_weeks), max(all_weeks)
    min_date = df["Date"].min()
    max_date = df["Date"].max()

    col_role, col_weeks, col_period = st.columns([1.2, 1.4, 1.6])

    with col_role:
        role_selected = st.selectbox("Choose team member", all_roles)

    with col_weeks:
        week_range = st.slider(
            "Weeks of year",
            min_value=int(min_week),
            max_value=int(max_week),
            value=(int(min_week), int(max_week)),
            step=1,
        )

    with col_period:
        st.caption(
            f"Roster period: **week {min_week}–{max_week}** "
            f"({len(all_weeks)} weeks), "
            f"{min_date:%d %b %Y} – {max_date:%d %b %Y}"
        )

    return role_selected, week_range


# ---------------------------------------------------------
# Build weekly roster table for a single role
# ---------------------------------------------------------
def generate_role_dashboard(df_role: pd.DataFrame) -> pd.DataFrame:
    """
    df_role: already filtered to a single Role (and desired week range).
    Returns one row per week with Sunday–Saturday columns + Hours + Wages.
    """
    if df_role.empty:
        return pd.DataFrame()

    df_role = df_role.copy()
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    weeks = sorted(df_role["WoY"].unique())
    dashboard_rows = []

    for w in weeks:
        week_data = df_role[df_role["WoY"] == w]
        row = {
            "Week of year": w,
            "Date": week_data["Date"].min(),  # week start date
        }
        hours_sum = 0.0
        wages_sum = 0.0

        for day in days:
            day_data = week_data[week_data["DoW"] == day]
            if not day_data.empty:
                # handle potential multiple shifts per day
                shifts = day_data.apply(
                    lambda r: f"{r['Start Time']} to {r['End Time']}", axis=1
                )
                row[day] = " / ".join(shifts.tolist())
                hours_sum += day_data["Hours Worked"].sum()
                wages_sum += day_data["Dollars"].sum()
            else:
                row[day] = "Off"

        row["Hours"] = hours_sum
        row["Wages"] = wages_sum
        dashboard_rows.append(row)

    df_dashboard = pd.DataFrame(dashboard_rows).sort_values("Week of year")
    return df_dashboard.reset_index(drop=True)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def compute_role_metrics(df_role_full: pd.DataFrame) -> dict:
    """
    df_role_full: all roster data for the selected role (full 12-week period).
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
# Info cards
# ---------------------------------------------------------
def info_card(title: str, value: str) -> str:
    return f"""
    <div style="
        padding: 18px;
        border-radius: 12px;
        background: #0F766E10;
        border: 1px solid #0F766E40;
        color: #0F172A;
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.08);
        margin-bottom: 12px;
        text-align: left;
    ">
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; opacity: 0.7;">
            {title}
        </div>
        <div style="font-size: 22px; margin-top: 4px; font-weight: 700;">
            {value}
        </div>
    </div>
    """


def render_info_cards(metrics: dict):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            info_card("Type", metrics["type_role"]), unsafe_allow_html=True
        )
        st.markdown(
            info_card("Category", metrics["category_role"]), unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            info_card(
                "Hours per week", f"{metrics['hours_per_week']:.2f}"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            info_card(
                "Total wages (roster period)",
                f"${metrics['total_wages']:,.2f}",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            info_card(
                "12-week wages",
                f"${metrics['wages_12_weeks']:,.2f}",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            info_card(
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
    # Pretty date format
    df_display["Date"] = df_display["Date"].dt.strftime("%A, %d %B %Y")

    # Numeric formatting
    df_display["Hours"] = df_display["Hours"].round(1)
    df_display["Wages"] = df_display["Wages"].round(2)

    styler = df_display.style.format(
        {
            "Hours": "{:.1f}",
            "Wages": "${:,.2f}",
        }
    )

    st.dataframe(styler, use_container_width=True)


def render_weekly_summary_chart(df_dashboard: pd.DataFrame):
    if df_dashboard.empty:
        return

    df_chart = df_dashboard[["Week of year", "Hours", "Wages"]].copy()
    df_chart["Week of year"] = df_chart["Week of year"].astype(int)

    fig = px.bar(
        df_chart,
        x="Week of year",
        y=["Hours", "Wages"],
        barmode="group",
        labels={"value": "", "variable": "", "Week of year": "Week of year"},
        title="Weekly hours and wages",
    )
    fig.update_layout(
        legend_title="",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def render():
    df = load_roster("data/roster_clean_data.xlsx")

    st.header("Roster explorer")

    role_selected, week_range = render_filter_section(df)

    # Full data for this role (for metrics / 12-week calc)
    df_role_full = df[df["Role"] == role_selected].copy()

    # Filtered data for table/chart based on slider range
    df_role_filtered = df_role_full[
        df_role_full["WoY"].between(week_range[0], week_range[1])
    ].copy()

    df_dashboard = generate_role_dashboard(df_role_filtered)
    metrics = compute_role_metrics(df_role_full)

    selected_metrics = compute_selected_metrics(df_role_filtered)

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

    st.markdown("---")
    render_roster_table(df_dashboard)

    st.markdown("---")
    render_weekly_summary_chart(df_dashboard)

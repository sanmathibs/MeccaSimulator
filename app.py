"""
Main Streamlit Application with Tab-Based Architecture

This is the entry point for the application. Each tab is implemented
as a separate module in the tabs/ directory for independent development.
"""

import streamlit as st
from tabs import (
    tab_ai_insight,
    tab_overview,
    tab_roster_weekly,
    tab_store,
    tab_rostering,
    tab_forecast,
    tab_event,
    tab_forecast_lab,
    tab_reforecast,
)


st.set_page_config(
    page_title="Mecca Simulator - with live forecasting",
    page_icon="🚀",
    layout="wide",
)


def inject_global_css():
    """One-time global styling for header + tabs."""
    print("Injecting global CSS styles...")

    st.markdown(
        """
        <style>
        /* Tighten default page padding */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }

        /* ========== HERO HEADER ========== */
        .mecca-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.1rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #0f172a, #1e293b);
            color: #f9fafb;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.45);
            margin-bottom: 1.5rem;
        }
        .mecca-hero-left {
            display: flex;
            align-items: center;
            gap: 1.1rem;
        }
        .mecca-hero-emoji {
            font-size: 2.7rem;
        }
        .mecca-hero-title {
            margin: 0;
            font-size: 2.0rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .mecca-hero-subtitle {
            margin: 0.2rem 0 0;
            font-size: 0.95rem;
            opacity: 0.85;
        }

        /* Custom button styles */
        div[data-testid="stButton"] > button {
            border-radius:999px;
            padding:0.35rem 0.95rem;
            border:1px solid #e5e7eb;
            background:linear-gradient(135deg,#f9fafb,#eef2ff);
            color:#111827;
            font-size:0.85rem;
            white-space: nowrap !important;
        }
        
        div[data-testid="stButton"] > button[kind="primary"] {
            background:linear-gradient(135deg,#6366f1,#3b82f6);
            color:#f9fafb;
            border-color:#6366f1;
            box-shadow:0 8px 20px rgba(99,102,241,0.45);
        }
        
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            border-color:#4f46e5;
            background:linear-gradient(135deg,#818cf8,#60a5fa);
            box-shadow:0 8px 20px rgba(79,70,229,0.45);
        }
        
        div[data-testid="stButton"] > button:hover {
            border-color:#6366f1;
            background:linear-gradient(135deg,#eef2ff,#e0f2fe);
            box-shadow:0 0 0 1px rgba(99,102,241,0.25);
        }
        
        div[data-testid="stButton"] > button:disabled {
            background: #0f172a !important;
            color: #f9fafb !important;
            border-color: #6366f1 !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.45) !important;
            opacity: 1 !important;
            cursor: default !important;
        }
        
        /* Selected tab style */
        .stTabs button {
            padding-left: 0.95rem;
            padding-right: 0.95rem;
        }

        .stTabs button[aria-selected="true"] {
            background-color: #FFF0E2 !important;
        }

        /* Filter chips (st.multiselect tags) – light purple theme */
        .stMultiSelect [data-baseweb="tag"] {
            background-color:#e2d5f0!important;
            border-radius:999px !important;
            border:1px solid #E9D5FF !important;
            color:#7b4dc4 !important;
            font-size:16px !important;
        }
        
        .stMultiSelect [data-baseweb="tag"] span {
            color:#2e0b63 !important;
        }
        
        .stMultiSelect [data-baseweb="tag"] svg {
            fill:#743acf !important;
        }
        
        /* Metrics */
        .stMetric > div {
            padding: 1.8rem 1.0rem !important;
            border-radius: 12px !important;
            border: 1px solid #e5e7eb !important;
            background: linear-gradient(135deg,#fdf2ff,#eef2ff) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15) !important; 
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Main application function."""
    inject_global_css()

    # ---------- HERO HEADER ----------
    st.markdown(
        """
        <div class="mecca-hero">
          <div class="mecca-hero-left">
            <div>
              <h1 class="mecca-hero-title">Mecca Simulator</h1>
              <p class="mecca-hero-subtitle">
                36-month demand forecasting & labour scheduling lab for MECCA stores.
              </p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for active tab if not exists
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0

    # Tab labels
    tab_labels = [
        "📊 Overview",
        "🏪 Store Explorer",
        "🔬 Forecast Lab",
        "🧪 Dynamic Forecast",
        "📈 Forecast Dashboard",
        "📋 Event & Seasonality",
        "👤 Rosters - Individual TM",
        "📅 Rosters - Weekly Coverage & Scheduling",
        "🤖 AI Insight Lab",
    ]

    # Create custom tab buttons with flexbox
    tab_container = st.container(horizontal=True, horizontal_alignment="distribute")
    with tab_container:
        for idx, label in enumerate(tab_labels):
            # Use disabled flag for selected tab
            is_selected = idx == st.session_state.active_tab
            if st.button(label, key=f"tab_{idx}", disabled=is_selected):
                st.session_state.active_tab = idx
                st.rerun()

    st.markdown("---")

    # Render content based on selected tab
    if st.session_state.active_tab == 0:
        tab_overview.render()
    elif st.session_state.active_tab == 1:
        tab_store.render()
    elif st.session_state.active_tab == 2:
        tab_forecast_lab.render()
    elif st.session_state.active_tab == 3:
        tab_reforecast.render()
    elif st.session_state.active_tab == 4:
        tab_forecast.render()
    elif st.session_state.active_tab == 5:
        tab_event.render()
    elif st.session_state.active_tab == 6:
        tab_rostering.render()
    elif st.session_state.active_tab == 7:
        tab_roster_weekly.render()
    elif st.session_state.active_tab == 8:
        tab_ai_insight.render()


if __name__ == "__main__":
    main()
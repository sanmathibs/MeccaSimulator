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
)


st.set_page_config(
    page_title="Mecca Simulator - with live forecasting",
    page_icon="🚀",
    layout="wide",
)

def inject_global_css():
    """One-time global styling for header + tabs."""
    if st.session_state.get("css_injected"):
        return
    st.session_state["css_injected"] = True

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

        /* ========== TABS AS PILLS ========== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: none;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.35rem 1.1rem;
            border-radius: 999px;
            background-color: #f3f4f6;
            color: #4b5563;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(148, 163, 184, 0.4);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #0f172a;
            color: #f9fafb;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.45);
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 0.9rem;
            margin-bottom: 0;
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

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "📊 Overview",
            "🏪 Store Explorer",
            "🔬 Forecast Lab",
            "📈 Forecast Dashboard",
            "📋 Event & Seasonality",
            "👤 Rosters - Individual TM",
            "📅 Rosters - Weekly Coverage & Scheduling",
            "🤖 AI Insight Lab",
        ]
    )

    with tab1:
        tab_overview.render()

    with tab2:
        tab_store.render()

    with tab3:
        tab_forecast_lab.render()

    with tab4:
        tab_forecast.render()

    with tab5:
        tab_event.render()

    with tab6:
        tab_rostering.render()

    with tab7:
        tab_roster_weekly.render()

    with tab8:
        tab_ai_insight.render()


if __name__ == "__main__":
    main()

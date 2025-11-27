"""
Tabs package initialization.

This package contains all individual tab modules.
Each tab is self-contained and can be developed independently.
"""

# Import all tab modules
from . import (
    tab_overview,
    tab_roster_weekly,
    tab_rostering,
    tab_store,
    tab_forecast,
    tab_event,
    tab_ai_insight,
    tab_reforecast,
)


__all__ = [
    "tab_overview",
    "tab_store",
    "tab_rostering",
    "tab_event",
    "tab_forecast",
    "tab_roster_weekly",
    "tab_ai_insight",
    "tab_reforecast",
]
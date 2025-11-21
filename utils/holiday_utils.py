"""Utility functions for holiday calendar generation and event classification.

Centralizes holiday date logic so both Prophet holiday DataFrame creation and
shutdown/event tagging share the same source of truth.
"""

from datetime import date, timedelta
from typing import List, Dict
from dateutil import easter
import pandas as pd


def get_holiday_definitions(year: int) -> List[Dict]:
    """Return list of holiday definition dicts for a given year.

    Each dict has: name, date (date object), lower_window, upper_window
    Windows reflect typical trading impact periods; exact date matching for
    event tagging uses only the date field.
    """
    holidays: List[Dict] = []

    # Father's Day (first Sunday of September)
    sept1 = date(year, 9, 1)
    fathers_day = sept1 + timedelta(days=(6 - sept1.weekday()) % 7)
    holidays.append(
        {
            "name": "Fathers_Day",
            "date": fathers_day,
            "lower_window": -1,
            "upper_window": 1,
        }
    )

    # Black Friday (last Friday of November)
    nov_fridays = [
        date(year, 11, d) for d in range(1, 31) if date(year, 11, d).weekday() == 4
    ]
    if nov_fridays:
        black_friday = nov_fridays[-1]
        holidays.append(
            {
                "name": "Black_Friday",
                "date": black_friday,
                "lower_window": -1,
                "upper_window": 3,
            }
        )

    # Christmas (Dec 25)
    christmas = date(year, 12, 25)
    holidays.append(
        {
            "name": "Christmas",
            "date": christmas,
            "lower_window": -14,
            "upper_window": 4,
        }
    )

    # New Year (Jan 1)
    new_year = date(year, 1, 1)
    holidays.append(
        {
            "name": "New_Year",
            "date": new_year,
            "lower_window": 0,
            "upper_window": 0,
        }
    )

    # Easter Sunday & Good Friday
    easter_sunday = easter.easter(year)
    good_friday = easter_sunday - timedelta(days=2)
    holidays.append(
        {
            "name": "Good_Friday",
            "date": good_friday,
            "lower_window": 0,
            "upper_window": 0,
        }
    )
    holidays.append(
        {
            "name": "Easter",
            "date": easter_sunday,
            "lower_window": -2,
            "upper_window": 1,
        }
    )

    # ANZAC Day (from 2025 onward)
    if year >= 2025:
        anzac_day = date(year, 4, 25)
        holidays.append(
            {
                "name": "ANZAC_Day",
                "date": anzac_day,
                "lower_window": 0,
                "upper_window": 0,
            }
        )

    return holidays


def build_holidays_dataframe(start_year: int, end_year: int) -> pd.DataFrame:
    """Build a Prophet-compatible holidays DataFrame for a year range inclusive."""
    rows = []
    for y in range(start_year, end_year + 1):
        for h in get_holiday_definitions(y):
            rows.append(
                {
                    "holiday": h["name"],
                    "ds": pd.Timestamp(h["date"]),
                    "lower_window": h["lower_window"],
                    "upper_window": h["upper_window"],
                }
            )
    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


def get_holiday(dt):
    """Return holiday name for the exact date of dt (no window logic)."""
    y = dt.year
    d = dt.date()
    for h in get_holiday_definitions(y):
        if h["date"] == d:
            return h["name"]
    return None

"""
Validation helpers.
Given a condition line and a generated date string, check whether all
four conditions are satisfied.
"""

from __future__ import annotations
import re
import datetime
from typing import Optional


MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3,  "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7,  "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
DAY_MAP = {
    "MON": 0, "TUE": 1, "WED": 2, "THU": 3,
    "FRI": 4, "SAT": 5, "SUN": 6,
}


def is_leap_year(year: int) -> bool:
    """Standard Gregorian leap-year rule."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def parse_date_safe(date_str: str) -> Optional[datetime.date]:
    """Return a datetime.date or None if the string is not a valid date."""
    try:
        parts = date_str.strip().split('-')
        dd, mm, yyyy = int(parts[0]), int(parts[1]), int(parts[2])
        return datetime.date(yyyy, mm, dd)
    except Exception:
        return None


def validate(condition_line: str, date_str: str) -> bool:
    """
    Return True iff date_str satisfies all four conditions in condition_line.
    Returns False if the date string is not a valid calendar date.
    """
    tokens = re.findall(r'\[([^\]]+)\]', condition_line)
    if len(tokens) < 4:
        return False
    day_tok, month_tok, leap_tok, decade_tok = tokens[:4]

    date = parse_date_safe(date_str)
    if date is None:
        return False

    # --- day-of-week ---
    expected_dow = DAY_MAP[day_tok]       # 0=Mon … 6=Sun
    if date.weekday() != expected_dow:
        return False

    # --- month ---
    if date.month != MONTH_MAP[month_tok]:
        return False

    # --- leap year ---
    expected_leap = leap_tok == "True"
    if is_leap_year(date.year) != expected_leap:
        return False

    # --- decade ---
    decade_start = int(decade_tok) * 10
    decade_end   = decade_start + 9
    if not (decade_start <= date.year <= decade_end):
        return False

    return True


def condition_accuracy(condition_lines, date_strings) -> float:
    """
    Compute the fraction of (condition, date) pairs that fully satisfy
    all four conditions. Useful as a training/eval metric.
    """
    correct = sum(
        validate(c, d)
        for c, d in zip(condition_lines, date_strings)
    )
    return correct / max(len(condition_lines), 1)

from datetime import date, datetime, timedelta
from calendar import monthrange


def parse_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def format_date(value):
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else None


def days_between(start, end):
    start_date = parse_date(start)
    end_date = parse_date(end)
    if start_date is None or end_date is None:
        return 10**9
    return (end_date - start_date).days


def prediction_period(selected_date):
    selected = parse_date(selected_date)
    return selected.strftime("%Y-%m")


def month_period(value):
    selected = parse_date(value)
    return selected.strftime("%Y-%m")


def add_months(period, months):
    year, month = [int(part) for part in str(period)[:7].split("-")]
    month_index = year * 12 + (month - 1) + months
    new_year = month_index // 12
    new_month = month_index % 12 + 1
    return f"{new_year:04d}-{new_month:02d}"


def previous_month_period(selected_date):
    return add_months(month_period(selected_date), -1)


def month_end_date(period):
    year, month = [int(part) for part in str(period)[:7].split("-")]
    return date(year, month, monthrange(year, month)[1])


def month_start_date(period):
    year, month = [int(part) for part in str(period)[:7].split("-")]
    return date(year, month, 1)


def evaluation_window(selected_date, window_days):
    selected = parse_date(selected_date)
    end_date = selected - timedelta(days=1)
    start_date = selected - timedelta(days=window_days)
    return start_date, end_date


def window_label(start_date, end_date):
    return f"{format_date(start_date)}_to_{format_date(end_date)}"

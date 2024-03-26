# datetime helpers
import datetime as dt

import pytz

_ams_tz = pytz.timezone('Europe/Amsterdam')
_utc_tz = pytz.utc


# Reminder:
# a = dt.datetime(2023, 4, 12, 0, 0, 0)
# print("Unaware", a)  -> Unaware 2023-04-12 00:00:00 # this is NOT
# print("UTC", to_utc_tz(a))  -> UTC 2023-04-12 00:00:00+00:00 # this is NOT
# print("AMS", to_ams_tz(a))  -> AMS 2023-04-12 00:00:00+02:00 # this is also okay
# print("AMS in UTC", to_ams_in_utc(a)) -> AMS in UTC 2023-04-11 22:00:00+00:00 # this is okay


def to_ams_local_tz(datetime: dt.datetime) -> dt.datetime:
    if datetime.tzinfo is not None:
        return datetime.astimezone(_ams_tz)
    return _ams_tz.localize(datetime)


def to_ams_in_utc(datetime: dt.datetime) -> dt.datetime:
    return _to_utc_tz(to_ams_local_tz(datetime))


def _to_utc_tz(datetime: dt.datetime) -> dt.datetime:
    if datetime.tzinfo is not None:
        return datetime.astimezone(_utc_tz)
    return _utc_tz.localize(datetime)


def get_month_range(year: int, month: int):
    start_date = dt.datetime(year, month, 1)

    next_month = start_date.replace(month=month + 1)
    end_date = next_month - dt.timedelta(days=1)

    return start_date, end_date


def get_week_range(year: int, week: int):
    first_day = dt.datetime(year, 1, 1)

    # if first_day of the year is already a Monday (==0) we good, else, math your way to the first monday of the year
    first_monday = first_day if first_day.weekday() == 0 \
        else first_day + dt.timedelta(days=(7 - first_day.weekday()))

    start_date = first_monday + dt.timedelta(weeks=week - 1)

    # Calculate the end date of the given week
    end_date = start_date + dt.timedelta(days=6)

    return start_date, end_date


def get_week_number_of_date(date: dt.datetime):
    date = date.replace(tzinfo=None)
    # Find the first day of the year
    first_day = dt.datetime(date.year, 1, 1)

    # if first_day of the year is already a Monday (==0) we good, else, math your way to the first monday of the year
    first_monday = first_day if first_day.weekday() == 0 \
        else first_day + dt.timedelta(days=(7 - first_day.weekday()))
    # Calculate the week number
    week_number = (date - first_monday).days // 7 + 1

    return week_number


def get_season(date: dt.datetime):
    month = date.month * 100
    day = date.day
    month_day = month + day  # combining month and day

    if (month_day >= 301) and (month_day <= 531):
        season = "Spring"
    elif (month_day > 531) and (month_day < 901):
        season = "Summer"
    elif (month_day >= 901) and (month_day <= 1130):
        season = "Autumn"
    elif (month_day > 1130) and (month_day <= 229):
        season = "Winter"
    else:
        raise IndexError("Invalid Input")

    return season

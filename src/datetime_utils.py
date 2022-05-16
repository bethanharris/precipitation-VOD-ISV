"""
Functions to aid conversions/calculations involving dates.

Decimal year expresses time continuously, e.g. 1 Jan 2000 00:00 = 2000.0, 3rd Jan 15:00 = 2000.0071721

Bethan Harris, UKCEH, 04/01/2021
"""


from datetime import datetime, timedelta
import numpy as np
from cftime import num2date, date2num


def generate_days_since_1970(min_year, max_year):
    """
    Get array of values for days since 1 Jan 1970 for each consecutive day in a given range of years.
    Parameters:
    min_year (int): Year at start of date range (dates will start on 1 Jan of min_year)
    max_year (int): Year at end of date range (final date will be 31 Dec of max_year)
    Returns:
    (numpy array, 1D): array of dates for each day in range, expressed as days since 1 Jan 1970
    """
    start_date = date2num(datetime(min_year, 1, 1), calendar='gregorian', units='days since 1970-01-01')
    end_date = date2num(datetime(max_year, 12, 31), calendar='gregorian', units='days since 1970-01-01')
    days_since_1970 = np.arange(start_date, end_date+1)
    return days_since_1970


def round_datetime_to_seconds(dt):
    """
    Round datetime object to nearest second.
    Parameters:
    dt (datetime obj): datetime to round
    Returns:
    (datetime obj) datetime rounded to nearest second
    """
    dt += timedelta(seconds=0.5)
    dt -= timedelta(seconds=dt.second, microseconds=dt.microsecond)
    return dt


def decimal_year_to_datetime(decimal_date):
    """
    Convert date expressed as a decimal year to a python datetime object, rounded to nearest second.
    Parameters:
    decimal_date (float): Date/time in decimal form (e.g. 1 Jan 2000 00:00 = 2000.0)
    Returns:
    (datetime obj): Input date as datetime object.
    """
    year = int(decimal_date)
    year_start = datetime(year, 1, 1)
    year_end = year_start.replace(year=year+1)
    seconds_in_year = float((year_end - year_start).total_seconds())
    second_of_year = (decimal_date - year) * seconds_in_year
    new_date = round_datetime_to_seconds(datetime(year, 1 , 1) + timedelta(seconds=second_of_year))
    # rounding to nearest second prevents precision errors (assuming only daily data being analysed anyway)
    return new_date


def datetime_to_decimal_year(dt):
    """
    Convert python datetime object to decimal year.
    Parameters:
    dt (datetime obj): datetime to convert
    Returns: date/time in decimal form (e.g. 1 Jan 2000 00:00 = 2000.0)
    
    """
    year_start = datetime(dt.year, 1, 1)
    year_end = year_start.replace(year=dt.year+1)
    seconds_in_year = float((year_end - year_start).total_seconds())
    seconds_to_date = float((dt-year_start).total_seconds())
    return dt.year + seconds_to_date/seconds_in_year


def days_since_1970_to_decimal_year(days):
    """
    Convert date from units of days since 1 Jan 1970 to decimal year.
    Parameters:
    days (int/float): days since 1 Jan 1970 (non-integer values can be used to express hours, minutes etc.)
    Returns:
    (float) date as decimal year
    """
    date = num2date(days, units='days since 1970-01-01', calendar='gregorian') 
    return datetime_to_decimal_year(date)
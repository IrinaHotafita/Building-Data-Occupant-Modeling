import pandas as pd

## Period

def filter_by_period(df, start, end):
    """
    Filters the DataFrame by a given time period.

    Parameters:
    - df: pandas DataFrame with a 'timestamp' column.
    - start: start of the period (datetime or string compatible with pd.to_datetime)
    - end: end of the period (datetime or string compatible with pd.to_datetime)

    Returns:
    - Filtered DataFrame with rows between start and end (inclusive).
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    return df.loc[mask]


## Weekdays

def filter_by_weekdays(df, weekdays):
    """
    Filters the DataFrame by specified weekdays.

    Parameters:
    - df: pandas DataFrame with a 'timestamp' column
    - weekdays: list of integers representing weekdays
        (0=Monday, 1=Tuesday, ..., 6=Sunday)

    Returns:
    - Filtered DataFrame with rows matching the given weekdays
    """
    # Ensure weekdays is a list
    if isinstance(weekdays, int):
        weekdays = [weekdays]
    
    mask = df['timestamp'].dt.dayofweek.isin(weekdays)
    return df.loc[mask]

## time of the day

def filter_by_time_of_day(df, start_hour, end_hour):
    """
    Filters the DataFrame to include only rows where the timestamp is between
    start_hour and end_hour (inclusive) for every day.

    Parameters:
    - df: pandas DataFrame with a 'timestamp' column
    - start_hour: int or float, start hour (e.g., 8 or 8.5 for 8:30 AM)
    - end_hour: int or float, end hour (e.g., 18 for 6 PM)

    Returns:
    - Filtered DataFrame
    """
    # Extract hour and minute as fractional hour
    hour_fraction = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    
    mask = (hour_fraction >= start_hour) & (hour_fraction <= end_hour)
    return df.loc[mask]

## All the three above in one function

# import pandas as pd

def filter_dataframe(df, start_date=None, end_date=None, weekdays=None, start_hour=None, end_hour=None):
    """
    Filters a DataFrame by date range, weekdays, and time of day.

    Parameters:
    - df: pandas DataFrame with a 'timestamp' column
    - start_date: string or datetime, start of period (inclusive) (datetime or string compatible with pd.to_datetime ; e.g. pd.Timestamp('2024-10-15', tz='Europe/Paris')
    - end_date: string or datetime, end of period (inclusive) (datetime or string compatible with pd.to_datetime ; e.g. pd.Timestamp('2024-10-20', tz='Europe/Paris')
    - weekdays: list of integers representing weekdays (0=Monday,...,6=Sunday)
    - start_hour: float/int, start hour of day (e.g., 8 or 8.5 for 8:30 AM)
    - end_hour: float/int, end hour of day (e.g., 18)

    Returns:
    - Filtered DataFrame
    """
    filtered = df.copy()
    
    # Filter by date range
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        filtered = filtered[filtered['timestamp'] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        filtered = filtered[filtered['timestamp'] <= end_date]
    
    # Filter by weekdays
    if weekdays is not None:
        if isinstance(weekdays, int):
            weekdays = [weekdays]
        filtered = filtered[filtered['timestamp'].dt.dayofweek.isin(weekdays)]
    
    # Filter by time of day
    if start_hour is not None and end_hour is not None:
        hour_fraction = filtered['timestamp'].dt.hour + filtered['timestamp'].dt.minute / 60
        filtered = filtered[(hour_fraction >= start_hour) & (hour_fraction <= end_hour)]

    # Hour of the day
    filtered['hour'] = filtered['timestamp'].dt.hour

    # Day of the week (integer)
    filtered['day_of_week'] = filtered['timestamp'].dt.dayofweek

    
    return filtered

# Filter for example the days in which the occupancy is > 1

def filter_days_by_sum(df, column, threshold):
    """
    Filters the DataFrame to keep only rows from days where the sum of a column exceeds a threshold.

    Parameters:
    - df: pandas DataFrame with a 'timestamp' column
    - column: str, the column to sum (e.g., 'A')
    - threshold: numeric, the threshold to compare the sum against

    Returns:
    - Filtered DataFrame
    """
    # Ensure 'date' column exists
    df = df.copy()
    df['date'] = df['timestamp'].dt.date

    # Sum by date
    daily_sum = df.groupby('date')[column].sum()

    # Get dates where sum > threshold
    valid_dates = daily_sum[daily_sum > threshold].index

    # Filter original DataFrame
    return df[df['date'].isin(valid_dates)]


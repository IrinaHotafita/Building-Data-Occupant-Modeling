import pandas as pd
import numpy as np
import pytz  # Python timezone library
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
import os

# Show current working directory
print(os.getcwd())

# Inputs
    # data with timestamp as index and with only the columns where we calculate CV(RMSE)

def frequency_analysis(data,sampleRates,method='linear',color='steelblue'):
    """
    Compute CV(RMSE), RMSE for all columns of the data and return all intermediate
    results + produce a boxplot.

    How to use it:
    import sys
    sys.path.append("../../00_Swiss_knife")
    from measure_frequency_analysis import frequency_analysis
    
    errors, daily_var, error_rel, error_rel_long, fig = frequency_analysis(data, sampleRates)
    
    if you need only fig: _, _, _, _, fig = frequency_analysis(data, sampleRates)

    # to save the figure
    fig.savefig("mfa_test.png", dpi=300, bbox_inches='tight')
    
    
    Parameters
    ----------
    - data : pandas.DataFrame
        The input dataframe with only timestamp and sensor data columns (no Unnamed:0 or such thing)

    - sampleRates :
        Ex : sampleRates = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    - method : str
        interpolation method
            method = 'linear' (default)

    - color : str
        boxplot colors
            color = 'lightblue' (default)
    returns
     -------
    errors : pandas.DataFrame
        Absolute RMSE values for each sensor and sampling rate.
    daily_variation_mean : dict
        Mean daily variation for each sensor.
    error_relative : pandas.DataFrame
        CV(RMSE) = RMSE / daily_variation_mean.
    error_relative_long : pandas.DataFrame
        Long-format version of CV(RMSE) for plotting.
    fig : matplotlib.figure.Figure
        The generated boxplot figure.
    
    """

    data = data.copy()
    data = data.set_index('timestamp')
    
    # CV(RMSE) estimation

    errors = pd.DataFrame()
    
    for column in data.columns:
    
        data_groundtruth = data[[column]].dropna()
        
        # creare measurements using different sampling rate
        data_compare = data_groundtruth
        sampleRates = sampleRates
    
        for sampleRate in sampleRates:
            measured = data_groundtruth.resample(f'{sampleRate}min').first()
            measured.columns = [f'{sampleRate}min']
            data_compare = pd.concat([data_compare, measured], axis=1)
            
        # fill in using linear interpolation
        for sampleRate in sampleRates:
            data_compare[f'{sampleRate}min'] = data_compare[f'{sampleRate}min'].interpolate(method=method)
            error = (((data_compare[f'{sampleRate}min'] - data_compare[column])**2).mean())**0.5
            errors.loc[column, sampleRate] = error
    # errors

    daily_variation_mean = {}
    
    data['date'] = data.index.date
    data['time'] = data.index.time
    
    for sensor_index in data.columns[:-2]:  # exclude the column of date and time
        
        # Reshape the data, dataframe for each sensor, row for date, column for minute of the day
        data_sensor = data[[sensor_index,'date','time']]
        data_daily = data_sensor.pivot_table(index = 'date', 
                                             columns = 'time',
                                             values = sensor_index)
        data_daily.dropna(inplace=True)
        variation_mean = (data_daily.max(axis=1) - data_daily.min(axis=1)).mean()
        
        daily_variation_mean[sensor_index] = variation_mean
    
    # daily_variation_mean

    error_relative = errors.copy(deep=True)

    for thermal_zone in error_relative.index:
        error_relative.loc[thermal_zone] = error_relative.loc[thermal_zone]/daily_variation_mean[thermal_zone]

    error_relative.head()

    import seaborn as sns
    import matplotlib.ticker as mtick

    # reformat the data to use sns plot functions
    error_relative_long = pd.melt(error_relative, value_vars=error_relative.columns)
    error_relative_long['value'] = error_relative_long['value']*100  # *100 to convert to percentage
    
    # error_relative_long.head()

    fig = plt.figure(figsize=(12, 5))
    ax = sns.boxplot(x="variable", y="value", data=error_relative_long, color=color )  
    
    ax.set_xlabel('Fréquence d\'aquisition [min]')
    ax.set_ylabel('Erreur relative')
    
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(xticks)
    ax.set_ylim(0, 20)
    ax.grid(True)

    
    return errors, daily_variation_mean, error_relative, error_relative_long, fig
## Function outputs
    # errors CV(RMSE)
    # daily_variation_mean (I should pay attention to this when working with other than Temperature)
    # error_relative
    # error_relative_long
    # figure

    #------------------------------------------------------------------

def frequency_analysis_RMSE(data, sampleRates, method='linear', color='steelblue'):
    """
    Compute RMSE for all columns of the data and return results + boxplot.

    Parameters
    ----------
    - data : pandas.DataFrame
        The input dataframe with timestamp as index (or a column named 'timestamp').

    - sampleRates : list
        e.g. [10,20,30,40,50,60,70,80,90,100,110,120]

    - method : str
        interpolation method (default 'linear')

    - color : str
        boxplot color (default 'steelblue')

    Returns
    -------
    errors : pandas.DataFrame
        RMSE values for each sensor and sampling rate.

    error_long : pandas.DataFrame
        Long-format RMSE for plotting.

    fig : matplotlib.figure.Figure
        The generated boxplot figure.
    """

    data = data.copy()

    # ensure timestamp is index
    if 'timestamp' in data.columns:
        data = data.set_index('timestamp')

    errors = pd.DataFrame()

    # ---- RMSE calculation ----
    for column in data.columns:
        data_groundtruth = data[[column]].dropna()
        data_compare = data_groundtruth.copy()

        for sampleRate in sampleRates:
            measured = data_groundtruth.resample(f'{sampleRate}min').first()
            measured.columns = [f'{sampleRate}min']
            data_compare = pd.concat([data_compare, measured], axis=1)

        # interpolate and compute RMSE
        for sampleRate in sampleRates:
            interp = data_compare[f'{sampleRate}min'].interpolate(method=method)
            error = np.sqrt(((interp - data_compare[column]) ** 2).mean())
            errors.loc[column, sampleRate] = error

    # ---- Prepare long-format table ----
    error_long = pd.melt(errors.reset_index(),
                         id_vars='index',
                         value_vars=errors.columns)
    error_long.columns = ['sensor', 'sampleRate', 'RMSE']

    # ---- Plot ----
    fig = plt.figure(figsize=(12, 5))
    ax = sns.boxplot(x="sampleRate", y="RMSE", data=error_long, color=color)

    ax.set_xlabel('Fréquence d\'aquisition [min]')
    ax.set_ylabel('RMSE')
    ax.grid(True)

    return errors, error_long, fig

#--------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def frequency_analysis_RMSE_curve(data, sampleRates, method='linear', colors=None):
    """
    Compute RMSE for each sensor at different sampling rates and plot a curve per sensor.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe with timestamp as index or a column named 'timestamp'.
    sampleRates : list
        List of sampling rates in minutes, e.g. [10,20,30,...].
    method : str
        Interpolation method (default 'linear').
    colors : list
        Optional list of colors for each sensor.

    Returns
    -------
    errors : pandas.DataFrame
        RMSE for each sensor at each sampling rate.
    fig : matplotlib.figure.Figure
        Line plot figure with RMSE curves.
    """

    data = data.copy()
    if 'timestamp' in data.columns:
        data = data.set_index('timestamp')

    errors = pd.DataFrame(index=data.columns, columns=sampleRates, dtype=float)

    for column in data.columns:
        data_groundtruth = data[[column]].dropna()
        data_compare = data_groundtruth.copy()

        for sampleRate in sampleRates:
            measured = data_groundtruth.resample(f'{sampleRate}min').first()
            measured.columns = [f'{sampleRate}min']
            data_compare = pd.concat([data_compare, measured], axis=1)

        for sampleRate in sampleRates:
            interp = data_compare[f'{sampleRate}min'].interpolate(method=method)
            error = np.sqrt(((interp - data_compare[column]) ** 2).mean())
            errors.loc[column, sampleRate] = error

    # ---- Plot RMSE curves ----
    fig, ax = plt.subplots(figsize=(12, 5))
    if colors is None:
        colors = plt.cm.tab10.colors  # default color cycle

    for i, column in enumerate(errors.index):
        ax.plot(errors.columns, errors.loc[column], marker='o', label=column, color=colors[i % len(colors)])

    ax.set_xlabel("Fréquence d\'aquisition [min]")
    ax.set_ylabel("RMSE")
    # ax.set_title("RMSE vs Sampling Rate per Sensor")
    ax.grid(True)
    ax.legend()
    
    return errors, fig

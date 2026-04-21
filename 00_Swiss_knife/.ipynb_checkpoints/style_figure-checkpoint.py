import matplotlib.pyplot as plt

def apply_my_theme(ax):
    """
    Apply a reusable styling/theme to a Matplotlib Axes object.

    Here is an example you can use this as inspiration :

    import pandas as pd
    import numpy as np
    import pytz
    import matplotlib.pyplot as plt
    
    import os
    
    print(os.getcwd())

    import matplotlib.dates as mdates
    import locale

    from matplotlib.ticker import FuncFormatter

    # set timestamp as index
    df = df.set_index("timestamp")

    %matplotlib widget

    # Définir la période à tracer
    start_date = "2025-04-02"
    end_date   = "2025-04-10"
    
    # Filtrer le DataFrame sur l'index (qui doit être de type datetime)
    df_period = df.loc[start_date:end_date]
    
    # Template for curve
    columns_to_plot = ["outTemp_C", "QFA2050_3_Temperature_u6_117"]
    colors     = ["#2b8cbe", "tomato"]
    labels     = ["Extérieur", "U6-117"]
    linestyles = ["--", "-"]
    markers    = [""]*len(columns_to_plot)
    widths     = [1.5, 1.5]
    alphas     = [1, 1]
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    for col, color, label, ls, marker, lw, alpha in zip(
            columns_to_plot, colors, labels, linestyles, markers, widths, alphas):
        if col in df_period.columns:
            ax.plot(df_period.index, df_period[col],
                    label=label,
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    linewidth=lw,
                    alpha=alpha)
    
    # Forcer les limites de l'axe x pour supprimer les marges
    ax.set_xlim([df_period.index.min(), df_period.index.max()+ pd.Timedelta(hours=0.2)])
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Température (°C)")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="-", alpha=0.5)
    
    # date formater
    import matplotlib.dates as mdates
    import locale
    
    locale.setlocale(locale.LC_TIME, 'French_France.1252')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b')) # %H:%M' ))
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    # plt.gcf().autofmt_xdate()

    # Decimal comma formatter for y-axis ---
    def comma_format(x, pos):
    return f"{x:.1f}".replace('.', ',')  # 1 decimal for temperature

    ax.yaxis.set_major_formatter(FuncFormatter(comma_format))
    
    # Apply the custom theme
    apply_my_theme(ax)
    
    fig.tight_layout()
    plt.show()
    
    """
    # Text sizes
    ax.title.set_size(16)           # Plot title
    ax.xaxis.label.set_size(18)     # X-axis title
    ax.yaxis.label.set_size(18)     # Y-axis title
    # ax.tick_params(axis='both', labelsize=16)  # Axis tick labels
    ax.tick_params(axis='x', labelsize=16)  # x-axis ticks length
    ax.tick_params(axis='y', labelsize=16)   # y-axis ticks length
    
    # Legend style
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(16)      # Legend labels
        # leg.set_title_fontsize(16)     # Legend title
        leg.set_frame_on(True)        # Optional: remove legend frame
    
    # Panel border (like panel.border in R)
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')
        spine.set_linewidth(1)
    
    # Optional: hide title
    ax.set_title("")  # equivalent to plot.title = element_blank()

    # Optional: grid style
    ax.grid(True, linestyle="-", alpha=0.6)

    # Optional: background color (like blank panel)
    ax.set_facecolor("white")
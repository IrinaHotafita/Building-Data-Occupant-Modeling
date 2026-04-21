import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_subplots_per_col_idx_X(df, cmap_name='tab10', figsize=None, linewidth=1.5, interactive=True):
    """
    Plot each column of a DataFrame in its own subplot (single column layout),
    using different colors for each subplot.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to plot. Rows are samples, columns are variables.
        PS : DON'T add timestamp in the dataframe
    cmap_name : str
        Matplotlib colormap name for line colors (default 'tab10').
    figsize : tuple or None
        Figure size (width, height). If None, computed automatically.
    linewidth : float
        Width of the plotted lines.
    interactive : bool
        If True, use `%matplotlib widget` for interactive plotting in Jupyter.
        If False, use `%matplotlib inline` for static plotting.
    """
    # Set backend dynamically
    if interactive:
        get_ipython().run_line_magic('matplotlib', 'widget')
    else:
        get_ipython().run_line_magic('matplotlib', 'inline')

    variables = df.columns
    df_reset = df.reset_index(drop=True)

    if figsize is None:
        figsize = (12, 3 * len(variables))

    fig, axes = plt.subplots(
        nrows=len(variables),
        ncols=1,
        figsize=figsize,
        sharex=True
    )

    if len(variables) == 1:
        axes = [axes]

    # Generate a color for each variable using the chosen colormap
    cmap = matplotlib.colormaps[cmap_name]
    colors = [cmap(i / max(1, len(variables)-1)) for i in range(len(variables))]

    for ax, col, c in zip(axes, variables, colors):
        ax.plot(df_reset.index, df_reset[col], color=c, linewidth=linewidth)
        ax.set_ylabel(col)
        ax.grid(True)

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------

import plotly.subplots as sp
import plotly.graph_objects as go

def plot_subplots_per_col_dt_X(
    df, 
    x_column=None, 
    height_per_subplot=300, 
    gridcolor='lightgray', 
    linecolor='black'
):
    """
    Plot each column of a DataFrame in its own Plotly subplot (single column layout),
    with grid lines, axis frames, white background, y-axis labels, and optional x-axis column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to plot. Rows are samples, columns are variables.
    x_column : str or None
        Column name to use as x-axis. If None, uses numeric index.
    height_per_subplot : int
        Height of each subplot in pixels (default 300).
    gridcolor : str
        Color of the grid lines (default 'lightgray').
    linecolor : str
        Color of the axis lines/frame (default 'black').
    """
    variables = df.columns.tolist()
    
    # Remove x_column from variables if present
    if x_column is not None and x_column in variables:
        variables.remove(x_column)
    
    # Determine x-axis values
    if x_column is not None:
        x_values = df[x_column]
    else:
        x_values = df.reset_index(drop=True).index
    
    df_reset = df.reset_index(drop=True)

    fig = sp.make_subplots(
        rows=len(variables),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03
    )

    for i, col in enumerate(variables, start=1):
        fig.add_trace(
            go.Scatter(x=x_values, y=df[col], mode='lines', name=col),
            row=i, col=1
        )
        # Set y-axis title for each subplot
        fig.update_yaxes(title_text=col, row=i, col=1)

    # Layout: background colors
    fig.update_layout(
        height=height_per_subplot * len(variables),
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Grid lines for all axes
    fig.update_xaxes(showgrid=True, gridcolor=gridcolor)
    fig.update_yaxes(showgrid=True, gridcolor=gridcolor)

    # Axis frame/box for all axes
    fig.update_xaxes(showline=True, linewidth=1, linecolor=linecolor, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=linecolor, mirror=True)

    # Label x-axis only for the last subplot
    fig.update_xaxes(title=x_column if x_column is not None else "Sample index", row=len(variables), col=1)

    fig.show()

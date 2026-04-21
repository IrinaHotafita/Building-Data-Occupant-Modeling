import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_presence_violins(df,
                          variables,
                          presence_cols,
                          colors=("green", "tomato"),
                          title=None, ylim=None,
                          ylabel="var [unit]",
                          save_path=None,
                          violinwidth=0.9):
    """
    Plot violin distributions for variables split by presence indicator.
    
    Parameters
    ----------
    - df : pandas.DataFrame
        The input dataframe.
        
    - variables : dict or list
        If dict: { "label" : "column_name", ... }
            Example: 
                variables={
                            "CO2 [ppm]": "EWATCH_3104_Co2_u6_118",
                            "T [°C]": "EWATCH_3104_Temperature_u6_118",
                            "$_Delta$T [°C]": "EWATCH_3104_Temperature_u6_118_corrected_ABS"
                          }
        If list: list of column names, labels auto-generated.
        
    - presence_cols : dict or list
        Presence columns aligned to variables.
        If dict: matching keys to variables dict
            Example:
                presence_cols={
                                "CO2 [ppm]": "EWATCH_3104_Presence_u6_118",
                                "T [°C]": "EWATCH_3104_Presence_u6_118",
                                "$_Delta$T [°C]": "EWATCH_3104_Presence_u6_118"
                              }
        If list: same order as variables list

    - colors : tuple *tuple just mean like this : mytuple = ("apple", "banana", "cherry")
        Colors for (presence=0, presence=1)
            Example:
                colors=("gray", "lightcoral")
    - title : str
        Title for the plot

    - ylim : tuple
        y-axis limits, e.g. (300,1500)

    - ylabel : str
        ylabel (default) "var [unit]"

    - save_path : str
        add the path to save the figure directly
        
    - violinwidth : float
        adjust width for more visibility of each violin
        
    """

    # Normalize inputs: allow dict OR list
    if isinstance(variables, list):
        variable_dict = {v: v for v in variables}
    else:
        variable_dict = variables

    if isinstance(presence_cols, list):
        presence_dict = dict(zip(variable_dict.keys(), presence_cols))
    else:
        presence_dict = presence_cols

    # Build long dataframe
        df_list = []
    for var_label, var_col in variable_dict.items():
        pres_col = presence_dict[var_label]
        temp = df[[var_col, pres_col]].copy().reset_index(drop=True)
        temp.columns = ["Value", "Presence"]
        temp["Variable"] = var_label
        df_list.append(temp)
    
    df_long = pd.concat(df_list, ignore_index=True)


    # Remove NaN presence rows
    df_long = df_long.dropna(subset=["Presence"])

    # Create group label
    df_long["Group"] = df_long["Variable"] + df_long["Presence"].astype(int).astype(str) #""" | Presence="""

    # Determine the plotting order
    # Sort by variable name first, then presence
    group_order = sorted(df_long["Group"].unique(), key=lambda x: (x.split(" | ")[0], x[-1]))

    # Build repeated color list
    color_list = []
    for g in group_order:
        if g.endswith("0"):
            color_list.append(colors[0])
        else:
            color_list.append(colors[1])

    # Plot
    plt.figure(figsize=(14, 6))
    ax = sns.violinplot(
        data=df_long,
        x="Group",
        y="Value",
        cut=0,
        density_norm='area',
        order=group_order,
        width=violinwidth,
        # inner="quart"
    )

    # Recolor violins
    for i, violin in enumerate(ax.collections[:len(group_order)]):
        violin.set_facecolor(color_list[i])
        violin.set_edgecolor("black")
        violin.set_alpha(0.8)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=colors[0], edgecolor="black", label="Presence = 0"),
        mpatches.Patch(facecolor=colors[1], edgecolor="black", label="Presence = 1"),
    ]
    ax.legend(handles=legend_handles, loc="upper right") #, """title="Presence" """

    # Styling
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel(ylabel)

    if ylim is not None:
        plt.ylim(ylim)

    if title:
        plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:                              
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

    
#----------------- Example

# # Importing function from my 00_Swiss_knife
# import sys
# sys.path.append("../../../00_Swiss_knife")

# from violin_presence import plot_presence_violins as vp

# vp(
#     df=df,
#     variables={
#         "U6-118 ": "EWATCH_3104_Co2_u6_118",
#         # "T [°C]": "EWATCH_3104_Temperature_u6_118",
#         # "ΔT [°C]": "EWATCH_3104_Temperature_u6_118_corrected_ABS"
#     },
#     presence_cols={
#         "U6-118 ": "EWATCH_3104_Presence_u6_118",
#         # "T [°C]": "EWATCH_3104_Presence_u6_118",
#         # "ΔT [°C]": "EWATCH_3104_Presence_u6_118"
#     },
#     # title="Room sensor violins",
#     ylim=(350, 1500),
#     ylabel="CO$_2$ [ppm]",
#     save_path='../../results/plots/violintest2.png'
# )
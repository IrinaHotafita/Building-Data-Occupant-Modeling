import pandas as pd

from rpy2 import robjects
from rpy2.robjects import globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

# Load R package
baseline = importr("baseline")

# Define the R function once
robjects.r("""
als_baseline_correction <- function(x0, var_name, lambda = 6, p = 0.05, maxit = 20) {
  x <- as.numeric(x0)
  x <- matrix(x, nrow = 1)
  
  y <- baseline::baseline.als(x, lambda = lambda, p = p, maxit = maxit)
  
  baseline <- matrix(y[["baseline"]], ncol = 1)
  corrected <- matrix(y[["corrected"]], ncol = 1)
  wgts <- matrix(y[["wgts"]], ncol = 1)
  
  colnames_df <- c(
    paste0(var_name, "_baseline"),
    paste0(var_name, "_corrected"),
    paste0(var_name, "_wgts")
  )
  
  result <- data.frame(baseline, corrected, wgts)
  colnames(result) <- colnames_df
  
  return(result)
}
""")

def als_baseline(dfinput, colname,
                 lambda_val=6, p_val=0.05, maxit_val=20):
    """
    Apply ALS baseline correction to a dataframe column (R baseline package)

    Parameters
    ----------
    dfinput : pandas DataFrame
    colname : str
        Name of the column to correct
    lambda_val, p_val, maxit_val : ALS parameters

    Note: input column must be numeric and contain no NaN.
    
    Returns
    -------
    df : pandas DataFrame (original + new ALS columns)
    """

    # Check for NaN in selected column
    if dfinput[colname].isna().any():
        raise ValueError(
            f"Column '{colname}' contains NaN values. "
            "Please clean or fill the data before using als_baseline()."
        )

    # Data safety { .copy() = “work on my own sandbox, never break the user’s DataFrame” }
    dfinput = dfinput.copy()
    
    # Convert df to R
    with localconverter(pandas2ri.converter):
        globalenv["df"] = pandas2ri.py2rpy(dfinput)

    # Pull column as R vector
    r_col = robjects.r(f"df${colname}")

    # Call R function
    result_r = robjects.r["als_baseline_correction"](r_col, colname, lambda_val, p_val, maxit_val)

    # Convert back to pandas
    with localconverter(pandas2ri.converter):
        result_df = pandas2ri.rpy2py(result_r)

    # Reset index of result_df to match dfinpu
    result_df.reset_index(drop=True, inplace=True)
    
    # Concatenate and get the result
    return pd.concat([dfinput.reset_index(drop=True), result_df], axis=1)

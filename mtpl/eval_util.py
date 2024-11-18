import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error,  average_precision_score
import matplotlib.pyplot as plt


def plot_quantiles(
    y_true,
    y_pred,
    weights = None,
    buckets: int = 10,
    **kwargs,
):
    df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "weight": weights}
    ).sort_values("y_pred").reset_index(drop=True)
    df["cum_weight"] = df["weight"].cumsum() / df["weight"].sum()
    step_size = 1 / buckets
    x = []
    y = []
    for i in range(buckets):
        min_weight = i * step_size
        max_weight = (i+1) * step_size
        sub_df = df[(df["cum_weight"] <= max_weight) & (df["cum_weight"] > min_weight)]
        mean_pred = np.average(sub_df["y_pred"], weights=sub_df["weight"])
        mean_true = np.average(sub_df["y_true"], weights=sub_df["weight"])
        x.append(mean_true)
        y.append(mean_pred)
        
    fig, ax = plt.subplots(**kwargs)
    ax.scatter(x, y)
    x_range = [min(x), max(x)]
    ax.plot(x_range, x_range, c="black")
    ax.set_xlabel("True Claim Amount per Year")
    ax.set_ylabel("Predicted Claim Amount per Year")
    return fig, ax
    

def get_metrics(y_true, y_pred, weights=None):
    return {
        "rmse": root_mean_squared_error(y_true, y_pred, sample_weight=weights),
        "mae": mean_absolute_error(y_true, y_pred, sample_weight=weights),
        "bias": np.average(y_true - y_pred, weights=weights),
        "aps": average_precision_score(np.where(y_true > 0, 1, 0), y_pred, sample_weight=weights),
    }

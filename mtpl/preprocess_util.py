import pandas as pd


def print_data_summary(df: pd.DataFrame, target_col: str = "claim_amount_by_time"):
    n = df.shape[0]
    n_pos = df[df[target_col] > 0].shape[0]
    print("Daten {:d}, positive Zielvariable: {:d} ({:.2f}%)".format(n, n_pos, n_pos/n*100))

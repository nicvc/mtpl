import arff
import pandas as pd


def load_freq_df(src: str = "data/freMTPL2freq.arff") -> pd.DataFrame:
    data_freq = arff.load(src)
    df = pd.DataFrame(
        data_freq,
        columns=[
            "IDpol",
            "ClaimNb",
            "Exposure",
            "Area",
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            "VehBrand",
            "VehGas",
            "Density",
            "Region",
        ],
    ).astype({"IDpol": "int64"})
    for c in ["Area", "VehBrand", "Region"]:
        df[c] = df[c].str.replace("'", "")
    return df


def load_sev_df(src: str = "data/freMTPL2sev.arff") -> pd.DataFrame:
    data_sev = arff.load(src)
    return pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"]).astype({"IDpol": "int64"})


def print_data_summary(df: pd.DataFrame, target_col: str):
    n = df.shape[0]
    n_pos = df[df[target_col] > 0].shape[0]
    print("Daten {:d}, positive Zielvariable: {:d} ({:.2f}%)".format(n, n_pos, n_pos/n*100))

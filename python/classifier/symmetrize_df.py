import numpy as np
import pandas as pd


def symmetrize_df(df: pd.DataFrame):
    """
    Symmetrize a dataframe by transforming variables correspondingly.
    """
    jet_features = []
    jet_features += [f"Jet{i}_eta" for i in "0123"]
    jet_features += [f"Jet{i}_phi" for i in "0123"]
    jet_features += [f"Jet{i}_pt" for i in "0123"]
    jet_features += [f"Jet{i}_m" for i in "0123"]

    assert all([f in df.columns for f in jet_features])

    # symmetrize eta
    eta_sign = np.where(df["Jet0_eta"] >= 0, 1, -1)
    for i in "0123":
        df[f"sym_Jet{i}_eta"] = df[f"Jet{i}_eta"] * eta_sign

    # rotate all phis by "Jet0_phi"
    for i in "123":
        df[f"sym_Jet{i}_phi"] = df[f"Jet{i}_phi"] - df["Jet0_phi"]
        # set phis to be in [-pi, pi]
        df[f"sym_Jet{i}_phi"] = (df[f"sym_Jet{i}_phi"] + np.pi) % (2 * np.pi) - np.pi

    df["sym_Jet0_phi"] = 0

    # symmetrize phis by fixing "sym_Jet1_phi" to be positive
    phi_sign = np.where(df["sym_Jet1_phi"] >= 0, 1, -1)
    for i in "123":
        df[f"sym_Jet{i}_phi"] = df[f"sym_Jet{i}_phi"] * phi_sign

    # just make copy of the original features pt and m

    for i in "0123":
        df[f"sym_Jet{i}_pt"] = df[f"Jet{i}_pt"]
        df[f"sym_Jet{i}_m"] = df[f"Jet{i}_m"]

    return df

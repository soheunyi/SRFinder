import numpy as np
import pandas as pd


def symmetrize_df(df: pd.DataFrame):
    """
    Symmetrize a dataframe by transforming variables correspondingly.
    canJet
    """
    jet_features = []
    jet_features += [f"canJet{i}_eta" for i in "0123"]
    jet_features += [f"canJet{i}_phi" for i in "0123"]
    jet_features += [f"canJet{i}_pt" for i in "0123"]
    jet_features += [f"canJet{i}_m" for i in "0123"]

    assert all([f in df.columns for f in jet_features])

    # symmetrize eta
    eta_sign = np.where(df["canJet0_eta"] >= 0, 1, -1)
    for i in "0123":
        df[f"sym_canJet{i}_eta"] = df[f"canJet{i}_eta"] * eta_sign

    # rotate all phis by "canJet0_phi"
    for i in "123":
        df[f"sym_canJet{i}_phi"] = df[f"canJet{i}_phi"] - df["canJet0_phi"]
        # set phis to be in [-pi, pi]
        df[f"sym_canJet{i}_phi"] = (df[f"sym_canJet{i}_phi"] + np.pi) % (
            2 * np.pi
        ) - np.pi

    df["sym_canJet0_phi"] = 0

    # symmetrize phis by fixing "sym_canJet1_phi" to be positive
    phi_sign = np.where(df["sym_canJet1_phi"] >= 0, 1, -1)
    for i in "123":
        df[f"sym_canJet{i}_phi"] = df[f"sym_canJet{i}_phi"] * phi_sign

    # just make copy of the original features pt and m

    for i in "0123":
        df[f"sym_canJet{i}_pt"] = df[f"canJet{i}_pt"]
        df[f"sym_canJet{i}_m"] = df[f"canJet{i}_m"]

    return df

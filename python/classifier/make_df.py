import pandas as pd
import ROOT

ROOT.gROOT.SetBatch(True)
import numpy as np
import os, sys
from glob import glob
from copy import copy

mZ, mH = 91.0, 125.0


def make_df(tree, fourTag=True, verbose=False):
    #    tree = ROOT.TChain("Tree")
    #    tree.Add(temptree)

    # Initialize TTree
    tree.SetBranchStatus("*", 0)
    # tree.SetBranchStatus("nJet", 1)
    tree.SetBranchStatus("Jet_pt", 1)
    tree.SetBranchStatus("Jet_eta", 1)
    tree.SetBranchStatus("Jet_phi", 1)
    tree.SetBranchStatus("Jet_mass", 1)
    tree.SetBranchStatus("weight", 1)
    tree.Show(0)

    nEvts = tree.GetEntries()
    assert nEvts > 0

    ##### Start Conversion #####

    # Event range to process
    iEvtStart = 0
    iEvtEnd = 1000
    iEvtEnd = nEvts
    assert iEvtEnd <= nEvts

    if verbose:
        print(" >> Processing entries: [", iEvtStart, "->", iEvtEnd, ")")

    nWritten = 0
    data = {
        "Jet0_pt": [],
        "Jet1_pt": [],
        "Jet2_pt": [],
        "Jet3_pt": [],
        "Jet0_eta": [],
        "Jet1_eta": [],
        "Jet2_eta": [],
        "Jet3_eta": [],
        "Jet0_phi": [],
        "Jet1_phi": [],
        "Jet2_phi": [],
        "Jet3_phi": [],
        "Jet0_m": [],
        "Jet1_m": [],
        "Jet2_m": [],
        "Jet3_m": [],
        "weight": [],
        "fourTag": [],
        "m01": [],
        "m23": [],
        "m02": [],
        "m13": [],
        "m03": [],
        "m12": [],
        "pt01": [],
        "pt23": [],
        "pt02": [],
        "pt13": [],
        "pt03": [],
        "pt12": [],
        "dR01": [],
        "dR23": [],
        "dR02": [],
        "dR13": [],
        "dR03": [],
        "dR12": [],
        "dR0123": [],
        "dR0213": [],
        "dR0312": [],
        "mZZ0123": [],
        "mZZ0213": [],
        "mZZ0312": [],
        "s4j": [],
        "m4j": [],
    }

    sw = ROOT.TStopwatch()
    sw.Start()
    for iEvt in list(range(iEvtStart, iEvtEnd)):

        # Initialize event
        tree.GetEntry(iEvt)
        if (iEvt + 1) % 1000 == 0 or iEvt + 1 == iEvtEnd:
            sys.stdout.write(
                "\rProcessed "
                + str(iEvt + 1)
                + " of "
                + str(nEvts)
                + " | "
                + str(int((iEvt + 1) * 100.0 / nEvts))
                + "% "
            )
            sys.stdout.flush()

        data["Jet0_pt"].append(copy(tree.Jet_pt[0]))
        data["Jet1_pt"].append(copy(tree.Jet_pt[1]))
        data["Jet2_pt"].append(copy(tree.Jet_pt[2]))
        data["Jet3_pt"].append(copy(tree.Jet_pt[3]))
        data["Jet0_eta"].append(copy(tree.Jet_eta[0]))
        data["Jet1_eta"].append(copy(tree.Jet_eta[1]))
        data["Jet2_eta"].append(copy(tree.Jet_eta[2]))
        data["Jet3_eta"].append(copy(tree.Jet_eta[3]))
        data["Jet0_phi"].append(copy(tree.Jet_phi[0]))
        data["Jet1_phi"].append(copy(tree.Jet_phi[1]))
        data["Jet2_phi"].append(copy(tree.Jet_phi[2]))
        data["Jet3_phi"].append(copy(tree.Jet_phi[3]))
        data["Jet0_m"].append(copy(tree.Jet_mass[0]))
        data["Jet1_m"].append(copy(tree.Jet_mass[1]))
        data["Jet2_m"].append(copy(tree.Jet_mass[2]))
        data["Jet3_m"].append(copy(tree.Jet_mass[3]))

        jets = [
            ROOT.TLorentzVector(),
            ROOT.TLorentzVector(),
            ROOT.TLorentzVector(),
            ROOT.TLorentzVector(),
        ]
        for i in range(4):
            jets[i].SetPtEtaPhiM(
                tree.Jet_pt[i], tree.Jet_eta[i], tree.Jet_phi[i], tree.Jet_mass[i]
            )

        d01, d23 = jets[0] + jets[1], jets[2] + jets[3]
        d02, d13 = jets[0] + jets[2], jets[1] + jets[3]
        d03, d12 = jets[0] + jets[3], jets[1] + jets[2]

        m01, m23 = d01.M(), d23.M()
        m02, m13 = d02.M(), d13.M()
        m03, m12 = d03.M(), d12.M()
        data["m01"].append(m01)
        data["m23"].append(m23)
        data["m02"].append(m02)
        data["m13"].append(m13)
        data["m03"].append(m03)
        data["m12"].append(m12)

        pt01, pt23 = d01.Pt(), d23.Pt()
        pt02, pt13 = d02.Pt(), d13.Pt()
        pt03, pt12 = d03.Pt(), d12.Pt()
        data["pt01"].append(pt01)
        data["pt23"].append(pt23)
        data["pt02"].append(pt02)
        data["pt13"].append(pt13)
        data["pt03"].append(pt03)
        data["pt12"].append(pt12)

        dR01 = jets[0].DeltaR(jets[1])
        dR23 = jets[2].DeltaR(jets[3])
        dR02 = jets[0].DeltaR(jets[2])
        dR13 = jets[1].DeltaR(jets[3])
        dR03 = jets[0].DeltaR(jets[3])
        dR12 = jets[1].DeltaR(jets[2])
        data["dR01"].append(dR01)
        data["dR23"].append(dR23)
        data["dR02"].append(dR02)
        data["dR13"].append(dR13)
        data["dR03"].append(dR03)
        data["dR12"].append(dR12)

        dR0123 = d01.DeltaR(d23)
        dR0213 = d02.DeltaR(d13)
        dR0312 = d03.DeltaR(d12)
        data["dR0123"].append(dR0123)
        data["dR0213"].append(dR0213)
        data["dR0312"].append(dR0312)

        # ZH code.
        ds0123 = [d01, d23] if m01 > m23 else [d23, d01]
        ds0213 = [d02, d13] if m02 > m13 else [d13, d02]
        ds0312 = [d03, d12] if m03 > m12 else [d12, d03]
        # mZH0123 = (ds0123[0]*(mH/ds0123[0].M()) + ds0123[1]*(mZ/ds0123[1].M())).M()
        # mZH0213 = (ds0213[0]*(mH/ds0213[0].M()) + ds0213[1]*(mZ/ds0213[1].M())).M()
        # mZH0312 = (ds0312[0]*(mH/ds0312[0].M()) + ds0312[1]*(mZ/ds0312[1].M())).M()
        # data['mZH0123'].append(mZH0123)
        # data['mZH0213'].append(mZH0213)
        # data['mZH0312'].append(mZH0312)

        mZZ0123 = (
            ds0123[0] * (mZ / ds0123[0].M()) + ds0123[1] * (mZ / ds0123[1].M())
        ).M()
        mZZ0213 = (
            ds0213[0] * (mZ / ds0213[0].M()) + ds0213[1] * (mZ / ds0213[1].M())
        ).M()
        mZZ0312 = (
            ds0312[0] * (mZ / ds0312[0].M()) + ds0312[1] * (mZ / ds0312[1].M())
        ).M()
        data["mZZ0123"].append(mZZ0123)
        data["mZZ0213"].append(mZZ0213)
        data["mZZ0312"].append(mZZ0312)

        # data['st'].append(copy(tree.st))
        # data['stNotCan'].append(copy(tree.stNotCan))

        data["s4j"].append(
            tree.Jet_pt[0] + tree.Jet_pt[1] + tree.Jet_pt[2] + tree.Jet_pt[3]
        )

        data["m4j"].append((jets[0] + jets[1] + jets[2] + jets[3]).M())

        data["weight"].append(copy(tree.weight))
        data["fourTag"].append(fourTag)

    # print

    data["weight"] = np.array(data["weight"], np.float32)
    data["fourTag"] = np.array(data["fourTag"], np.bool_)
    data["Jet0_pt"] = np.array(data["Jet0_pt"], np.float32)
    data["Jet1_pt"] = np.array(data["Jet1_pt"], np.float32)
    data["Jet2_pt"] = np.array(data["Jet2_pt"], np.float32)
    data["Jet3_pt"] = np.array(data["Jet3_pt"], np.float32)
    data["Jet0_eta"] = np.array(data["Jet0_eta"], np.float32)
    data["Jet1_eta"] = np.array(data["Jet1_eta"], np.float32)
    data["Jet2_eta"] = np.array(data["Jet2_eta"], np.float32)
    data["Jet3_eta"] = np.array(data["Jet3_eta"], np.float32)
    data["Jet0_phi"] = np.array(data["Jet0_phi"], np.float32)
    data["Jet1_phi"] = np.array(data["Jet1_phi"], np.float32)
    data["Jet2_phi"] = np.array(data["Jet2_phi"], np.float32)
    data["Jet3_phi"] = np.array(data["Jet3_phi"], np.float32)
    data["Jet0_m"] = np.array(data["Jet0_m"], np.float32)
    data["Jet1_m"] = np.array(data["Jet1_m"], np.float32)
    data["Jet2_m"] = np.array(data["Jet2_m"], np.float32)
    data["Jet3_m"] = np.array(data["Jet3_m"], np.float32)
    data["m01"] = np.array(data["m01"], np.float32)
    data["m23"] = np.array(data["m23"], np.float32)
    data["m02"] = np.array(data["m02"], np.float32)
    data["m13"] = np.array(data["m13"], np.float32)
    data["m03"] = np.array(data["m03"], np.float32)
    data["m12"] = np.array(data["m12"], np.float32)
    data["pt01"] = np.array(data["pt01"], np.float32)
    data["pt23"] = np.array(data["pt23"], np.float32)
    data["pt02"] = np.array(data["pt02"], np.float32)
    data["pt13"] = np.array(data["pt13"], np.float32)
    data["pt03"] = np.array(data["pt03"], np.float32)
    data["pt12"] = np.array(data["pt12"], np.float32)
    data["dR01"] = np.array(data["dR01"], np.float32)
    data["dR23"] = np.array(data["dR23"], np.float32)
    data["dR02"] = np.array(data["dR02"], np.float32)
    data["dR13"] = np.array(data["dR13"], np.float32)
    data["dR03"] = np.array(data["dR03"], np.float32)
    data["dR12"] = np.array(data["dR12"], np.float32)
    data["dR0123"] = np.array(data["dR0123"], np.float32)
    data["dR0213"] = np.array(data["dR0213"], np.float32)
    data["dR0312"] = np.array(data["dR0312"], np.float32)
    data["mZZ0123"] = np.array(data["mZZ0123"], np.float32)
    data["mZZ0213"] = np.array(data["mZZ0213"], np.float32)
    data["mZZ0312"] = np.array(data["mZZ0312"], np.float32)
    data["s4j"] = np.array(data["s4j"], np.float32)
    data["m4j"] = np.array(data["m4j"], np.float32)

    if verbose:
        for key, value in data.items():
            print(key)
            print(value.shape)

    df = pd.DataFrame(data)

    if verbose:
        print("df.dtypes")
        print(df.dtypes)
        print("df.shape", df.shape)

    return df


"""

    df.to_hdf(outfile, key='df', format='table', mode='w')

    sw.Stop()
    print("\n")
    print(" >> nWritten:", nWritten)
    print(" >> Real time:", sw.RealTime()/60.,"minutes")
    print(" >> CPU time: ", sw.CpuTime() /60.,"minutes")
    print(" >> ======================================")
"""

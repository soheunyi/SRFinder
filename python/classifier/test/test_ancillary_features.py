import torch
import ROOT
import sys

sys.path.insert(0, "../")
from ancillary_features import PtEtaPhiM_to_E, get_M_jj, LorentzSum, deltaR


def test():
    for _ in range(100):
        Pt1, Pt2 = 100 * torch.rand(2)
        Eta1, Eta2 = -3 + 6 * torch.rand(2)
        Phi1, Phi2 = -torch.pi + 2 * torch.pi * torch.rand(2)
        M1, M2 = 10 * torch.rand(2)

        jet1 = ROOT.TLorentzVector()
        jet1.SetPtEtaPhiM(Pt1, Eta1, Phi1, M1)
        jet2 = ROOT.TLorentzVector()
        jet2.SetPtEtaPhiM(Pt2, Eta2, Phi2, M2)

        # print(get_M_jj(Pt1, Pt2, Eta1, Eta2, Phi1, Phi2, M1, M2))
        # print((jet1 + jet2).M())

        Pt_jj, Eta_jj, Phi_jj, M_jj = LorentzSum(
            Pt1, Pt2, Eta1, Eta2, Phi1, Phi2, M1, M2
        )
        jet_jj = jet1 + jet2
        e_tol = 1e-3
        try:
            assert torch.abs(Pt_jj - jet_jj.Pt()) < e_tol
            assert torch.abs(Eta_jj - jet_jj.Eta()) < e_tol
            assert torch.abs(Phi_jj - jet_jj.Phi()) < e_tol
            assert torch.abs(M_jj - jet_jj.M()) < e_tol
        except AssertionError:
            print(Pt_jj, jet_jj.Pt())
            print(Pt_jj - jet_jj.Pt())
            print(Eta_jj, jet_jj.Eta())
            print(Eta_jj - jet_jj.Eta())
            print(Phi_jj, jet_jj.Phi())
            print(Phi_jj - jet_jj.Phi())
            print(M_jj, jet_jj.M())
            print(M_jj - jet_jj.M())
            raise


if __name__ == "__main__":
    test()

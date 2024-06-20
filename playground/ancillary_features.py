# Make ancillary features (dijet / quadjet) from jet


import torch


def PtEtaPhiM_to_E(Pt, Eta, _Phi, m):
    return (m**2 + Pt**2 * (1 + torch.sinh(Eta) ** 2)) ** 0.5


def get_M01(
    Pt0: torch.Tensor,
    Pt1: torch.Tensor,
    Eta0: torch.Tensor,
    Eta1: torch.Tensor,
    Phi0: torch.Tensor,
    Phi1: torch.Tensor,
    m0: torch.Tensor,
    m1: torch.Tensor,
):
    E0 = PtEtaPhiM_to_E(Pt0, Eta0, Phi0, m0)
    E1 = PtEtaPhiM_to_E(Pt1, Eta1, Phi1, m1)
    return (
        m0**2
        + m1**2
        + 2
        * (
            E0 * E1
            - Pt0 * Pt1 * (torch.cos(Phi0 - Phi1) + torch.sinh(Eta0) * torch.sinh(Eta1))
        )
    ) ** 0.5


def PtEtaPhi_to_PxPyPz(Pt, Eta, Phi):
    Px = Pt * torch.cos(Phi)
    Py = Pt * torch.sin(Phi)
    Pz = Pt * torch.sinh(Eta)
    return Px, Py, Pz


def LorentzSum(jet0: torch.Tensor, jet1: torch.Tensor, dim=3):
    Pt0, Eta0, Phi0, m0 = get_jet_features(jet0, dim=dim)
    Pt1, Eta1, Phi1, m1 = get_jet_features(jet1, dim=dim)

    Px0, Px1 = Pt0 * torch.cos(Phi0), Pt1 * torch.cos(Phi1)
    Py0, Py1 = Pt0 * torch.sin(Phi0), Pt1 * torch.sin(Phi1)
    Pz0, Pz1 = Pt0 * torch.sinh(Eta0), Pt1 * torch.sinh(Eta1)
    Pt01 = ((Px0 + Px1) ** 2 + (Py0 + Py1) ** 2) ** 0.5
    Eta01 = torch.arcsinh((Pz0 + Pz1) / Pt01)
    Phi01 = torch.arctan2(Py0 + Py1, Px0 + Px1)
    m01 = get_M01(Pt0, Pt1, Eta0, Eta1, Phi0, Phi1, m0, m1)

    return get_jet_from_features(Pt01, Eta01, Phi01, m01)


def deltaR(jet0: torch.Tensor, jet1: torch.Tensor):
    Eta0, Phi0 = jet0[:, 1:2, :], jet0[:, 2:3, :]
    Eta1, Phi1 = jet1[:, 1:2, :], jet1[:, 2:3, :]
    # _, Eta0, Phi0, _ = get_jet_features(jet0)
    # _, Eta1, Phi1, _ = get_jet_features(jet1)
    dPhi = Phi0 - Phi1
    dPhi = (dPhi + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.sqrt((Eta0 - Eta1) ** 2 + dPhi**2)


def get_jet_features(jet: torch.Tensor, dim=3):
    if dim == 3:
        Pt, Eta, Phi, m = (
            jet[:, 0:1, :],
            jet[:, 1:2, :],
            jet[:, 2:3, :],
            jet[:, 3:4, :],
        )
    elif dim == 2:
        Pt, Eta, Phi, m = (
            jet[:, 0:1],
            jet[:, 1:2],
            jet[:, 2:3],
            jet[:, 3:4],
        )
    else:
        raise ValueError("dim should be 2 or 3")
    return Pt, Eta, Phi, m


def get_jet_from_features(
    Pt: torch.Tensor, Eta: torch.Tensor, Phi: torch.Tensor, m: torch.Tensor
):
    return torch.cat([Pt, Eta, Phi, m], dim=1)


def jets_to_dijets(
    jet0: torch.Tensor,
    jet1: torch.Tensor,
    jet2: torch.Tensor,
    jet3: torch.Tensor,
    dim=3,
):
    dijet01 = LorentzSum(jet0, jet1, dim=dim)
    dijet23 = LorentzSum(jet2, jet3, dim=dim)
    dijet02 = LorentzSum(jet0, jet2, dim=dim)
    dijet13 = LorentzSum(jet1, jet3, dim=dim)
    dijet03 = LorentzSum(jet0, jet3, dim=dim)
    dijet12 = LorentzSum(jet1, jet2, dim=dim)

    return dijet01, dijet23, dijet02, dijet13, dijet03, dijet12


def scale_jet(jet: torch.Tensor, scale: torch.Tensor):
    Pt, Eta, Phi, m = get_jet_features(jet)
    Pt = Pt * scale
    m = m * scale
    return get_jet_from_features(Pt, Eta, Phi, m)


def two_dijets_to_quadjet(
    dijet0: torch.Tensor,
    dijet1: torch.Tensor,
):
    mZ = 91.0

    m01 = dijet0[:, 3:4, :]
    m23 = dijet1[:, 3:4, :]

    leadjet01 = torch.where(
        m01 > m23,
        dijet0,
        dijet1,
    )
    followjet01 = torch.where(
        m01 > m23,
        dijet1,
        dijet0,
    )
    quadjet01 = LorentzSum(
        scale_jet(leadjet01, mZ / leadjet01[:, 3:4, :]),
        scale_jet(followjet01, mZ / followjet01[:, 3:4, :]),
    )

    return quadjet01


def dijets_to_quadjets(
    dijet01: torch.Tensor,
    dijet23: torch.Tensor,
    dijet02: torch.Tensor,
    dijet13: torch.Tensor,
    dijet03: torch.Tensor,
    dijet12: torch.Tensor,
):
    quadjet0123 = two_dijets_to_quadjet(dijet01, dijet23)
    quadjet0213 = two_dijets_to_quadjet(dijet02, dijet13)
    quadjet0312 = two_dijets_to_quadjet(dijet03, dijet12)

    return quadjet0123, quadjet0213, quadjet0312


def get_ancillary_features(J: torch.Tensor):
    # J is a tensor of shape (batch, 4 * 4)
    nj_features = 4
    J = J.view(-1, nj_features, 4)
    jet0, jet1, jet2, jet3 = J[:, :, 0:1], J[:, :, 1:2], J[:, :, 2:3], J[:, :, 3:4]

    # augmented jet features
    augmented_jet_features = torch.cat(
        [
            jet0,
            jet1,
            jet2,
            jet3,
            jet0,
            jet2,
            jet1,
            jet3,
            jet0,
            jet3,
            jet1,
            jet2,
        ],
        dim=2,
    ).view(-1, nj_features * 4)

    dijet01, dijet23, dijet02, dijet13, dijet03, dijet12 = jets_to_dijets(
        jet0, jet1, jet2, jet3
    )
    # quadjet0123, quadjet0213, quadjet0312 = dijets_to_quadjets(
    #     dijet01, dijet23, dijet02, dijet13, dijet03, dijet12
    # )

    m01, m23, m02, m13, m03, m12 = (
        dijet01[:, 3:4, :],
        dijet23[:, 3:4, :],
        dijet02[:, 3:4, :],
        dijet13[:, 3:4, :],
        dijet03[:, 3:4, :],
        dijet12[:, 3:4, :],
    )
    dR01, dR23, dR02, dR13, dR03, dR12 = (
        deltaR(jet0, jet1),
        deltaR(jet2, jet3),
        deltaR(jet0, jet2),
        deltaR(jet1, jet3),
        deltaR(jet0, jet3),
        deltaR(jet1, jet2),
    )

    # dijet ancillary features
    dijet_ancillary_features = torch.cat(
        [m01, m23, m02, m13, m03, m12, dR01, dR23, dR02, dR13, dR03, dR12], dim=1
    )

    dR0123, dR0213, dR0312 = (
        deltaR(dijet01, dijet23),
        deltaR(dijet02, dijet13),
        deltaR(dijet03, dijet12),
    )

    m4j = LorentzSum(dijet01, dijet23)[:, 3:4, :]

    # quadjet ancillary features
    quadjet_ancillary_features = torch.cat(
        [dR0123, dR0213, dR0312, m4j, m4j, m4j], dim=1
    )

    return augmented_jet_features, dijet_ancillary_features, quadjet_ancillary_features


def get_m4j(X):
    jet00 = X[:, [0 + 4 * i for i in range(4)]]
    jet01 = X[:, [1 + 4 * i for i in range(4)]]

    jet10 = X[:, [2 + 4 * i for i in range(4)]]
    jet11 = X[:, [3 + 4 * i for i in range(4)]]

    dijet0 = LorentzSum(jet00, jet01, dim=2)
    dijet1 = LorentzSum(jet10, jet11, dim=2)
    quadjet = LorentzSum(dijet0, dijet1, dim=2)
    return quadjet[:, 3].cpu().numpy()

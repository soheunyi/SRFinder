import torch
import torch.nn as nn
import torch.nn.functional as F
from network_blocks import (
    GhostBatchNorm1d,
    NonLU,
    conv1d,
    DijetResNetBlock,
    QuadjetResNetBlock,
    scaler,
)
from ancillary_features import get_ancillary_features


class FvTEncoder(nn.Module):
    def __init__(
        self,
        dim_input_jet_features: int,
        dim_dijet_features: int,
        dim_quadjet_features: int,
        device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        depth: int = 1,
    ):
        """
        nj: number of jet
        """
        super().__init__()

        self.input_dim = dim_input_jet_features * 4  # 4 features per jet
        self.output_dim = dim_quadjet_features

        self.device = device
        self.debug = False
        self.dim_j = dim_input_jet_features

        # engineered dijet features
        self.dim_engineered_d = 2  # m and deltaR
        # total dijet features
        self.dim_d = dim_dijet_features
        # engineered quadjet features
        self.dim_engineered_q = 2  # m and deltaR
        # total quadjet features
        self.dim_q = dim_quadjet_features

        self.dim_out = dim_quadjet_features

        self.name = "ResNet_{}_{}".format(
            dim_dijet_features,
            dim_quadjet_features,
        )

        self.nR = 1
        self.canJetScaler = scaler(self.dim_j, device)

        self.othJetScaler = None

        self.dijetScaler = scaler(self.dim_engineered_d, device)
        self.quadjetScaler = scaler(self.dim_engineered_q, device)

        # embed inputs to DijetResNetBlock in target feature space
        self.jetPtGBN = GhostBatchNorm1d(1)  # only apply to pt
        self.jetEtaGBN = GhostBatchNorm1d(
            1, bias=False
        )  # learn scale for eta, but keep bias at zero for eta flip symmetry to make sense
        self.jetMassGBN = GhostBatchNorm1d(1)  # only apply to mass

        self.jetEmbed = conv1d(
            self.dim_j,
            self.dim_d,
            1,
            name="jet embed",
            batchNorm=False,
        )
        self.dijetGBN = GhostBatchNorm1d(self.dim_engineered_d)
        self.dijetEmbed1 = conv1d(
            self.dim_engineered_d,
            self.dim_d,
            1,
            name="dijet embed",
            batchNorm=False,
        )

        # Stride=3 Kernel=3 reinforce dijet features, in parallel update jet features for next reinforce layer
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|
        self.DijetResNetBlock = DijetResNetBlock(
            self.dim_d,
            device=self.device,
            depth=depth,
            nAveraging=1,
        )

        # embed inputs to QuadjetResNetBlock in target feature space
        self.dijetEmbed2 = conv1d(
            self.dim_d,
            self.dim_q,
            1,
            name="dijet embed",
            batchNorm=False,
        )
        self.quadjetGBN = GhostBatchNorm1d(self.dim_engineered_q)
        self.quadjetEmbed = conv1d(
            self.dim_engineered_q,
            self.dim_q,
            1,
            name="quadjet embed",
            batchNorm=False,
        )

        # Stride=3 Kernel=3 reinforce quadjet features, in parallel update dijet features for next reinforce layer
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4|2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|
        self.QuadjetResNetBlock = QuadjetResNetBlock(
            self.dim_q,
            device=self.device,
            depth=depth,
            nAveraging=1,
        )

        self.depth = depth

        # Create event_conv layers based on depth
        self.event_conv_layers = nn.ModuleList()
        for i in range(self.depth):
            self.event_conv_layers.append(
                conv1d(
                    self.dim_q,
                    self.dim_q,
                    1,
                    name=f"event convolution {i+1}",
                    batchNorm=(i < self.depth - 1),  # No batch norm for the last layer
                )
            )

    def get_quadjet_pixels(self, j: torch.Tensor, d: torch.Tensor, q: torch.Tensor):
        #
        # Build up dijet pixels with jet pixels and dijet ancillary features
        #

        # Embed the jet 4-vectors and dijet ancillary features into the target feature space
        j = self.jetEmbed(j)
        d = self.DijetResNetBlock(j, d)
        #
        # Build up quadjet pixels with dijet pixels and dijet ancillary features
        #

        # Embed the dijet pixels and quadjet ancillary features into the target feature space
        # d0 from DijetResNetBlock since the number of dijet and quadjet features are the same
        d = self.dijetEmbed2(d)
        q = self.QuadjetResNetBlock(d, q)

        return q

    def forward(self, j: torch.Tensor) -> torch.Tensor:
        n = j.shape[0]
        j = j.view(n, self.dim_j, 4)

        j, d, q = get_ancillary_features(j)

        j = j.view(n, self.dim_j, 12)
        d = d.view(n, self.dim_engineered_d, 6)
        q = q.view(n, self.dim_engineered_q, 3)

        # Scale inputs
        #
        j = self.canJetScaler(j)
        d = self.dijetScaler(d)
        q = self.quadjetScaler(q)

        # Learn optimal scale for these inputs
        jPt, jEta, jPhi, jMass = j[:, 0:1, :], j[:, 1:2, :], j[:, 2:3, :], j[:, 3:4, :]

        jPt, jEta, jMass = (
            self.jetPtGBN(jPt),
            self.jetEtaGBN(jEta),
            self.jetMassGBN(jMass),
        )
        j = torch.cat((jPt, jEta, jPhi, jMass), dim=1)
        d = self.dijetGBN(d)
        q = self.quadjetGBN(q)

        # can do these here because they have no eta/phi information
        d = self.dijetEmbed1(d)
        q = self.quadjetEmbed(q)

        # compute the quadjet pixels and average them over the symmetry transformations
        q = self.get_quadjet_pixels(j, d, q)

        # Apply ResNet structure based on depth
        for i in range(self.depth):
            q0 = q.clone()
            q = self.event_conv_layers[i](q)
            q = q + q0
            q = NonLU(q, self.training)

        return q

import torch
import torch.nn as nn
import torch.nn.functional as F
from network_blocks import (
    GhostBatchNorm1d,
    NonLU,
    conv1d,
    DijetResNetBlock,
    layerOrganizer,
    QuadjetResNetBlock,
    scaler,
)
from ancillary_features import get_ancillary_features
from black_box_network import BlackBoxNetwork


class FvTEncoder(nn.Module, BlackBoxNetwork):
    def __init__(
        self,
        dim_input_jet_features: int,
        dim_intermed_dijet_features: int,
        dim_intermed_quadjet_features: int,
        dim_output: int,
        device: str = torch.device("cpu"),
    ):
        """
        nj: number of jet
        """
        super().__init__()

        # For BlackBoxNetwork initialization
        self.input_dim = dim_input_jet_features * 4  # 4 features per jet
        self.output_dim = dim_output

        self.device = device
        self.debug = False
        self.dim_j = dim_input_jet_features

        # engineered dijet features
        self.dim_engineered_d = 2  # m and deltaR
        # total dijet features
        self.dim_d = dim_intermed_dijet_features
        # engineered quadjet features
        self.dim_engineered_q = 2  # m and deltaR
        # total quadjet features
        self.dim_q = dim_intermed_quadjet_features

        self.dim_out = dim_output

        self.name = "ResNet" + "_%d_%d_%d" % (
            dim_intermed_dijet_features,
            dim_intermed_quadjet_features,
            dim_output,
        )

        self.nR = 1
        self.layers = layerOrganizer()
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

        self.layers.addLayer(self.jetEmbed)
        self.layers.addLayer(self.dijetEmbed1)

        # Stride=3 Kernel=3 reinforce dijet features, in parallel update jet features for next reinforce layer
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|
        self.DijetResNetBlock = DijetResNetBlock(
            self.dim_d,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.jetEmbed.index, self.dijetEmbed1.index],
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

        self.layers.addLayer(self.dijetEmbed2, [self.DijetResNetBlock.outputLayer])
        self.layers.addLayer(self.quadjetEmbed, startIndex=self.dijetEmbed2.index)

        # Stride=3 Kernel=3 reinforce quadjet features, in parallel update dijet features for next reinforce layer
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4|2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|
        self.QuadjetResNetBlock = QuadjetResNetBlock(
            self.dim_q,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.dijetEmbed2.index, self.quadjetEmbed.index],
            nAveraging=1,
        )

        self.event_conv_1 = conv1d(
            self.dim_out,
            self.dim_out,
            1,
            name="event convolution 1",
            batchNorm=True,
        )
        self.event_conv_2 = conv1d(
            self.dim_out,
            self.dim_out,
            1,
            name="event convolution 2",
            batchNorm=False,
        )

        # Calculalte score for each quadjet, add them together with corresponding weight, and go to final output layer
        self.select_q = conv1d(
            self.dim_out, 1, 1, name="quadjet selector", batchNorm=True
        )

        self.layers.addLayer(
            self.event_conv_1, [self.QuadjetResNetBlock.reinforce2.conv.index]
        )
        self.layers.addLayer(self.event_conv_2, [self.event_conv_1.index])
        self.layers.addLayer(self.select_q, [self.event_conv_2.index])

    def get_quadjet_pixels(
        self, j: torch.Tensor, mask, d: torch.Tensor, q: torch.Tensor
    ):
        #
        # Build up dijet pixels with jet pixels and dijet ancillary features
        #

        # Embed the jet 4-vectors and dijet ancillary features into the target feature space
        j = self.jetEmbed(j)
        j0 = j.clone()
        d0 = d.clone()
        j = NonLU(j, self.training)
        d = NonLU(d, self.training)

        d, d0 = self.DijetResNetBlock(
            j, d, j0=j0, d0=d0, o=None, mask=mask, debug=self.debug
        )

        #
        # Build up quadjet pixels with dijet pixels and dijet ancillary features
        #

        # Embed the dijet pixels and quadjet ancillary features into the target feature space
        # d0 from DijetResNetBlock since the number of dijet and quadjet features are the same
        d = self.dijetEmbed2(d)
        d = d + d0

        q0 = q.clone()
        d = NonLU(d, self.training)
        q = NonLU(q, self.training)

        q, q0 = self.QuadjetResNetBlock(d, q, d0, q0)

        return q, q0

    def forward(self, j):
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

        # do the same data prep for the other jets if we are using them
        mask = None

        # compute the quadjet pixels and average them over the symmetry transformations
        q, q0 = self.get_quadjet_pixels(j, mask, d, q)

        # Everything from here on out has no dependence on eta/phi flips and minimal dependence on phi rotations

        q = self.event_conv_1(q)
        q = q + q0
        q = NonLU(q, self.training)

        q = self.event_conv_2(q)
        q = q + q0
        q = NonLU(q, self.training)

        # compute a score for each event view (quadjet)
        q_score = self.select_q(q)
        # convert the score to a 'probability' with softmax.
        # This way the classifier is learning which view is most relevant to the classification task at hand.
        q_score = F.softmax(q_score, dim=-1)
        # add together the quadjets with their corresponding probability weight
        event_repr = torch.matmul(q, q_score.transpose(1, 2)).view(-1, self.dim_out)

        return event_repr

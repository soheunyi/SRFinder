import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import (
    GhostBatchNorm1d,
    NonLU,
    conv1d,
    dijetResNetBlock,
    layerOrganizer,
    quadjetResNetBlock,
    scaler,
)
from ancillary_features import get_ancillary_features


class SymmetrizedResNet(nn.Module):
    def __init__(
        self,
        jetFeatures: int,
        dijetFeatures: int,
        quadjetFeatures: int,
        combinatoricFeatures: int,
        device: str = torch.device("cpu"),
        nClasses: int = 1,
    ):
        super().__init__()
        self.device = device
        self.debug = False
        self.nj = jetFeatures
        self.nd, self.nAd = (
            dijetFeatures,
            2,
        )  # total dijet features, engineered dijet features
        self.nq, self.nAq = (
            quadjetFeatures,
            2,
        )  ##6 #total quadjet features, engineered quadjet features

        self.ne = combinatoricFeatures

        self.name = "ResNet" + "_%d_%d_%d" % (dijetFeatures, quadjetFeatures, self.ne)
        self.nClasses = nClasses

        self.nR = 1
        self.layers = layerOrganizer()
        self.canJetScaler = scaler(self.nj, device)

        self.othJetScaler = None

        self.dijetScaler = scaler(self.nAd, device)
        self.quadjetScaler = scaler(self.nAq, device)

        # embed inputs to dijetResNetBlock in target feature space
        self.jetPtGBN = GhostBatchNorm1d(1)  # only apply to pt
        self.jetEtaGBN = GhostBatchNorm1d(
            1, bias=False
        )  # learn scale for eta, but keep bias at zero for eta flip symmetry to make sense
        self.jetMassGBN = GhostBatchNorm1d(1)  # only apply to mass

        self.jetEmbed = conv1d(self.nj, self.nd, 1, name="jet embed", batchNorm=False)
        self.dijetGBN = GhostBatchNorm1d(self.nAd)
        self.dijetEmbed1 = conv1d(
            self.nAd, self.nd, 1, name="dijet embed", batchNorm=False
        )

        self.layers.addLayer(self.jetEmbed)
        self.layers.addLayer(self.dijetEmbed1)

        # Stride=3 Kernel=3 reinforce dijet features, in parallel update jet features for next reinforce layer
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|
        self.dijetResNetBlock = dijetResNetBlock(
            self.nj,
            self.nd,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.jetEmbed.index, self.dijetEmbed1.index],
            nAveraging=1,
        )

        # embed inputs to quadjetResNetBlock in target feature space
        self.dijetEmbed2 = conv1d(
            self.nd, self.nq, 1, name="dijet embed", batchNorm=False
        )
        self.quadjetGBN = GhostBatchNorm1d(self.nAq)
        self.quadjetEmbed = conv1d(
            self.nAq, self.nq, 1, name="quadjet embed", batchNorm=False
        )

        self.layers.addLayer(self.dijetEmbed2, [self.dijetResNetBlock.outputLayer])
        self.layers.addLayer(self.quadjetEmbed, startIndex=self.dijetEmbed2.index)

        # Stride=3 Kernel=3 reinforce quadjet features, in parallel update dijet features for next reinforce layer
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4|2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|
        self.quadjetResNetBlock = quadjetResNetBlock(
            self.nd,
            self.nq,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.dijetEmbed2.index, self.quadjetEmbed.index],
            nAveraging=1,
        )

        self.eventConv1 = conv1d(
            self.ne, self.ne, 1, name="event convolution 1", batchNorm=True
        )
        self.eventConv2 = conv1d(
            self.ne, self.ne, 1, name="event convolution 2", batchNorm=False
        )

        # Calculalte score for each quadjet, add them together with corresponding weight, and go to final output layer
        self.select_q = conv1d(self.ne, 1, 1, name="quadjet selector", batchNorm=True)
        self.out = conv1d(self.ne, self.nClasses, 1, name="out", batchNorm=True)

        self.layers.addLayer(
            self.eventConv1, [self.quadjetResNetBlock.reinforce2.conv.index]
        )
        self.layers.addLayer(self.eventConv2, [self.eventConv1.index])
        self.layers.addLayer(self.select_q, [self.eventConv2.index])
        self.layers.addLayer(self.out, [self.eventConv2.index, self.select_q.index])

    def invPart(self, j: torch.Tensor, mask, d: torch.Tensor, q: torch.Tensor):
        #
        # Build up dijet pixels with jet pixels and dijet ancillary features
        #

        # Embed the jet 4-vectors and dijet ancillary features into the target feature space
        j = self.jetEmbed(j)
        j0 = j.clone()
        d0 = d.clone()
        j = NonLU(j, self.training)
        d = NonLU(d, self.training)

        d, d0 = self.dijetResNetBlock(
            j, d, j0=j0, d0=d0, o=None, mask=mask, debug=self.debug
        )

        #
        # Build up quadjet pixels with dijet pixels and dijet ancillary features
        #

        # Embed the dijet pixels and quadjet ancillary features into the target feature space
        d = self.dijetEmbed2(d)
        d = (
            d + d0
        )  # d0 from dijetResNetBlock since the number of dijet and quadjet features are the same
        q0 = q.clone()
        d = NonLU(d, self.training)
        q = NonLU(q, self.training)

        q, q0 = self.quadjetResNetBlock(d, q, d0=d0, q0=q0)

        return q, q0

    def forward(self, j):
        n = j.shape[0]
        j = j.view(n, self.nj, 4)

        j, d, q = get_ancillary_features(j)

        j = j.view(n, self.nj, 12)
        d = d.view(n, self.nAd, 6)
        q = q.view(n, self.nAq, 3)

        # print(j.shape, d.shape, q.shape)
        #
        # Scale inputs
        #
        j = self.canJetScaler(j)
        d = self.dijetScaler(d)
        q = self.quadjetScaler(q)

        # Learn optimal scale for these inputs
        # j[:,0:1,:] = self.jetPtGBN(  j[:,0:1,:])
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
        q, q0 = self.invPart(j, mask, d, q)

        # Everything from here on out has no dependence on eta/phi flips and minimal dependence on phi rotations

        q = self.eventConv1(q)
        q = q + q0
        q = NonLU(q, self.training)

        q = self.eventConv2(q)
        q = q + q0
        q = NonLU(q, self.training)

        # compute a score for each event view (quadjet)
        q_score = self.select_q(q)
        # convert the score to a 'probability' with softmax.
        # This way the classifier is learning which view is most relevant to the classification task at hand.
        q_score = F.softmax(q_score, dim=-1)
        # add together the quadjets with their corresponding probability weight
        e = torch.matmul(q, q_score.transpose(1, 2))
        q_score = q_score.view(n, 3)

        e = e.view(n, self.ne, 1)
        # project the final event-level pixel into the class score space
        c_score = self.out(e)
        c_score = c_score.view(n, self.nClasses)

        return c_score, q_score

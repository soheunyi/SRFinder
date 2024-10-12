import collections
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Lin_View(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


def ReLU(x):
    return F.relu(x)


def SiLU(
    x,
):  # SiLU https://arxiv.org/pdf/1702.03118.pdf   Swish https://arxiv.org/pdf/1710.05941.pdf
    return x * torch.sigmoid(x)


def NonLU(x, training=False) -> torch.Tensor:  # Non-Linear Unit
    # return ReLU(x)
    # return F.rrelu(x, training=training)
    # return F.leaky_relu(x, negative_slope=0.1)
    return SiLU(x)
    # return F.elu(x)


def ncr(n, r):
    r = min(r, n - r)
    if r == 0:
        return 1
    numer, denom = 1, 1
    for i in range(n, n - r, -1):
        numer *= i
    for i in range(1, r + 1, 1):
        denom *= i
    return numer // denom  # double slash means integer division or "floor" division


class Activation(nn.Module):
    def __init__(
        self,
        activation: Literal[
            "ReLU", "RReLU", "LeakyReLU", "SiLU", "ELU", "identity", "sigmoid", "sine"
        ],
    ):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        if self.activation == "ReLU":
            return F.relu(x)
        elif self.activation == "RReLU":
            return F.rrelu(x)
        elif self.activation == "LeakyReLU":
            return F.leaky_relu(x)
        elif self.activation == "SiLU":
            return F.silu(x)
        elif self.activation == "ELU":
            return F.elu(x)
        elif self.activation == "identity":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "sine":
            return torch.sin(x)
        else:
            raise NotImplementedError


class ScaleAndShift(nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return self.scale * x + self.shift


class Clip(nn.Module):
    def __init__(self, min: float, max: float):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clip(x, min=self.min, max=self.max)


class MixedActivation(nn.Module):
    def __init__(self, activation_nets: list[nn.Module]):
        super().__init__()
        self.activations = nn.ModuleList(activation_nets)

    def forward(self, x):
        x_new = torch.zeros_like(x)
        for i in range(len(self.activations)):
            x_new[..., i] = self.activations[i](x[..., i])
        return x_new


class stats:
    def __init__(self):
        self.grad = collections.OrderedDict()
        self.mean = collections.OrderedDict()
        self.std = collections.OrderedDict()
        self.summary = ""

    def update(self, attr, grad):
        try:
            self.grad[attr] = torch.cat((self.grad[attr], grad), dim=0)
        except (KeyError, TypeError):
            self.grad[attr] = grad.clone()

    def compute(self):
        self.summary = ""
        self.grad["combined"] = None
        for attr, grad in self.grad.items():
            try:
                self.grad["combined"] = torch.cat((self.grad["combined"], grad), dim=1)
            except TypeError:
                self.grad["combined"] = grad.clone()

            self.mean[attr] = grad.mean(dim=0).norm()
            self.std[attr] = grad.std()
            # self.summary += attr+': <%1.1E> +/- %1.1E r=%1.1E'%(self.mean[attr],self.std[attr],self.mean[attr]/self.std[attr])
        self.summary = "grad: <%1.1E> +/- %1.1E SNR=%1.1f" % (
            self.mean["combined"],
            self.std["combined"],
            (self.mean["combined"] / self.std["combined"]).log10(),
        )

    def dump(self):
        for attr, grad in self.grad.items():
            print(attr, grad.shape, grad.mean(dim=0).norm(2), grad.std())

    def reset(self):
        for attr in self.grad:
            self.grad[attr] = None


def make_hook(gradStats, module, attr):
    def hook(grad):
        gradStats.update(attr, grad / getattr(module, attr).norm(2))

    return hook


class scaler(
    nn.Module
):  # https://arxiv.org/pdf/1705.08741v2.pdf has what seem like typos in GBN definition. I've replaced the running mean and std rules with Adam-like updates.
    def __init__(self, features, device):
        super().__init__()
        self.features = features
        self.register_buffer(
            "m", torch.zeros((1, self.features, 1), dtype=torch.float).to(device)
        )
        # For type hinting
        self.m: Tensor

        self.register_buffer(
            "s", torch.ones((1, self.features, 1), dtype=torch.float).to(device)
        )
        self.s: Tensor

    def forward(self, x, mask=None, debug=False):
        x = x - self.m
        x = x / self.s
        return x


class GhostBatchNorm1d(
    nn.Module
):  # https://arxiv.org/pdf/1705.08741v2.pdf has what seem like typos in GBN definition.
    # I've replaced the running mean and std rules with Adam-like updates.
    def __init__(
        self,
        features: int,
        ghost_batch_size: int = 32,
        number_of_ghost_batches: int = 32,
        nAveraging: int = 1,
        eta: float = 0.9,
        bias: bool = True,
    ):
        super().__init__()
        self.features = features

        # Declare buffers with type hints
        self.register_buffer("gbs", torch.tensor(ghost_batch_size, dtype=torch.long))
        self.gbs: Tensor  # Type hint for Pylance

        self.register_buffer(
            "ngb", torch.tensor(number_of_ghost_batches * nAveraging, dtype=torch.long)
        )
        self.ngb: Tensor

        self.register_buffer(
            "bessel_correction",
            torch.tensor(
                ghost_batch_size / (ghost_batch_size - 1.0), dtype=torch.float
            ),
        )
        self.bessel_correction: Tensor

        self.register_buffer("eta", torch.tensor(eta, dtype=torch.float))
        self.eta: Tensor

        self.register_buffer("m", torch.zeros((1, self.features, 1), dtype=torch.float))
        self.m: Tensor

        self.register_buffer("s", torch.ones((1, self.features, 1), dtype=torch.float))
        self.s: Tensor

        self.register_buffer(
            "m_biased", torch.zeros((1, self.features, 1), dtype=torch.float)
        )
        self.m_biased: Tensor

        self.register_buffer(
            "s_biased", torch.zeros((1, self.features, 1), dtype=torch.float)
        )
        self.s_biased: Tensor

        # use Adam style updates for running mean and standard deviation https://arxiv.org/pdf/1412.6980.pdf
        self.register_buffer("t", torch.tensor(0, dtype=torch.float))
        self.t: Tensor

        self.register_buffer("alpha", torch.tensor(0.001, dtype=torch.float))
        self.alpha: Tensor

        self.register_buffer("beta1", torch.tensor(0.9, dtype=torch.float))
        self.beta1: Tensor

        self.register_buffer("beta2", torch.tensor(0.999, dtype=torch.float))
        self.beta2: Tensor

        self.register_buffer("eps", torch.tensor(1e-5, dtype=torch.float))
        self.eps: Tensor

        self.register_buffer(
            "m_biased_first_moment",
            torch.zeros((1, self.features, 1), dtype=torch.float),
        )
        self.m_biased_first_moment: Tensor

        self.register_buffer(
            "s_biased_first_moment",
            torch.zeros((1, self.features, 1), dtype=torch.float),
        )
        self.s_biased_first_moment: Tensor

        self.register_buffer(
            "m_biased_second_moment",
            torch.zeros((1, self.features, 1), dtype=torch.float),
        )
        self.m_biased_second_moment: Tensor

        self.register_buffer(
            "s_biased_second_moment",
            torch.zeros((1, self.features, 1), dtype=torch.float),
        )
        self.s_biased_second_moment: Tensor

        self.register_buffer(
            "m_first_moment", torch.zeros((1, self.features, 1), dtype=torch.float)
        )
        self.m_first_moment: Tensor

        self.register_buffer(
            "s_first_moment", torch.zeros((1, self.features, 1), dtype=torch.float)
        )
        self.s_first_moment: Tensor

        self.register_buffer(
            "m_second_moment", torch.zeros((1, self.features, 1), dtype=torch.float)
        )
        self.m_second_moment: Tensor

        self.register_buffer(
            "s_second_moment", torch.zeros((1, self.features, 1), dtype=torch.float)
        )
        self.s_second_moment: Tensor

        self.gamma = nn.Parameter(torch.ones(self.features))
        self.bias = nn.Parameter(torch.zeros(self.features))
        self.bias.requires_grad = bias

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, debug: bool = False):
        if self.training:
            batch_size = x.shape[0]
            pixels = x.shape[2]
            # if self.ngb: # if number of ghost batches is specified, compute corresponding ghost batch size

            self.gbs = batch_size // self.ngb
            # self.ngb = batch_size // self.gbs

            # self.register_buffer('gbs', torch.tensor(batch_size//16, dtype=torch.long))
            # else: # ghost batch size is specified, compute corresponding number of ghost batches
            # self.ngb = batch_size // self.gbs

            #
            # Apply batch normalization with Ghost Batch statistics
            #

            # TODO: Problematic!!!!!!!!!!!!!! What if the number of ghost batches is not a factor of the batch size?
            # The last batch need not be the same size as the others.
            # print("x.shape", x.shape)
            x = (
                x.transpose(1, 2)
                .contiguous()
                .view(self.ngb, self.gbs * pixels, self.features, 1)
            )
            # print("x.shape", x.shape)
            # raise ValueError("stop here")

            if mask is None:
                gbm = x.mean(dim=1, keepdim=True) if self.bias.requires_grad else 0
                gbv = x.var(dim=1, keepdim=True)
                gbs = (gbv + self.eps).sqrt()

                # Use mean over ghost batches for running mean and std
                bm = gbm.detach().mean(dim=0) if self.bias.requires_grad else 0
                bs = gbs.detach().mean(dim=0)  # / self.bessel_correction
            else:
                # Compute masked mean and std for each ghost batch
                mask = mask.view(self.ngb, self.gbs * pixels, 1, 1)
                nUnmasked = (mask == 0).sum(dim=1, keepdim=True).float()
                denomMean = (
                    nUnmasked + (nUnmasked == 0).float()
                )  # prevent divide by zero
                denomVar = (
                    nUnmasked
                    + 2 * (nUnmasked == 0).float()
                    + (nUnmasked == 1).float()
                    - 1
                )  # prevent divide by zero with bessel correction
                x = x.masked_fill(mask, 0)
                xs = x.sum(dim=1, keepdim=True)
                gbm = xs / denomMean if self.bias.requires_grad else 0
                x2 = x**2
                x2s = x2.sum(dim=1, keepdim=True)
                x2m = x2s / denomMean
                gbv = x2m - gbm**2
                gbv = gbv * nUnmasked / denomVar
                gbs = (gbv + self.eps).sqrt()

                # Compute masked mean and std over the whole batch
                nUnmasked = nUnmasked.sum(dim=0)
                denomMean = (
                    nUnmasked + (nUnmasked == 0).float()
                )  # prevent divide by zero
                denomVar = (
                    nUnmasked
                    + 2 * (nUnmasked == 0).float()
                    + (nUnmasked == 1).float()
                    - 1
                )  # prevent divide by zero with bessel correction
                bm = (
                    xs.detach().sum(dim=0) / denomMean if self.bias.requires_grad else 0
                )
                x2s = x2s.detach().sum(dim=0)
                x2m = x2s / denomMean
                bv = x2m - bm**2
                bv = bv * nUnmasked / denomVar
                bs = (bv + self.eps).sqrt()

            x = x - gbm
            x = x / gbs
            x = x.view(batch_size, pixels, self.features)
            x = self.gamma * x
            x = x + self.bias
            x = x.transpose(
                1, 2
            )  # back to standard indexing for convolutions: [batch, feature, pixel]

            #
            # Keep track of running mean and standard deviation.
            #

            # Simplest possible method
            self.m = self.eta * self.m + (1 - self.eta) * bm
            self.s = self.eta * self.s + (1 - self.eta) * bs

            ###########################################################################
            #################### How Introduced GBN is Computed #######################
            ###########################################################################

            # weights = self.eta ** torch.arange(
            #     self.ngb, dtype=self.m.dtype, device=self.m.device
            # )
            # weights = weights.view(self.ngb, 1, 1, 1)

            # self.m = (
            #     self.eta**self.ngb * self.m
            #     + (1 - self.eta) * torch.sum(weights * gbm[:, 0:1, :, :], dim=0)
            #     if self.bias.requires_grad
            #     else self.m
            # )
            # self.s = self.eta**self.ngb * self.s + (1 - self.eta) * torch.sum(
            #     weights * gbs[:, 0:1, :, :], dim=0
            # )

            ###########################################################################
            ###########################################################################
            ###########################################################################

            # # Simplest method + bias correction
            # self.m_biased = self.eta*self.m_biased + (1-self.eta)*bm
            # self.s_biased = self.eta*self.s_biased + (1-self.eta)*bs
            # # increment time step for use in bias correction
            # self.t = self.t+1
            # self.m = self.m_biased / (1-self.eta**self.t)
            # self.s = self.s_biased / (1-self.eta**self.t)

            # # Adam inspired method
            # # get 'gradients'
            # m_grad = bm - self.m
            # s_grad = bs - self.s

            # # update biased first moment estimate
            # self.m_biased_first_moment  = self.beta1 * self.m_biased_first_moment   +  (1-self.beta1) * m_grad
            # self.s_biased_first_moment  = self.beta1 * self.s_biased_first_moment   +  (1-self.beta1) * s_grad

            # # update biased second moment estimate
            # self.m_biased_second_moment = self.beta2 * self.m_biased_second_moment  +  (1-self.beta2) * m_grad**2
            # self.s_biased_second_moment = self.beta2 * self.s_biased_second_moment  +  (1-self.beta2) * s_grad**2

            # # increment time step for use in bias correction
            # self.t = self.t+1

            # # correct bias
            # self.m_first_moment  = self.m_biased_first_moment  / (1-self.beta1**self.t)
            # self.s_first_moment  = self.s_biased_first_moment  / (1-self.beta1**self.t)
            # self.m_second_moment = self.m_biased_second_moment / (1-self.beta2**self.t)
            # self.s_second_moment = self.s_biased_second_moment / (1-self.beta2**self.t)

            # # update running mean and standard deviation
            # self.m = self.m + self.alpha * self.m_first_moment / (self.m_second_moment+self.eps).sqrt()
            # self.s = self.s + self.alpha * self.s_first_moment / (self.s_second_moment+self.eps).sqrt()

            return x
        else:
            # inference stage, use running mean and standard deviation
            x = x - self.m.type(x.dtype)
            x = x / self.s.type(x.dtype)
            x = x.transpose(1, 2)
            x = x * self.gamma
            x = x + self.bias
            x = x.transpose(1, 2)

            return x


class conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        groups=1,
        name=None,
        index=None,
        doGradStats=False,
        hiddenIn=False,
        hiddenOut=False,
        batchNorm=False,
        batchNormMomentum=0.9,
        nAveraging=1,
    ):
        super().__init__()
        self.bias = (
            bias and not batchNorm
        )  # if doing batch norm, bias is in BN layer, not convolution
        self.module = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=self.bias,
            groups=groups,
        )
        if batchNorm:
            self.batchNorm = GhostBatchNorm1d(
                out_channels, nAveraging=nAveraging, eta=batchNormMomentum, bias=bias
            )  # nn.BatchNorm1d(out_channels)
        else:
            self.batchNorm = False

        self.hiddenIn = hiddenIn
        if self.hiddenIn:
            self.moduleHiddenIn = nn.Conv1d(in_channels, in_channels, 1)
        self.hiddenOut = hiddenOut
        if self.hiddenOut:
            self.moduleHiddenOut = nn.Conv1d(out_channels, out_channels, 1)
        self.name = name
        self.index = index
        self.gradStats = None
        self.k = 1.0 / (in_channels * kernel_size)
        if doGradStats:
            self.gradStats = stats()
            self.module.weight.register_hook(
                make_hook(self.gradStats, self.module, "weight")
            )
            # self.module.bias  .register_hook( make_hook(self.gradStats, self.module, 'bias'  ) )

    def randomize(self):
        if self.hiddenIn:
            nn.init.uniform_(self.moduleHiddenIn.weight, -(self.k**0.5), self.k**0.5)
            nn.init.uniform_(self.moduleHiddenIn.bias, -(self.k**0.5), self.k**0.5)
        if self.hiddenOut:
            nn.outit.uniform_(self.moduleHiddenOut.weight, -(self.k**0.5), self.k**0.5)
            nn.outit.uniform_(self.moduleHiddenOut.bias, -(self.k**0.5), self.k**0.5)
        nn.init.uniform_(self.module.weight, -(self.k**0.5), self.k**0.5)
        if self.bias:
            nn.init.uniform_(self.module.bias, -(self.k**0.5), self.k**0.5)

    def forward(self, x: torch.Tensor, mask=None, debug=False) -> torch.Tensor:
        if self.hiddenIn:
            x = NonLU(self.moduleHiddenIn(x), self.moduleHiddenIn.training)
        if self.hiddenOut:
            x = NonLU(self.module(x), self.module.training)
            return self.moduleHiddenOut(x)

        x = self.module(x)
        if self.batchNorm:
            x = self.batchNorm(x, mask=mask, debug=debug)
        return x


class linear(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        name=None,
        index=None,
        doGradStats=False,
        bias=True,
    ):
        super().__init__()
        self.module = nn.Linear(in_channels, out_channels, bias=bias)
        self.bias = bias
        self.name = name
        self.index = index
        self.gradStats = None
        self.k = 1.0 / in_channels
        if doGradStats:
            self.gradStats = stats()
            self.module.weight.register_hook(
                make_hook(self.gradStats, self.module, "weight")
            )
            # self.module.bias  .register_hook( make_hook(self.gradStats, self.module, 'bias'  ) )

    def randomize(self):
        nn.init.uniform_(self.module.weight, -(self.k**0.5), self.k**0.5)
        if self.bias:
            nn.init.uniform_(self.module.bias, -(self.k**0.5), self.k**0.5)

    def forward(self, x):
        return self.module(x)


class layerOrganizer:
    def __init__(self):
        self.layers = collections.OrderedDict()
        self.nTrainableParameters = 0

    def addLayer(self, newLayer, inputLayers=None, startIndex=1):
        if inputLayers:
            # [layer.index for layer in inputLayers]
            inputIndicies = inputLayers
            newLayer.index = max(inputIndicies) + 1
        else:
            newLayer.index = startIndex

        try:
            self.layers[newLayer.index].append(newLayer)
        except (KeyError, AttributeError):
            self.layers[newLayer.index] = [newLayer]

    def countTrainableParameters(self):
        self.nTrainableParameters = 0
        for index in self.layers:
            for layer in self.layers[index]:
                for param in layer.parameters():
                    self.nTrainableParameters += (
                        param.numel() if param.requires_grad else 0
                    )

    def setLayerRequiresGrad(self, index, requires_grad=True):
        self.countTrainableParameters()
        print("Change trainable parameters from", self.nTrainableParameters, end=" ")
        try:  # treat index as list of indices
            for i in index:
                for layer in self.layers[i]:
                    for param in layer.parameters():
                        param.requires_grad = requires_grad
        except TypeError:  # index is just an int
            for layer in self.layers[index]:
                for param in layer.parameters():
                    param.requires_grad = requires_grad
        self.countTrainableParameters()
        print("to", self.nTrainableParameters)

    def initLayer(self, index):
        try:  # treat index as list of indices
            print("Rerandomize layer indicies", index)
            for i in index:
                for layer in self.layers[i]:
                    layer.randomize()
        except TypeError:  # index is just an int
            print("Rerandomize layer index", index)
            for layer in self.layers[index]:
                layer.randomize()

    def computeStats(self):
        for index in self.layers:
            for layer in self.layers[index]:
                layer.gradStats.compute()

    def resetStats(self):
        for index in self.layers:
            for layer in self.layers[index]:
                layer.gradStats.reset()

    def print(self):
        for index in self.layers:
            print("----- Layer %2d -----" % (index))
            for layer in self.layers[index]:
                print("|", layer.name.ljust(40), end="")
            print("")
            for layer in self.layers[index]:
                if layer.gradStats:
                    print("|", layer.gradStats.summary.ljust(40), end="")
                else:
                    print("|", " " * 40, end="")
            print("")
            # for layer in self.layers[index]:
            #     print('|',str(layer.module).ljust(45), end=' ')
            # print('|')


class MultiHeadAttention(
    nn.Module
):  # https://towardsdatascience.com/how-to-code-the-Transformer-in-pytorch-24db27c8f9ec https://arxiv.org/pdf/1706.03762.pdf
    def __init__(
        self,
        dim_query=8,
        dim_key=8,
        dim_value=8,
        dim_attention=8,
        heads=2,
        dim_valueAttention=None,
        groups_query=1,
        groups_key=1,
        groups_value=1,
        dim_out=8,
        outBias=False,
        selfAttention=False,
        layers=None,
        inputLayers=None,
        bothAttention=False,
        iterations=1,
    ):
        super().__init__()

        self.h = heads  # len(heads) #heads
        self.da = dim_attention  # sum(heads)#dim_attention
        self.dq = dim_query
        self.dk = dim_key
        self.dv = dim_value
        self.dh = self.da // self.h  # max(heads) #self.da // self.h
        self.dva = dim_valueAttention if dim_valueAttention else dim_attention
        self.dvh = self.dva // self.h
        self.do = dim_out
        self.iter = iterations
        self.sqrt_dh = np.sqrt(self.dh)
        self.selfAttention = selfAttention
        self.bothAttention = bothAttention

        self.q_linear = conv1d(
            self.dq,
            self.da,
            1,
            groups=groups_query,
            name="attention query linear",
            batchNorm=False,
        )
        self.k_linear = conv1d(
            self.dk,
            self.da,
            1,
            groups=groups_key,
            name="attention key   linear",
            batchNorm=False,
        )
        self.v_linear = conv1d(
            self.dv,
            self.dva,
            1,
            groups=groups_value,
            name="attention value linear",
            batchNorm=False,
        )
        if self.bothAttention:
            self.sq_linear = conv1d(
                self.dk,
                self.da,
                1,
                groups=groups_key,
                name="self attention query linear",
            )
            self.so_linear = conv1d(
                self.dva, self.dk, 1, name="self attention out linear"
            )
        self.o_linear = conv1d(
            self.dva,
            self.do,
            1,
            stride=1,
            name="attention out   linear",
            bias=outBias,
            batchNorm=True,
        )

        self.negativeInfinity = torch.tensor(-1e9, dtype=torch.float).to("cpu")

        if layers:
            layers.addLayer(self.q_linear, inputLayers)
            layers.addLayer(self.k_linear, inputLayers)
            layers.addLayer(self.v_linear, inputLayers)
            layers.addLayer(
                self.o_linear,
                [self.q_linear.index, self.v_linear.index, self.k_linear.index],
            )

    def attention(self, q, k, v, mask, debug=False):

        q_k_overlap = torch.matmul(q, k.transpose(2, 3)) / self.sqrt_dh
        if mask is not None:
            if self.selfAttention:
                q_k_overlap = q_k_overlap.masked_fill(mask, self.negativeInfinity)
            mask = mask.transpose(2, 3)
            q_k_overlap = q_k_overlap.masked_fill(mask, self.negativeInfinity)

        v_probability = F.softmax(
            q_k_overlap, dim=-1
        )  # compute joint probability distribution for which values best correspond to the query
        v_weights = v_probability  # * v_score
        if mask is not None:
            v_weights = v_weights.masked_fill(mask, 0)
            if self.selfAttention:
                mask = mask.transpose(2, 3)
                v_weights = v_weights.masked_fill(mask, 0)
        if debug:
            print("q\n", q[0])
            print("k\n", k[0])
            print("mask\n", mask[0])
            print("v_probability\n", v_probability[0])
            print("v_weights\n", v_weights[0])
            print("v\n", v[0])

        output = torch.matmul(v_weights, v)
        if debug:
            print("output\n", output[0])
            input()
        return output

    def forward(
        self, q, k, v, q0=None, mask=None, qLinear=0, debug=False, selfAttention=False
    ):

        bs = q.shape[0]
        qsl = q.shape[2]
        sq = None
        k = self.k_linear(k, mask)
        v = self.v_linear(v, mask)

        # check if all items are going to be masked
        sl = mask.shape[1]
        vqk_mask = mask.sum(dim=1) == sl
        vqk_mask = vqk_mask.view(bs, 1, 1).repeat(1, 1, qsl)

        # #hack to make unequal head dimensions 3 and 6, add three zero padded features before splitting into two heads of dim 6
        # q = F.pad(input=q, value=0, pad=(0,0,1,0,0,0))
        # k = F.pad(input=k, value=0, pad=(0,0,1,0,0,0))
        # v = F.pad(input=v, value=0, pad=(0,0,1,0,0,0))

        # split into heads
        k = k.view(bs, self.h, self.dh, -1)
        v = v.view(bs, self.h, self.dvh, -1)

        # transpose to get dimensions bs * h * sl * (da//h==dh)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)
        mask = mask.view(bs, 1, sl, 1)

        # now do q transformations iter number of times
        for i in range(1, self.iter + 1):
            # q0 = q.clone()
            if selfAttention:
                q = self.sq_linear(q)
            else:
                q = self.q_linear(q, vqk_mask)

            q = q.view(bs, self.h, self.dh, -1)
            q = q.transpose(2, 3)

            # calculate attention
            vqk = self.attention(
                q, k, v, mask, debug
            )  # outputs a linear combination of values (v) given the overlap of the queries (q) with the keys (k)

            # concatenate heads and put through final linear layer
            vqk = vqk.transpose(2, 3).contiguous().view(bs, self.dva, qsl)

            if debug:
                print("vqk\n", vqk[0])
                input()
            if selfAttention:
                vqk = self.so_linear(vqk)
            else:
                vqk = self.o_linear(vqk, vqk_mask)

            # if all the input items are masked, we don't want the bias term of the output layer to have any impact
            vqk = vqk.masked_fill(vqk_mask, 0)

            q = q0 + vqk

            if i == self.iter:
                q0 = q.clone()
            if not selfAttention:
                q = NonLU(q, self.training)

        return q, q0


class Norm(
    nn.Module
):  # https://towardsdatascience.com/how-to-code-the-Transformer-in-pytorch-24db27c8f9ec#1b3f
    def __init__(self, d_model, eps=1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        x = x.transpose(self.dim, -1)
        x = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        x = x.transpose(self.dim, -1)
        return x


class encoder(nn.Module):
    def __init__(
        self, inFeatures, hiddenFeatures, outFeatures, dropout, transpose=False
    ):
        super().__init__()
        self.ni = inFeatures
        self.nh = hiddenFeatures
        self.no = outFeatures
        self.d = dropout
        self.transpose = transpose

        self.input = nn.Linear(self.ni, self.nh)
        self.dropout = nn.Dropout(self.d)
        self.output = nn.Linear(self.nh, self.no)

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2)
        x = self.input(x)
        x = self.dropout(x)
        x = NonLU(x, self.training)
        x = self.output(x)
        if self.transpose:
            x = x.transpose(1, 2)
        return x


class Transformer(
    nn.Module
):  # Attention is All You Need https://arxiv.org/pdf/1706.03762.pdf
    def __init__(self, features):
        self.nf = features
        self.encoderSelfAttention = MultiHeadAttention(1, self.nf, selfAttention=True)
        self.normEncoderSelfAttention = Norm(self.nf)
        self.encoderFF = encoder(self.nf, self.nf * 2)  # *4 in the paper
        self.normEncoderFF = Norm(self.nf)

        self.decoderSelfAttention = MultiHeadAttention(1, self.nf, selfAttention=True)
        self.normDecoderSelfAttention = Norm(self.nf)
        self.decoderAttention = MultiHeadAttention(1, self.nf)
        self.normDecoderAttention = Norm(self.nf)
        self.decoderFF = encoder(self.nf, self.nf * 2)  # *4 in the paper
        self.normDecoderFF = Norm(self.nf)

    def forward(self, inputs, mask, outputs):
        # encoder block
        inputs = self.normEncoderSelfAttention(
            inputs + self.encoderSelfAttention(inputs, inputs, inputs, mask=mask)
        )
        inputs = self.normEncoderFF(inputs + self.encoderFF(inputs))

        # decoder block
        outputs = self.normDecoderSelfAttention(
            outputs + self.decoderSelfAttention(outputs, outputs, outputs)
        )
        outputs = self.normDecoderAttention(
            outputs + self.decoderAttention(outputs, inputs, inputs, mask=mask)
        )
        outputs = self.normDecoderFF(outputs + self.decoderFF(outputs))

        return outputs


class MultijetAttention(nn.Module):
    def __init__(
        self,
        dim_j,
        embedFeatures,
        attentionFeatures,
        nh=1,
        layers=None,
        inputLayers=None,
    ):
        super().__init__()
        self.dim_j = dim_j
        self.ne = embedFeatures
        self.na = attentionFeatures
        self.nh = nh
        self.jetEmbed = conv1d(5, 5, 1, name="other jet embed", batchNorm=False)
        self.jetConv1 = conv1d(5, 5, 1, name="other jet convolution 1", batchNorm=True)
        # self.jetConv2 = conv1d(5, 5, 1, name='other jet convolution 2', batchNorm=False)

        layers.addLayer(self.jetEmbed)
        layers.addLayer(self.jetConv1, [self.jetEmbed.index])
        inputLayers.append(self.jetConv1.index)

        self.attention = MultiHeadAttention(
            dim_query=self.ne,
            dim_key=5,
            dim_value=5,
            dim_attention=8,
            heads=2,
            dim_valueAttention=10,
            dim_out=self.ne,
            groups_query=1,
            groups_key=1,
            groups_value=1,
            selfAttention=False,
            outBias=False,
            layers=layers,
            inputLayers=inputLayers,
            bothAttention=False,
            iterations=2,
        )
        self.outputLayer = self.attention.o_linear.index

    def forward(self, q, kv, mask, q0=None, qLinear=0, debug=False):
        if debug:
            print("q\n", q[0])
            print("kv\n", kv[0])
            print("mask\n", mask[0])

        kv = self.jetEmbed(kv, mask)
        kv0 = kv.clone()
        kv = NonLU(kv, self.training)

        kv = self.jetConv1(kv, mask)
        kv = kv + kv0
        kv = NonLU(kv, self.training)

        # kv = self.jetConv2(kv, mask)
        # kv = kv+kv0
        # kv = NonLU(kv, self.training)

        q, q0 = self.attention(
            q, kv, kv, q0=q0, mask=mask, debug=debug, selfAttention=False
        )

        return q, q0


class DijetReinforceLayer(nn.Module):
    def __init__(self, dim_d, batchNorm=False, nAveraging=4):
        super().__init__()
        self.dim_d = dim_d
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|
        self.conv = conv1d(
            self.dim_d,
            self.dim_d,
            kernel_size=3,
            stride=3,
            name="dijet reinforce convolution",
            batchNorm=batchNorm,
            nAveraging=nAveraging,
        )

    def forward(self, j, d):
        d = torch.cat(
            (
                j[:, :, 0:2],
                d[:, :, 0:1],
                j[:, :, 2:4],
                d[:, :, 1:2],
                j[:, :, 4:6],
                d[:, :, 2:3],
                j[:, :, 6:8],
                d[:, :, 3:4],
                j[:, :, 8:10],
                d[:, :, 4:5],
                j[:, :, 10:12],
                d[:, :, 5:6],
            ),
            dim=2,
        )
        d = self.conv(d)
        return d


class DijetResNetBlock(nn.Module):
    def __init__(
        self,
        dim_d,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        layers: layerOrganizer = None,
        inputLayers=None,
        nAveraging=4,
    ):
        super().__init__()
        self.dim_d = dim_d
        self.device = device

        self.reinforce1 = DijetReinforceLayer(
            self.dim_d, batchNorm=True, nAveraging=nAveraging
        )
        self.convJ = conv1d(
            self.dim_d,
            self.dim_d,
            1,
            name="jet convolution",
            batchNorm=True,
            nAveraging=nAveraging,
        )
        self.reinforce2 = DijetReinforceLayer(
            self.dim_d, batchNorm=False, nAveraging=nAveraging
        )

        layers.addLayer(self.reinforce1.conv, inputLayers)
        layers.addLayer(self.convJ, [inputLayers[0]])
        layers.addLayer(
            self.reinforce2.conv, [self.convJ.index, self.reinforce1.conv.index]
        )

        self.outputLayer = self.reinforce2.conv.index

        self.MultijetAttention = None

    def forward(self, j: torch.Tensor, d: torch.Tensor):

        d0 = d.clone()
        d = self.reinforce1(j, d)
        d = NonLU(d, self.training)
        d = d + d0

        j0 = j.clone()
        j = self.convJ(j)
        j = NonLU(j, self.training)
        j = j + j0

        d0 = d.clone()
        d = self.reinforce2(j, d)
        d = NonLU(d, self.training)
        d = d + d0

        return d


class QuadjetReinforceLayer(nn.Module):
    def __init__(self, dim_q, batchNorm=False, nAveraging=4):
        super().__init__()
        self.dim_q = dim_q

        # make fixed convolution to compute average of dijet pixel pairs (symmetric bilinear)
        self.sym = nn.Conv1d(
            self.dim_q, self.dim_q, 2, stride=2, bias=False, groups=self.dim_q
        )
        self.sym.weight.data.fill_(0.5)
        self.sym.weight.requires_grad = False

        # make fixed convolution to compute difference of dijet pixel pairs (antisymmetric bilinear)
        self.antisym = nn.Conv1d(
            self.dim_q, self.dim_q, 2, stride=2, bias=False, groups=self.dim_q
        )
        self.antisym.weight.data.fill_(0.5)
        self.antisym.weight.data[:, :, 1] *= -1
        self.antisym.weight.requires_grad = False

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|
        self.conv = conv1d(
            self.dim_q,
            self.dim_q,
            3,
            stride=3,
            name="quadjet reinforce convolution",
            batchNorm=batchNorm,
            nAveraging=nAveraging,
        )

    def forward(self, d: torch.Tensor, q: torch.Tensor):
        d_sym = self.sym(d)
        d_antisym = torch.abs(self.antisym(d))
        q = torch.cat(
            (
                d_sym[:, :, 0:1],
                d_antisym[:, :, 0:1],
                q[:, :, 0:1],
                d_sym[:, :, 1:2],
                d_antisym[:, :, 1:2],
                q[:, :, 1:2],
                d_sym[:, :, 2:3],
                d_antisym[:, :, 2:3],
                q[:, :, 2:3],
            ),
            2,
        )
        q = self.conv(q)
        return q


class QuadjetResNetBlock(nn.Module):
    def __init__(
        self,
        dim_q,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        layers: layerOrganizer = None,
        inputLayers=None,
        nAveraging=4,
    ):
        super().__init__()
        self.dim_q = dim_q
        self.device = device

        self.reinforce1 = QuadjetReinforceLayer(
            self.dim_q, batchNorm=True, nAveraging=nAveraging
        )
        self.convD = conv1d(
            self.dim_q,
            self.dim_q,
            1,
            name="dijet convolution",
            batchNorm=True,
            nAveraging=nAveraging,
        )
        self.reinforce2 = QuadjetReinforceLayer(
            self.dim_q, batchNorm=False, nAveraging=nAveraging
        )

        layers.addLayer(self.reinforce1.conv, inputLayers)
        layers.addLayer(self.convD, [inputLayers[0]])
        layers.addLayer(
            self.reinforce2.conv, [self.convD.index, self.reinforce1.conv.index]
        )

    def forward(self, d: torch.Tensor, q: torch.Tensor):

        q0 = q.clone()
        q = self.reinforce1(d, q)
        q = NonLU(q, self.training)
        q = q + q0

        d0 = d.clone()
        d = self.convD(d)
        d = NonLU(d, self.training)
        d = d + d0

        q0 = q.clone()
        q = self.reinforce2(d, q)
        q = NonLU(q, self.training)
        q = q + q0

        return q

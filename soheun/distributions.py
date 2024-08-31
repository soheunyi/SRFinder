import numpy as np
import torch
import torch.distributions as dist

# import scipy.stats as stats
from abc import ABC
from typing import List


# Domain: [0, 1] x [0, 1]


class Distribution(ABC):
    def __init__(self):
        pass

    def pdf(self, x: torch.Tensor):
        pass

    def sample(self, n: int):
        pass

    @property
    def dim(self):
        pass


class TruncatedDistribution(Distribution):
    """
    Truncate the distribution to have the domain [0, 1]^d
    """

    def __init__(self, distrib: Distribution):
        self.distrib = distrib

        # Integrate pdf over domain D = [0, 1]^d via Monte Carlo
        # Estimates p = P(x \in D)
        # Max variance = p * (1 - p) / n_samples <= 0.25 / n_samples
        n_samples = 1_000_000
        samples = self.distrib.sample(n_samples)
        # check how many samples are in [0, 1]^d
        self.pdf_const = torch.mean(
            (samples >= 0).all(axis=-1) & (samples <= 1).all(axis=-1), dtype=torch.float
        )

        # if p is too small, print waring
        if self.pdf_const < 0.01:
            print("Warning: pdf_const is too small")

    def pdf(self, x):
        return self.distrib.pdf(x) / self.pdf_const

    def sample(self, n):
        n_left = n
        samples = torch.tensor([])
        while n_left > 0:
            new_samples = self.distrib.sample(n_left)
            new_samples = new_samples[
                (new_samples >= 0).all(axis=-1) & (new_samples <= 1).all(axis=-1)
            ]
            samples = (
                new_samples if samples.size == 0 else torch.cat((samples, new_samples))
            )
            n_left = n - samples.shape[0]

        return samples

    @property
    def dim(self):
        return self.distrib.dim


class Exponential2D(Distribution):
    def __init__(self, loc_x, scale_x, loc_y, scale_y):
        assert loc_x <= 0 and loc_y <= 0
        self.loc_x = loc_x
        self.scale_x = scale_x
        self.loc_y = loc_y
        self.scale_y = scale_y

    def pdf_base(self, loc, scale):
        def pdf_base_inner(x: torch.Tensor):
            return torch.exp(-(x - loc) / scale) / scale

        return pdf_base_inner

    def pdf(self, x):
        return self.pdf_base(self.loc_x, self.scale_x)(x[..., 0]) * self.pdf_base(
            self.loc_y, self.scale_y
        )(x[..., 1])

    def sample(self, n):
        x = dist.Exponential(1 / self.scale_x).sample((n,)) + self.loc_x
        y = dist.Exponential(1 / self.scale_y).sample((n,)) + self.loc_y

        return torch.stack((x, y), dim=1)

    @property
    def dim(self):
        return 2


class Exponential(Distribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        assert len(loc.shape) == 1
        assert loc.shape == scale.shape
        assert (loc <= 0).all() and (scale > 0).all()

        self.loc = loc
        self.scale = scale
        self.d = loc.shape[0]

    def pdf(self, x: torch.Tensor):
        return torch.prod(torch.exp(-(x - self.loc) / self.scale) / self.scale, axis=-1)

    def sample(self, n):
        return self.scale * dist.Exponential(1).sample((n, self.dim)) + self.loc

    @property
    def dim(self):
        return self.d


class Gaussian(Distribution):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        self.mean = mean
        self.cov = cov
        self.mvn = dist.MultivariateNormal(self.mean, self.cov)

    def pdf(self, x: torch.Tensor):
        return torch.exp(self.mvn.log_prob(x))

    def sample(self, n):
        return self.mvn.sample((n,))

    @property
    def dim(self):
        return self.mean.shape[0]


class Uniform(Distribution):
    def __init__(self, dim: int):
        self.d = dim

    def pdf(self, x: torch.Tensor):
        assert x.shape[-1] == self.dim
        return torch.ones(x.shape[:-1])

    def sample(self, n: int):
        return dist.Uniform(0, 1).sample((n, self.dim))

    @property
    def dim(self):
        return self.d


class DistributionMixture(Distribution):
    def __init__(self, distributions: List[Distribution], weights: List[float]):
        assert len(set(distrib.dim for distrib in distributions)) == 1
        self.distributions = distributions
        self.weights = torch.tensor(weights)
        self.weights = self.weights / torch.sum(self.weights)
        self.n_distributions = len(distributions)

    def pdf(self, x):
        return torch.sum(
            self.weights * torch.stack([d.pdf(x) for d in self.distributions], axis=-1),
            axis=-1,
        )

    def sample(self, n):
        n_samples_list = torch.bincount(
            torch.multinomial(self.weights, n, replacement=True),
            minlength=self.n_distributions,
        )

        samples = torch.tensor([])
        for i, n_samples in enumerate(n_samples_list):
            if n_samples > 0:
                new_samples = self.distributions[i].sample(n_samples)
                samples = new_samples if i == 0 else torch.cat((samples, new_samples))

        # shuffle samples
        samples = samples[torch.randperm(samples.shape[0])]

        return samples

    @property
    def dim(self):
        return self.distributions[0].dim


class CartesianProductDistribution(Distribution):
    def __init__(self, distributions: List[Distribution]):
        self.distributions = distributions
        self.d = sum(d.dim for d in distributions)

    def pdf(self, x):
        pdf_vals = []
        dim_pointer = 0
        for i, dist in enumerate(self.distributions):
            pdf_val = self.distributions[i].pdf(
                x[..., dim_pointer : dim_pointer + dist.dim]
            )
            pdf_vals.append(pdf_val)
            dim_pointer += dist.dim

        return torch.prod(torch.stack(pdf_vals, axis=-1), axis=-1)

    def sample(self, n):
        return torch.cat([d.sample(n) for d in self.distributions], axis=-1)

    @property
    def dim(self):
        return self.d


def gaussian_bump(mean, cov):
    return TruncatedDistribution(Gaussian(mean, cov))


def exponential_background_2d(loc_x, scale_x, loc_y, scale_y):
    return TruncatedDistribution(Exponential2D(loc_x, scale_x, loc_y, scale_y))


def exponential_background(loc: torch.Tensor, scale: torch.Tensor):
    return TruncatedDistribution(Exponential(loc, scale))


def uniform_background(dim):
    return Uniform(dim)

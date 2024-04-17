# implement various likelihood functions


from abc import ABC
import torch
import torch.distributions as dist
from constants import VIRT_ZERO, VIRT_INF


class Likelihood(ABC):
    def __init__(self):
        pass

    def log_likelihood(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def n_params(self) -> int:
        pass

    @property
    def params(self) -> torch.Tensor:
        pass

    @params.setter
    def params(self, value):
        pass

    def sample(self, shape: torch.Size, params: torch.Tensor) -> torch.Tensor:
        pass


class TruncatedGaussianLikelihood(Likelihood):
    def __init__(self, lb=-torch.inf, ub=torch.inf):
        assert lb < ub, "Lower bound should be less than upper bound"
        self.lb = lb
        self.ub = ub

    def cdf_bounds(self, params: torch.Tensor):
        self.params = params
        try:
            gaussian_dist = dist.Normal(self.mu, torch.exp(self.logvar / 2))
        except:
            print(self.mu, self.logvar)
            print(self.mu.shape, self.logvar.shape)
            print(self.mu.device, self.logvar.device)
            print(self.mu.dtype, self.logvar.dtype)
            raise
        cdf_lower = (
            torch.tensor([0.0]).to(self.mu.device)
            if self.lb == -torch.inf
            else gaussian_dist.cdf(torch.tensor(self.lb))
        )
        cdf_upper = (
            torch.tensor([1.0]).to(self.mu.device)
            if self.ub == torch.inf
            else gaussian_dist.cdf(torch.tensor(self.ub))
        )
        return cdf_lower, cdf_upper

    def log_likelihood(self, x: torch.Tensor, params: torch.Tensor):
        cdf_lower, cdf_upper = self.cdf_bounds(params)
        gaussian_dist = dist.Normal(self.mu, torch.exp(self.logvar / 2))
        normalizer = torch.clip(cdf_upper - cdf_lower, min=VIRT_ZERO)
        log_probs = gaussian_dist.log_prob(x)
        # clip log_probs where x is outside bounds
        log_probs = torch.where(
            (x < self.lb) | (x > self.ub),
            -torch.log(torch.tensor(VIRT_INF)).to(x.device),
            log_probs,
        )

        if torch.any(torch.isnan(log_probs)):
            print(
                self.mu[torch.isnan(log_probs)],
                self.logvar[torch.isnan(log_probs)],
                self.lb,
                self.ub,
                cdf_lower[torch.isnan(log_probs)],
                cdf_upper[torch.isnan(log_probs)],
            )
            print(
                self.mu[torch.isnan(log_probs)].shape,
                self.logvar[torch.isnan(log_probs)].shape,
                self.lb,
                self.ub,
                cdf_lower[torch.isnan(log_probs)].shape,
                cdf_upper[torch.isnan(log_probs)].shape,
            )

        return log_probs - torch.log(normalizer.to(x.device))

    @property
    def n_params(self):
        return 2

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value: torch.Tensor):
        assert (
            value.shape[-1] == 2
        ), "TruncatedGaussianLikelihood takes two parameters: shape[-1] should be 2"
        self._params = value
        self.mu = value[..., 0]
        self.logvar = value[..., 1]

    def sample(self, shape: torch.Size, params: torch.Tensor):
        self.params = params
        std_normal_dist = dist.Normal(
            torch.zeros_like(self.mu), torch.ones_like(self.logvar)
        )
        # sample via inverse CDF
        # sample (shape, mu.shape) from uniform distribution
        u = torch.rand(shape + self.mu.shape).to(self.mu.device)

        cdf_lower, cdf_upper = self.cdf_bounds(params)
        normalizer = cdf_upper - cdf_lower
        cdf = cdf_lower + u * normalizer
        samples = self.mu + torch.exp(self.logvar / 2) * std_normal_dist.icdf(cdf)

        # resample -inf or inf values
        isinf = torch.isinf(samples)
        inf_out_params = torch.any(isinf.view(-1, *self.mu.shape), dim=0)
        # try accept-reject for inf_out_params
        if torch.any(inf_out_params):
            print("Hello")
            print(
                self.mu[inf_out_params],
                self.logvar[inf_out_params],
                self.lb,
                self.ub,
                cdf_lower[inf_out_params],
                cdf_upper[inf_out_params],
            )
            gaussian_dist = dist.Normal(
                self.mu[inf_out_params], torch.exp(self.logvar[inf_out_params] / 2)
            )
            new_samples = gaussian_dist.sample(shape)
            accepts = (new_samples >= self.lb) & (new_samples <= self.ub)
            # resample until all samples are within bounds
            while not torch.all(accepts):
                new_samples[~accepts] = gaussian_dist.sample(shape)[~accepts]
                accepts = (new_samples >= self.lb) & (new_samples <= self.ub)

            samples[..., inf_out_params] = new_samples

        return samples


class ConstantLikelihood(Likelihood):
    def __init__(self, value=0.0):
        self.value = value

    def log_likelihood(self, x, params):
        self.params = params  # should be empty
        return torch.zeros_like(x).to(x.device)

    @property
    def n_params(self):
        return 0

    @property
    def params(self):
        return torch.tensor([])

    @params.setter
    def params(self, value):
        assert value.shape[-1] == 0, "ConstantLikelihood takes no parameters"

    def sample(self, shape):
        return torch.zeros(shape)

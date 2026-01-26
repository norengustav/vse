# Author: (c) 2025 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)


from collections.abc import Callable
from dataclasses import dataclass

import torch
from typing_extensions import override

import src.ml.math
import src.ml.rnn
import src.ml.rnn_vi


@dataclass
class VSEParameters:
    """Helper class to store VSE parameters and hyperparameters (for training)"""

    # Number of hidden states in the process
    num_states: int
    # Number of dimensions of the measurements i.e. num observable dimensions
    num_obs: int
    # Length of each measured trajectory
    trajectory_length: int
    # Number of samples used in reparametrization trick
    num_reparametrization_samples: int


class VSE(torch.nn.Module):
    def __init__(
        self,
        parameters: VSEParameters,
        rnn_parameters: src.ml.rnn.RNNParameters,
        h_fn: Callable[[torch.Tensor], torch.Tensor],
        measurement_cov_cw: torch.Tensor,
    ) -> None:
        super().__init__()

        self.vse_parameters: VSEParameters = parameters
        self.device: torch.device = rnn_parameters.device
        # RNN for prior distribution, returns mean and cov, contains theta parameters
        self.rnn_prior: src.ml.rnn.RNNModel = src.ml.rnn.RNNModel(rnn_parameters).to(
            self.device
        )
        # RNN for VI distribution (q), returns mean and cov, contains psi parameters
        self.rnn_vi_posterior: src.ml.rnn_vi.RNNModel = src.ml.rnn_vi.RNNModel(
            rnn_parameters
        ).to(self.device)

        self.learning_rate: float = rnn_parameters.learning_rate
        self.epochs: int = rnn_parameters.epochs

        self.h_fn = h_fn
        self.measurement_cov_cw = measurement_cov_cw.to(self.device)

    def push_model(self, device: torch.device):
        self.device = device
        _ = self.to(device=device)
        _ = self.measurement_cov_cw.to(device)
        # Evil code, if this is not done it causes mismatched devices error
        # far away from here
        self.rnn_prior.parameters.device = device
        self.rnn_vi_posterior.parameters.device = device

    def prior_predictions(self, y_batch: torch.Tensor):
        prior_means, prior_vars = self.rnn_prior.forward(y_batch)
        prior_covs = torch.diag_embed(prior_vars)
        return prior_means, prior_covs

    def compute_predictions(self, y_batch: torch.Tensor):
        vi_posterior_means, vi_posterior_vars = self.rnn_vi_posterior.forward(y_batch)
        vi_posterior_covs = torch.diag_embed(vi_posterior_vars)
        return vi_posterior_means, vi_posterior_covs

    @override
    def forward(self, y_batch: torch.Tensor) -> torch.Tensor:
        """Returns batch loss"""
        batch_size = len(y_batch)
        # Reparametrization trick part
        vi_posterior_means, vi_posterior_vars = self.rnn_vi_posterior.forward(y_batch)
        vi_posterior_covs = torch.diag_embed(vi_posterior_vars)
        vi_posterior_cholesky_of_covs = torch.linalg.cholesky(vi_posterior_covs)
        # Reparametrization samples
        epsilon = torch.normal(
            mean=0,
            std=1,
            size=(
                self.vse_parameters.num_reparametrization_samples,
                batch_size,
                self.vse_parameters.trajectory_length,
                self.vse_parameters.num_states,
            ),
        ).to(self.device)
        vi_posterior_samples = vi_posterior_means + torch.einsum(
            "btij,lbtj->lbti", vi_posterior_cholesky_of_covs, epsilon
        )
        h_x = self.h_fn(vi_posterior_samples)

        gauss_log_pdf = src.ml.math.gaussian_log_pdf(
            data_x=y_batch,
            mean=h_x,
            cov=self.measurement_cov_cw,
        ).mean(0)

        # Kulback-Leibler part
        prior_means, prior_vars = self.rnn_prior.forward(y_batch)
        prior_covs = torch.diag_embed(prior_vars)

        kl_det_part = torch.logdet(prior_covs) - torch.logdet(vi_posterior_covs)
        m_minus_m = vi_posterior_means - prior_means
        inverse_prior_covs = torch.inverse(prior_covs)
        kl_norm_part_right = torch.einsum(
            "...ij,...j->...i",
            inverse_prior_covs,
            m_minus_m,
        )
        kl_norm_part = torch.einsum("...i,...i", m_minus_m, kl_norm_part_right)

        kl_trace_part = torch.einsum(
            "...ii", torch.matmul(inverse_prior_covs, vi_posterior_covs)
        )

        kl_part = -0.5 * (
            kl_det_part - self.vse_parameters.num_states + kl_norm_part + kl_trace_part
        )

        # Full loss
        loss = -(gauss_log_pdf.mean() + kl_part.mean())

        return loss

# Author: (c) 2025 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)

import math

import torch


def gaussian_log_pdf(
    data_x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor
) -> torch.Tensor:
    """
    data_x.shape = (*, k)
    mean.shape = (*, k)
    cov.shape = (*, k, k)
    return shape = (*)
    """
    k_dim = mean.shape[-1]
    pi_part = k_dim * math.log(math.pi * 2)
    det_cov_part = torch.logdet(cov)

    x_minus_mean = data_x - mean
    exp_right_part = torch.einsum("...ij,...j->...i", torch.inverse(cov), x_minus_mean)
    exp_part = torch.einsum("...i,...i", x_minus_mean, exp_right_part)

    out = -0.5 * (pi_part + det_cov_part + exp_part)
    return out

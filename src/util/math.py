# Modified by: (c) 2024 Gustav Norén
# Original author: (c) 2023 Anubhab Ghosh
# This code is licensed under MIT licence (see LICENSE file for details)


import numpy as np
import torch


def decibel_to_linear(x_var):
    return 10 ** (x_var / 10)


def linear_to_decibel(x_var):
    assert x_var != 0, "x_var is zero"
    return 10 * np.log10(x_var)


def compute_inverse(data_x):
    u, s, vh = torch.svd(data_x)
    return vh @ (torch.diag(1 / s.reshape((-1,))) @ u.T)


def create_diag(x_var):
    return torch.diag_embed(x_var)


def normalize(x_var):
    return (x_var - x_var.mean(0)) / x_var.std(0)

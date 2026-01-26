# Modified by: (c) 2025 Gustav Norén
# Original author: (c) 2023 Anubhab Ghosh
# This code is licensed under MIT licence (see LICENSE file for details)

import itertools

import numpy as np
import torch

IMG_SIZE = 8
X1_RANGE_START = -30
X1_RANGE_END = 30
X2_RANGE_START = -40
X2_RANGE_END = 40


def _camera_model_core_numpy(x):
    grid_points = np.vstack(
        [
            np.array([i, j])
            for i, j in itertools.product(
                np.linspace(X1_RANGE_START, X1_RANGE_END, IMG_SIZE),
                np.linspace(X2_RANGE_START, X2_RANGE_END, IMG_SIZE),
            )
        ]
    )
    d_arr = np.linalg.norm(grid_points - x[:2], axis=1) ** 2
    h_x_arr = 10.0 * np.exp(-0.5 * np.divide(d_arr, 1e-12 + np.abs(np.array(x[2]))))
    return h_x_arr


def _camera_model_core_torch(x):
    grid_points = torch.stack(
        [
            torch.Tensor([i, j])
            for i, j in itertools.product(
                torch.linspace(X1_RANGE_START, X1_RANGE_END, IMG_SIZE),
                torch.linspace(X2_RANGE_START, X2_RANGE_END, IMG_SIZE),
            )
        ]
    ).to(x.device)
    x_new = x.clone()
    if len(x.shape) == 3:
        grid_points_rep = torch.repeat_interleave(
            torch.repeat_interleave(
                grid_points.unsqueeze(0), x.shape[1], dim=0
            ).unsqueeze(0),
            x.shape[0],
            dim=0,
        )
        x_rep = torch.repeat_interleave(x_new.unsqueeze(-2), IMG_SIZE**2, dim=-2)
        d_arr = (
            torch.norm(
                torch.div(
                    grid_points_rep - x_rep[..., :2],
                    1e-12 + (torch.abs(x_rep[..., 2]) ** (0.5)).unsqueeze(-1),
                ),
                dim=-1,
            )
            ** 2
        )
    elif len(x.shape) == 4:
        grid_points_rep = torch.repeat_interleave(
            torch.repeat_interleave(
                torch.repeat_interleave(
                    grid_points.unsqueeze(0), x.shape[2], dim=0
                ).unsqueeze(0),
                x.shape[1],
                dim=0,
            ).unsqueeze(0),
            x.shape[0],
            dim=0,
        )
        x_rep = torch.repeat_interleave(x_new.unsqueeze(-2), IMG_SIZE**2, dim=-2)
        d_arr = (
            torch.norm(
                torch.div(
                    grid_points_rep - x_rep[..., :2],
                    1e-12 + (torch.abs(x_rep[..., 2]) ** (0.5)).unsqueeze(-1),
                ),
                dim=-1,
            )
            ** 2
        )
    elif len(x.shape) == 1:
        d_arr = (
            torch.norm(
                torch.div(
                    grid_points - x_new[:2],
                    1e-12 + (torch.abs(x[2]) ** (0.5)).unsqueeze(-1),
                ),
                dim=-1,
            )
            ** 2
        )
    h_x_arr = 10.0 * torch.exp(-0.5 * d_arr)
    return h_x_arr


def camera_model(x):
    if type(x).__module__ == np.__name__:
        h_x_arr = _camera_model_core_numpy(x)
    elif type(x).__module__ == torch.__name__:
        h_x_arr = _camera_model_core_torch(x)
    return h_x_arr

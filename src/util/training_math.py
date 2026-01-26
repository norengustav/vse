# Modified by: (c) 2024 Gustav Norén
# Original author: (c) 2023 Anubhab Ghosh
# This code is licensed under MIT licence (see LICENSE file for details)

import torch


def count_params(model):
    """
    Counts two types of parameters:

    - Total no. of parameters in the model (including trainable parameters)
    - Number of trainable parameters (i.e. parameters whose gradients will be computed)

    """
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad is True
    )
    return total_num_params, total_num_trainable_params


def mse_loss(x, xhat):
    loss = torch.nn.MSELoss(reduction="none")
    return loss(xhat, x)


def mse_loss_decibel(x, xhat):
    noise_p = mse_loss(xhat, x).mean((1, 2))
    return 10 * torch.log10(noise_p).mean()


def nmse_loss(x, xhat):
    # loss = nn.MSELoss(reduction='mean')
    # noise_p = loss(xhat, x)
    # signal_p = loss(x, torch.zeros_like(x))
    return mse_loss_decibel(xhat, x) - mse_loss_decibel(x, torch.zeros_like(x))
    # return 10*torch.log10(noise_p / signal_p)


def nmse_loss_std(x, xhat):
    loss = torch.nn.MSELoss(reduction="none")
    noise_p = loss(xhat, x)
    signal_p = loss(x, torch.zeros_like(x))
    return (
        10 * torch.log10(noise_p.mean((1, 2))) - 10 * torch.log10(signal_p.mean((1, 2)))
    ).std()


def mse_loss_decibel_std(x, xhat):
    loss = torch.nn.MSELoss(reduction="none")
    noise_p = loss(xhat, x).mean((1, 2))
    return (10 * torch.log10(noise_p)).std()

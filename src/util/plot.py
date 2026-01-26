# Author: (c) 2025 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)

import matplotlib.pyplot as plt
import numpy as np
from sympy.printing.pretty.stringpict import line_width
import torch


def plot_nmse_curve(smnr_db_list, nmse_list, nmse_std_list, savefig_name=None):
    # Plotting the NMSE Curve
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    plt.figure()
    plt.title("NMSE Curve")
    plt.errorbar(
        smnr_db_list,
        nmse_list,
        fmt="r-.",
        yerr=nmse_std_list,
        linewidth=2.0,
        label="veda",
    )
    plt.xlabel("SMNR (in dB)")
    plt.ylabel("NMSE (in dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if savefig_name is not None:
        plt.savefig(savefig_name)
    # plt.show()
    plt.close()


def plot_one_3d_trajectory(
    x_trajectory,
    fmt="k-",
    cross_indices=None,
    label=None,
    title=None,
    savefig_name=None,
):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=-0.15, right=1.05, bottom=0.05, top=1.06)
    ax.plot(
        x_trajectory[:, 0],
        x_trajectory[:, 1],
        x_trajectory[:, 2],
        fmt,
        label=label,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    if cross_indices is not None:
        for i in cross_indices:
            ax.plot(
                x_trajectory[i, 0],
                x_trajectory[i, 1],
                x_trajectory[i, 2],
                "xr",
                markeredgewidth=2,
                markersize=12,
            )
    if title is not None:
        plt.title(title, y=1.0, pad=-21)
    if label is not None:
        plt.legend()

    if savefig_name is not None:
        plt.savefig(savefig_name)
    plt.close()


def plot_two_3d_trajectory(
    x_1,
    x_2,
    fmt_1="k-",
    fmt_2="r--",
    label_1=None,
    label_2=None,
    title=None,
    savefig_name=None,
):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    fig = plt.figure(figsize=(4.5, 4))
    plt.subplots_adjust(left=-0.15, right=1.05, bottom=0.05, top=1.0)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        x_1[:, 0],
        x_1[:, 1],
        x_1[:, 2],
        fmt_1,
        label=label_1,
    )
    ax.plot(
        x_2[:, 0],
        x_2[:, 1],
        x_2[:, 2],
        fmt_2,
        label=label_2,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    if title is not None:
        plt.title(title)
    if label_1 is not None or label_2 is not None:
        plt.legend()

    if savefig_name is not None:
        plt.savefig(savefig_name)
    plt.close()


def plot_x_and_est_one_axis_with_confidence(
    x_true_one_axis,
    x_estimate_one_axis,
    x_estimate_std,
    fmt_x="k-",
    fmt_est="r--",
    facecolor="red",
    alpha=0.4,
    label_x="$\\mathbf{x}^{true}$",
    label_est="$\\hat{\\mathbf{x}}^{vidanse}$",
    title=None,
    savefig_name=None,
    sigma=1.0,
):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    plt.figure(layout="constrained", figsize=(6, 3))
    plt.grid(visible=True)
    plt.plot(x_true_one_axis, fmt_x, label=label_x, linewidth=2)
    plt.plot(x_estimate_one_axis, fmt_est, label=label_est, linewidth=2)
    plt.fill_between(
        np.arange(len(x_estimate_one_axis)),
        x_estimate_one_axis - sigma * x_estimate_std,
        x_estimate_one_axis + sigma * x_estimate_std,
        facecolor="red",
        alpha=alpha,
    )
    # plt.set_xlabel("$t$")
    if title is not None:
        plt.title(title)
    if label_x is not None or label_est is not None:
        plt.legend()
    if savefig_name is not None:
        plt.savefig(savefig_name)
    plt.close()


def plot_all_axes_two_trajectories(
    x_1,
    x_2,
    fmt_1="k-",
    fmt_2="r--",
    label_1=None,
    label_2=None,
    title=None,
    savefig_name=None,
):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    # plt.figure(figsize=(8, 5))
    # plt.grid(visible=True)
    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.14, top=0.99, hspace=0.3)
    # plt.tight_layout()
    plt.subplot(311)
    plt.plot(x_1[:, 0], fmt_1, label=label_1)
    plt.plot(x_2[:, 0], fmt_2, label=label_2)
    plt.ylabel("$x_1$", rotation="horizontal")
    if label_1 is not None or label_2 is not None:
        plt.legend(loc="upper right")
    plt.subplot(312)
    plt.plot(x_1[:, 1], fmt_1, label=label_1)
    plt.plot(x_2[:, 1], fmt_2, label=label_2)
    plt.ylabel("$x_2$", rotation="horizontal")
    plt.subplot(313)
    plt.plot(x_1[:, 2], fmt_1, label=label_1)
    plt.plot(x_2[:, 2], fmt_2, label=label_2)
    plt.ylabel("$x_3$", rotation="horizontal")
    plt.xlabel("$t$")
    if title is not None:
        plt.title(title)
    if savefig_name is not None:
        plt.savefig(savefig_name)
    plt.close()


def plot_camera_snapshot(
    y_data_i,
    savefig_name,
    title=None,
    colorbar=False,
):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    plt.figure(layout="constrained")
    if title is not None:
        plt.title(title)
    plt.axis("off")
    if colorbar:
        plt.imshow(y_data_i, cmap="gray", vmin=-2.0, vmax=4.0)
        plt.colorbar()
    else:
        plt.imshow(y_data_i, cmap="gray")
    if savefig_name is not None:
        plt.savefig(savefig_name, bbox_inches="tight")
    plt.close()

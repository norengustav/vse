# Author: (c) 2025 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)
import glob
import pickle as pkl
from timeit import default_timer as timer

import torch

import src.data.util
import src.util.plot
import src.util.training_math
from src.data.basic_data_parameters import BasicDataParameters


def test_one_model(outpath: str, model_dir: str, basic_data: BasicDataParameters):
    basic_data_str = src.data.util.basic_data_to_str(basic_data)
    model_files = glob.glob(f"model_{basic_data_str}*", root_dir=model_dir)

    data_dir = outpath + "/data"
    datafile = src.data.util.create_full_data_path(
        basic_data, dataset_basepath=data_dir, test=True
    )
    with open(datafile, "rb") as datafile_handel:
        test_data_dict = pkl.load(datafile_handel)
    test_data_x = torch.as_tensor(test_data_dict["data_x"], dtype=torch.float32)
    test_data_y = torch.as_tensor(test_data_dict["data_y"], dtype=torch.float32)

    modelfile = model_dir + "/" + model_files[0]
    print(modelfile)
    model = torch.load(modelfile, weights_only=False)
    model.eval()
    model.push_model(torch.device("cpu"))

    with torch.no_grad():
        start = timer()
        model_means, model_covs = model.compute_predictions(
            test_data_y.to(model.device)
        )
        end = timer()
    model_means = model_means.to("cpu")
    model_covs = model_covs.to("cpu")

    nmse_model = src.util.training_math.nmse_loss(test_data_x, model_means)
    nmse_model_std = src.util.training_math.nmse_loss_std(test_data_x, model_means)

    print(f"prediction time {end - start} s")
    print(
        f"smnr: {basic_data.smnr_db}, "
        f"nmse_model: {nmse_model}, nmse_model_std: {nmse_model_std}"
    )

    fig_index = 0
    src.util.plot.plot_one_3d_trajectory(
        x_trajectory=test_data_x[fig_index].numpy(),
        fmt="k-",
        label=None,
        # title=f"$\\mathbf{{x}}^{{true}}$, SMNR {basic_data.smnr_db}",
        savefig_name=f"{outpath}/x_smnr_{basic_data.smnr_db}.pdf",
    )
    src.util.plot.plot_one_3d_trajectory(
        x_trajectory=model_means[fig_index].numpy(),
        fmt="r-",
        label=None,
        # title=f"$\\hat{{\\mathbf{{x}}}}^{{vse}}$, SMNR {basic_data.smnr_db}",
        savefig_name=f"{outpath}/xhat_smnr_{basic_data.smnr_db}.pdf",
    )
    src.util.plot.plot_two_3d_trajectory(
        x_1=test_data_x[fig_index].numpy(),
        x_2=model_means[fig_index].numpy(),
        fmt_1="k-",
        fmt_2="r--",
        label_1="$\\mathbf{x}^{true}$",
        label_2="$\\hat{\\mathbf{x}}^{vse}$",
        # title=f"Estimate, SMNR {basic_data.smnr_db}",
        savefig_name=f"{outpath}/x_xhat_smnr_{basic_data.smnr_db}.pdf",
    )
    src.util.plot.plot_x_and_est_one_axis_with_confidence(
        x_true_one_axis=test_data_x[fig_index, :, 0].numpy(),
        x_estimate_one_axis=model_means[fig_index, 0:100, 0].numpy(),
        x_estimate_std=model_covs[fig_index, :, 0, 0].numpy(),
        savefig_name=f"{outpath}/state_w_lims_smnr_{basic_data.smnr_db}.pdf",
    )
    src.util.plot.plot_all_axes_two_trajectories(
        x_1=test_data_x[fig_index, :, :].numpy(),
        x_2=model_means[fig_index, :, :].numpy(),
        label_1="$\\mathbf{x}^{true}$",
        label_2="$\\hat{\\mathbf{x}}^{vse}$",
        savefig_name=f"{outpath}/axes_all_{basic_data.smnr_db}.pdf",
    )
    src.util.plot.plot_camera_snapshot(
        y_data_i=torch.reshape(test_data_y[fig_index, 25, :], (8, 8)).numpy(),
        savefig_name=f"{outpath}/y_smnr_{basic_data.smnr_db}.pdf",
        title=f"SMNR {basic_data.smnr_db} dB",
    )

    return (
        nmse_model,
        nmse_model_std,
    )


def test(outpath, basic_data_list):
    model_dir = outpath + "/models"
    smnr_db_list = []
    nmse_model_list = []
    nmse_model_std_list = []
    for basic_data in basic_data_list:
        (
            nmse_model_i,
            nmse_model_std_i,
        ) = test_one_model(outpath=outpath, model_dir=model_dir, basic_data=basic_data)
        nmse_model_list.append(nmse_model_i)
        nmse_model_std_list.append(nmse_model_std_i)
        smnr_db_list.append(basic_data.smnr_db)

    src.util.plot.plot_nmse_curve(
        smnr_db_list,
        nmse_model_list,
        nmse_model_std_list,
        savefig_name=f"{outpath}/nmse_plot.pdf",
    )

# Author: (c) 2025 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)

import pickle as pkl

import torch

import src.data.measurement_functions as mfn
import src.data.ssm_models
import src.data.util
import src.ml.entry
import src.ml.rnn as rnn
import src.ml.vse as vse
import src.util.dataset
from src.data.basic_data_parameters import BasicDataParameters


def train(outpath: str, basic_data: BasicDataParameters):
    print(f"Training model smnr {basic_data.smnr_db}")
    ngpu = 1
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
    )
    print(f"Train Device Used:{device}")

    batch_size = 64

    rnn_parameters = rnn.RNNParameters(
        input_size=basic_data.num_obs,
        output_size=basic_data.num_states,
        model_type="gru",
        hidden_size=80,
        hidden_layers=2,
        learning_rate=1e-3,
        epochs=500,
        dense_layer_size=64,
        tolerance=5e-2,
        device=device,
    )

    veda_parameters = vse.VSEParameters(
        num_states=basic_data.num_states,
        num_obs=basic_data.num_obs,
        trajectory_length=basic_data.trajectory_length,
        num_reparametrization_samples=10,
    )

    data_dir = outpath + "/data"
    datafile = src.data.util.create_full_data_path(
        basic_data, dataset_basepath=data_dir
    )
    with open(datafile, "rb") as datafile_handel:
        train_data_dict = pkl.load(datafile_handel)

    model = vse.VSE(
        parameters=veda_parameters,
        rnn_parameters=rnn_parameters,
        # h_fn=h_fn,
        h_fn=mfn.camera_model,
        measurement_cov_cw=torch.from_numpy(
            train_data_dict["data_measurement_cov_cw"][-1, :, :]
        ).to(dtype=torch.float32),
    )

    train_dataset = src.util.dataset.SeriesDataset(z_xy_dict=train_data_dict)

    _, tr_indices, val_indices, test_indices = src.util.dataset.obtain_tr_val_test_idx(
        dataset=train_dataset, supervised=0.0, test=0.1, val=0.16
    )

    train_loader, val_loader, test_loader = src.util.dataset.get_dataloaders(
        dataset=train_dataset,
        batch_size=batch_size,
        tr_indices=tr_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    src.ml.entry.train(
        model=model,
        basic_data=basic_data,
        train_loader=train_loader,
        val_loader=val_loader,
        model_save_dir=outpath + "/models",
        device=device,
    )

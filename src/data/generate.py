# Modified by: (c) 2024 Gustav Norén
# Original author: (c) 2023 Anubhab Ghosh
# This code is licensed under MIT licence (see LICENSE file for details)

import os

import numpy as np

import src.data.ssm_models
import src.data.util
import src.util.files
from src.data.basic_data_parameters import BasicDataParameters


def _ssm_model_picker(ssm_model: str, num_states, num_obs):
    delta_t = 0.02
    decimation_factor = 2

    out_model = -1
    if ssm_model == "LorenzSSM":
        out_model = src.data.ssm_models.LorenzSSM(
            num_states=num_states,
            num_obs=num_obs,
            param_j=5,
            delta=delta_t,
            delta_d=delta_t / decimation_factor,
            decimate=True,
            alpha=0.0,
            mu_e=np.zeros((num_states,)),
            mu_w=np.zeros((num_obs,)),
        )
    else:
        print("Invalid ssm_model")
    return out_model


def generate_ssm_data(model, trajectory_length, sigma_e2_db, smnr_db):
    """This function generates a single pair
    (state trajectory, measurement trajectory) for the given ssm_model
    where the state trajectory has been generated using process noise corresponding to
    `sigma_e2_db` and the measurement trajectory has been generated using measurement noise
    corresponding to `smnr_db`.

    Args:
        model (object): ssm model object
        trajectory_length (int): length of measurement trajectory,
            state trajectory is of length trajectory_length as it has an initial state of the ssm model
        sigma_e2_db (float): process noise in dB scale
        smnr_db (float): measurement noise in dB scale

    Returns:
        X_arr: numpy array of size (trajectory_length, model.n_states)
        Y_arr: numpy array of size (trajectory_length, model.n_obs)
    """

    data_x, data_y, measurement_cov_cw = model.generate_single_sequence(
        trajectory_length=trajectory_length, sigma_e2_db=sigma_e2_db, smnr_db=smnr_db
    )

    return data_x, data_y, measurement_cov_cw


def generate_state_observation_pairs(
    basic_data_parameters: BasicDataParameters,
    ssm_model,
    test: bool = False,
):
    """This function generate several state trajectory and measurement trajectory pairs

    Args:
        basic_data_parameters: BasicDataParameters

    Returns:
        data_dictionary: dictionary containing the ssm model object used for generating data, the generated data,
            lengths of trajectories, etc.
    """
    data_dictionary = {}

    data_x = []
    data_y = []
    data_measurement_cov_cw = []
    ssm_model_dict = {}
    trajectory_length = 0
    num_trajectories = 0
    if test:
        trajectory_length = basic_data_parameters.test_trajectory_length
        num_trajectories = basic_data_parameters.num_test_trajectories
    else:
        trajectory_length = basic_data_parameters.trajectory_length
        num_trajectories = basic_data_parameters.num_trajectories

    model_type = basic_data_parameters.dataset_type
    ssm_model_dict[model_type] = ssm_model

    for _ in range(num_trajectories):
        data_xi, data_yi, data_measurement_cov_cw_i = generate_ssm_data(
            ssm_model,
            trajectory_length,
            basic_data_parameters.sigma_e2_db,
            basic_data_parameters.smnr_db,
        )
        data_x.append(data_xi)
        data_y.append(data_yi)
        data_measurement_cov_cw.append(data_measurement_cov_cw_i)

    data_dictionary["basic_data_parameters"] = basic_data_parameters
    data_dictionary["ssm_model_dict"] = ssm_model_dict
    data_dictionary["data_x"] = np.asarray(data_x)
    data_dictionary["data_y"] = np.asarray(data_y)
    data_dictionary["data_measurement_cov_cw"] = np.asarray(data_measurement_cov_cw)

    return data_dictionary


def create_and_save_dataset(
    basic_data_parameters,
    filename,
    ssm_model,
    test: bool = False,
):
    """Calls most the functions above to generate data and saves it at the desired location given by `filename`.

    Args:
        basic_data_parameters: BasicDataParameters
        filename (str): string describing output filename with full / absolute path.
        parameters_dict (dict): dictionary containing parameters for various SSM models
    """
    data_dictionary = generate_state_observation_pairs(
        basic_data_parameters=basic_data_parameters,
        ssm_model=ssm_model,
        test=test,
    )

    src.util.files.save_dataset(data_dictionary, filename=filename)


def generate_data(
    basic_data_parameters: BasicDataParameters,
    output_path: str,
    generate_test_data=True,
):
    output_path = output_path + "/data"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create the full path for the datafile
    datafilename = src.data.util.create_full_data_path(
        basic_data_parameters=basic_data_parameters,
        dataset_basepath=output_path,
    )

    ssm_model = _ssm_model_picker(
        basic_data_parameters.dataset_type,
        basic_data_parameters.num_states,
        basic_data_parameters.num_obs,
    )

    # If the dataset hasn't been already created, create the dataset
    if not os.path.isfile(datafilename):
        print(f"Creating the data file: {datafilename}")
        create_and_save_dataset(
            basic_data_parameters=basic_data_parameters,
            filename=datafilename,
            ssm_model=ssm_model,
        )
    else:
        print(f"Dataset {datafilename} is already present!")

    # Generate test data equivalently
    if generate_test_data:
        datafilename = src.data.util.create_full_data_path(
            basic_data_parameters=basic_data_parameters,
            dataset_basepath=output_path,
            test=True,
        )
        if not os.path.isfile(datafilename):
            print(f"Creating the data file: {datafilename}")
            create_and_save_dataset(
                basic_data_parameters=basic_data_parameters,
                filename=datafilename,
                ssm_model=ssm_model,
                test=True,
            )
        else:
            print(f"Dataset {datafilename} is already present!")

    print("Done...")


def generate(basic_data_parameters: BasicDataParameters, outpath: str, filename: str):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    ssm_model = _ssm_model_picker(
        basic_data_parameters.dataset_type,
        basic_data_parameters.num_states,
        basic_data_parameters.num_obs,
    )

    # If the dataset hasn't been already created, create the dataset
    if not os.path.isfile(filename):
        print(f"Creating the data file: {filename}")
        create_and_save_dataset(
            basic_data_parameters=basic_data_parameters,
            filename=filename,
            ssm_model=ssm_model,
        )
    else:
        print(f"Dataset {filename} is already present!")


def create_subfolders(path: str, subfolders: list[str]):
    if not os.path.exists(path):
        os.makedirs(path)
    for folder in subfolders:
        os.makedirs(path + "/" + folder)

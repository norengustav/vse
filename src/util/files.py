# Modified by: (c) 2024 Gustav Norén
# Original author: (c) 2023 Anubhab Ghosh
# This code is licensed under MIT licence (see LICENSE file for details)
import json
import os
import pickle as pkl

import numpy as np


def create_splits_file_name(dataset_filename, splits_filename):
    _idx_dset_info = dataset_filename.rfind("m")
    idx_splitfilename = splits_filename.rfind(".pkl")
    splits_filename_modified = splits_filename[:idx_splitfilename] + ".pkl"  # + "_" + dataset_filename[idx_dset_info:]
    return splits_filename_modified


def create_file_paths(params_combination_list, filepath, main_exp_name):
    list_of_logfile_paths = []
    # Creating the logfiles
    for params in params_combination_list:
        exp_folder_name = "trajectories_M{}_P{}_N{}/".format(
            params["num_trajectories"], params["num_realizations"], params["N_seq"]
        )

        # print(os.path.join(log_filepath, main_exp_name, exp_folder_name))
        full_path_exp_folder = os.path.join(filepath, main_exp_name, exp_folder_name)
        list_of_logfile_paths.append(full_path_exp_folder)
        os.makedirs(full_path_exp_folder, exist_ok=True)

    return list_of_logfile_paths


def get_list_of_config_files(
    model_type, options, dataset_mode="pfixed", params_combination_list=None, main_exp_name=None
):
    # logfile_path = "./log/estimate_theta_{}/".format(dataset_mode)
    # modelfile_path = "./models/"
    if main_exp_name is None:
        main_exp_name = "{}_L{}_H{}_multiple".format(
            model_type, options[model_type]["n_layers"], options[model_type]["n_hidden"]
        )
    else:
        pass

    base_config_dirname = os.path.dirname("./config/configurations_alltheta_pfixed.json")

    list_of_config_folder_paths = create_file_paths(
        params_combination_list=params_combination_list, filepath=base_config_dirname, main_exp_name=main_exp_name
    )

    # list_of_gs_jsonfile_paths = create_file_paths(params_combination_list=params_combination_list,
    #                                        filepath=modelfile_path,
    #                                        main_exp_name=main_exp_name)

    list_of_config_files = []

    for i, config_folder_path in enumerate(list_of_config_folder_paths):
        config_filename = "configurations_alltheta_pfixed_gru_M{}_P{}_N{}.json".format(
            params_combination_list[i]["num_trajectories"],
            params_combination_list[i]["num_realizations"],
            params_combination_list[i]["N_seq"],
        )
        os.makedirs(config_folder_path, exist_ok=True)
        config_file_name_full = os.path.join(config_folder_path, config_filename)
        list_of_config_files.append(config_file_name_full)

    # Print out the model files
    # print("Config files to be created at:")
    # print(list_of_config_files)

    return list_of_config_files


class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_splits_file(splits_filename):
    with open(splits_filename, "rb") as handle:
        splits = pkl.load(handle)
    return splits


def load_saved_dataset(filename):
    with open(filename, "rb") as handle:
        z_xy = pkl.load(handle)
    return z_xy


def save_dataset(z_xy, filename):
    # Saving the dataset
    with open(filename, "wb") as handle:
        pkl.dump(z_xy, handle, protocol=pkl.HIGHEST_PROTOCOL)


def check_if_dir_or_file_exists(file_path, file_name=None):
    flag_dir = os.path.exists(file_path)
    if file_name is not None:
        flag_file = os.path.isfile(os.path.join(file_path, file_name))
    else:
        flag_file = None
    return flag_dir, flag_file

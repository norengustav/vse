# Author: (c) 2025 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)

import os

from src.data.basic_data_parameters import BasicDataParameters


def basic_data_to_str(basic_data: BasicDataParameters) -> str:
    outstr = (
        f"{basic_data.dataset_type}_"
        f"T{basic_data.trajectory_length}_"
        f"N{basic_data.num_trajectories}_"
        f"m{basic_data.num_states}_"
        f"n{basic_data.num_obs}_"
        f"s{basic_data.sigma_e2_db}_"
        f"smnr{basic_data.smnr_db}_"
        f"tN{basic_data.num_test_trajectories}_"
        f"tT{basic_data.test_trajectory_length}"
    )

    return outstr


def create_full_data_path(
    basic_data_parameters: BasicDataParameters,
    dataset_basepath: str,
    test: bool = False,
):
    """Create the dataset based on the dataset parameters, currently this name is
    partially hard-coded and should be the same in the correspodning parsing function
    for the `main`.py files.

    Args:
        basic_data_parameters: BasicDataParameters
        dataset_basepath (str, optional): Location to save the data. Defaults to
            "./data/".
        test: create test filename

    Returns:
        dataset_fullpath: string describing output filename with full / absolute path.
    """
    test_train_str = ""
    if test:
        test_train_str = "test_data"
    else:
        test_train_str = "train_data"
    basic_data_str = basic_data_to_str(basic_data_parameters)
    datafile = f"{test_train_str}_{basic_data_str}.pkl"

    dataset_fullpath = os.path.join(dataset_basepath, datafile)
    return dataset_fullpath

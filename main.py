# Author: (c) 2025 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)

import os
import shutil

import src.data.generate
import src.entry
import src.ml.test
from src.data.basic_data_parameters import BasicDataParameters


def clean_out_folders():
    # clean out if it exists
    if os.path.exists("out"):
        shutil.rmtree("out")


def main():
    clean_out_folders()
    outpath = "out"

    trajectory_length = 100
    num_trajectories = 100
    num_test_trajectories = 100
    test_trajectory_length = 100
    num_states = 3
    num_obs = 64
    sigma_e2_db = -10.0
    # smnr_list = [-5.0, 0.0, 10.0, 20.0, 30.0]
    # smnr_list = [20.0, 30.0]
    smnr_list = [0.0]
    dataset_type = "LorenzSSM"
    basic_data_list = [
        BasicDataParameters(
            trajectory_length=trajectory_length,
            num_trajectories=num_trajectories,
            num_states=num_states,
            num_obs=num_obs,
            sigma_e2_db=sigma_e2_db,
            smnr_db=smnr,
            dataset_type=dataset_type,
            num_test_trajectories=num_test_trajectories,
            test_trajectory_length=test_trajectory_length,
        )
        for smnr in smnr_list
    ]

    # Generate data
    for basic_data in basic_data_list:
        src.data.generate.generate_data(
            basic_data_parameters=basic_data,
            output_path=outpath,
            generate_test_data=True,
        )

    # Train models
    for basic_data in basic_data_list:
        src.entry.train(outpath=outpath, basic_data=basic_data)

    # Test models
    src.ml.test.test(outpath, basic_data_list)


if __name__ == "__main__":
    main()

# Author: (c) 2024 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)

from dataclasses import dataclass


@dataclass
class BasicDataParameters:
    # Length of each such training data trajectory
    trajectory_length: int = 100
    # The number of i.i.d. trajectories that constitute the training data
    num_trajectories: int = 1000
    # Number of hidden states in the process
    num_states: int = 3
    # Number of observable dimensions in the measurement system
    num_obs: int = 64
    # Set the process noise level (in dB)
    # TODO: remove from basic data parameters, for non-generated data this is unknown
    sigma_e2_db: float = -10.0
    # Signal to measurement noise ratio (in dB)
    smnr_db: float = 30.0
    # Dynamical system e.g. LorenzSSM
    dataset_type: str = "LorenzSSM"
    # Test data trajectories
    num_test_trajectories: int = 100
    test_trajectory_length: int = 1000

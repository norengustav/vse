# Modified by: (c) 2024 Gustav Norén
# Original author: (c) 2023 Anubhab Ghosh
# This code is licensed under MIT licence (see LICENSE file for details)

import math

import numpy as np

import src.data.measurement_functions as mfn
import src.util.math


class LorenzSSM:
    """
    This class defines the state space model for the Lorenz-63 attractor (LorenzSSM) for `alpha=0` or
    for the Chen attractor (ChenSSM) using `alpha=1`. The simulation in discrete-time was done using a Taylor series
    approximation of the matrix exponential (ref. to equation in the manuscript).

    The function described in `A_fn` is in accordance with the generalized Lorenz form introduced in
    Čelikovský, Sergej, and Guanrong Chen. "On the generalized Lorenz canonical form."
    Chaos, Solitons & Fractals 26.5 (2005): 1271-1276.
    """

    def __init__(
        self,
        num_states,
        num_obs,
        param_j,
        delta,
        delta_d,
        alpha=0.0,
        decimate=False,
        mu_e=None,
        mu_w=None,
        matrix_h=None,
        use_taylor=True,
    ) -> None:
        self.n_states = num_states
        self.param_j = param_j
        self.delta = delta
        self.alpha = alpha  # alpha = 0 -- Lorenz attractor, alpha = 1 -- Chen attractor
        self.delta_d = delta_d
        self.n_obs = num_obs
        self.decimate = decimate
        self.mu_e = mu_e
        if matrix_h is None:
            self.matrix_h = np.eye(self.n_obs)
        else:
            self.matrix_h = matrix_h
        self.mu_w = mu_w
        self.use_Taylor = use_taylor
        self.rng = np.random.default_rng()

    def matrix_a_fn(self, z):
        return np.array(
            [
                [-(10 + 25 * self.alpha), (10 + 25 * self.alpha), 0],
                [(28 - 35 * self.alpha), (29 * self.alpha - 1), -z],
                [0, z, -(8.0 + self.alpha) / 3],
            ]
        )

    def f_linearize(self, x):
        self.F = np.eye(self.n_states)
        for j in range(1, self.param_j + 1):
            # self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)
            self.F += np.linalg.matrix_power(
                self.matrix_a_fn(x[0]) * self.delta, j
            ) / math.factorial(j)

        return self.F @ x

    def set_state_cov(self, sigma_e2=0.1):
        self.state_cov_ce = sigma_e2 * np.eye(self.n_states)

    def set_measurement_cov(self, sigma_w2=1.0):
        self.measurement_cov_cw = sigma_w2 * np.eye(self.n_obs)

    def generate_state_sequence(self, trajectory_length, sigma_e2_db):
        self.sigma_e2 = src.util.math.decibel_to_linear(sigma_e2_db)
        self.set_state_cov(sigma_e2=self.sigma_e2)
        x_lorenz = np.zeros((trajectory_length, self.n_states))
        e_k_arr = self.rng.multivariate_normal(
            self.mu_e, self.state_cov_ce, size=(trajectory_length,)
        )

        for t in range(0, trajectory_length - 1):
            x_lorenz[t + 1] = self.f_linearize(x_lorenz[t]) + e_k_arr[t]

        x_lorenz_d = x_lorenz[0 : trajectory_length : self.decimation_k, :]

        # Normalization of the data
        # x_lorenz_d = src.util.math.normalize(x_lorenz_d)

        # Put bias on the mean for Lorenz
        # if self.alpha == 0.0:
        #     x_lorenz_d = x_lorenz_d + 30

        return x_lorenz_d

    def generate_measurement_sequence(self, x_lorenz, trajectory_length, smnr_db=10.0):
        # signal_p = ((self.h_fn(x_lorenz) - np.zeros_like(x_lorenz))**2).mean()
        hx_lorenz = np.asarray(
            [
                mfn.camera_model(x_lorenz[i, :]).reshape((-1,))
                for i in range(x_lorenz.shape[0])
            ]
        )
        signal_p = np.var(hx_lorenz)
        self.sigma_w2 = signal_p / src.util.math.decibel_to_linear(smnr_db)
        self.set_measurement_cov(sigma_w2=self.sigma_w2)
        w_k_arr = self.rng.multivariate_normal(
            self.mu_w, self.measurement_cov_cw, size=(trajectory_length,)
        )
        y_lorenz = np.zeros((trajectory_length, self.n_obs))

        for t in range(0, trajectory_length):
            y_lorenz[t] = mfn.camera_model(x_lorenz[t]) + w_k_arr[t]

        return y_lorenz

    def generate_single_sequence(self, trajectory_length, sigma_e2_db, smnr_db):
        generation_trajectory_length = trajectory_length
        self.decimation_k = 1
        if self.decimate:
            self.decimation_k = int(self.delta / self.delta_d)
            generation_trajectory_length = trajectory_length * self.decimation_k

        x_lorenz = self.generate_state_sequence(
            trajectory_length=generation_trajectory_length, sigma_e2_db=sigma_e2_db
        )
        y_lorenz = self.generate_measurement_sequence(
            x_lorenz=x_lorenz,
            trajectory_length=generation_trajectory_length // self.decimation_k,
            smnr_db=smnr_db,
        )

        return x_lorenz, y_lorenz, self.measurement_cov_cw

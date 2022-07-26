#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : analyzer.py
created time : 2022/07/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np
from scipy.signal import savgol_filter
from mdpy.analyser import AnalyserResult
from mdpy.unit import *
from mdpy.utils import check_quantity


DENOMINATOR_FACTOR = 1e-10


class Analyzer:
    def __init__(
        self,
        num_bins: int,
        bin_range: list[float],
        spring_constant,
        temperature,
        cv_file_path: str,
    ) -> None:
        self._num_bins = num_bins
        self._bin_range = bin_range
        self._spring_constant = check_quantity(
            spring_constant, default_energy_unit / default_length_unit**2
        )
        self._temperature = check_quantity(temperature, default_temperature_unit)
        self._cv_file_path = cv_file_path

        # Attributes
        self._num_trajectories: int = 0
        self._num_samples_per_trajectories: int = 0
        # cv_trajectory: (num_trajectories, num_samples_per_trajectories)
        self._cv_trajectory: np.ndarray = None
        # cv_center: (num_trajectories)
        self._cv_center: np.ndarray = None

        # Parse input
        self._parse_cv_file()

    def _parse_cv_file(self):
        cv_res, cv_center = [], []
        self._num_trajectories = 0
        self._num_samples_per_trajectories = 0
        with open(self._cv_file_path, "r") as f:
            line = f.readline()
            # Template: Start sampling with cv at [0, 0, -2.000]
            # line[line.index("[") + 1 : -2] -> '0, 0, -2.000'
            # cv_center.append(
            #     [float(i) for i in line[line.index("[") + 1 : -2].split()]
            # )
            cv_center.append(float(line.split("at ")[-1]))
            cur_cv_res = []
            while line:
                line = f.readline()
                if line.startswith("Start"):
                    cv_res.append(np.array(cur_cv_res))  # Append result of last cv

                    # cv_center.append(
                    #     [float(i) for i in line[line.index("[") + 1 : -2].split(",")]
                    # )
                    cv_center.append(float(line.split("at ")[-1]))
                    cur_cv_res = []
                else:
                    data = [float(i) for i in line.split()]
                    if data != []:
                        cur_cv_res.append(data)
        cv_res.append(np.array(cur_cv_res))
        self._cv_trajectory = np.stack(cv_res)[:, :, 2]
        self._cv_center = np.stack(cv_center)  # [:, 2]
        (
            self._num_trajectories,
            self._num_samples_per_trajectories,
        ) = self._cv_trajectory.shape
        pbc = 22.5384
        scaled = self._cv_trajectory / pbc
        scaled = np.round(scaled) - scaled
        print(scaled * pbc)

    def analysis(self, max_iterations=500, error_tolerance=1e-15):
        cv_hist_array = np.zeros([self._num_trajectories, self._num_bins])
        bias_factor_array = np.zeros([self._num_trajectories, self._num_bins])
        num_samples_vec = np.zeros([self._num_trajectories, 1])
        for trajectory_index in range(self._num_trajectories):
            cv_hist_array[trajectory_index, :], bin_edges = np.histogram(
                self._cv_trajectory[trajectory_index, :],
                self._num_bins,
                range=(self._bin_range[0], self._bin_range[1]),
            )
            bin_width = bin_edges[1] - bin_edges[0]
            bin_centers = bin_edges[:-1] + bin_width / 2
            bias_factor_array[trajectory_index, :] = np.exp(
                -(
                    self._spring_constant
                    * Quantity(
                        (self._cv_center[trajectory_index] - bin_centers) ** 2,
                        nanometer**2,
                    )
                    / self._temperature
                    / KB
                ).value
            )
            num_samples_vec[trajectory_index, 0] = self._num_samples_per_trajectories
        # Iteration solve
        normalization_vec = np.ones([self._num_trajectories, 1])
        p_est_cur = cv_hist_array.sum(0) / (
            (num_samples_vec * normalization_vec * bias_factor_array).sum(0)
            + DENOMINATOR_FACTOR
        )
        for i in range(max_iterations):
            normalization_vec = 1.0 / (
                (bias_factor_array * p_est_cur).sum(1) * bin_width + DENOMINATOR_FACTOR
            ).reshape([self._num_trajectories, 1])
            p_est_cur, p_est_pre = (
                cv_hist_array.sum(0)
                / (
                    (num_samples_vec * normalization_vec * bias_factor_array).sum(0)
                    + DENOMINATOR_FACTOR
                ),
                p_est_cur,
            )
            msd_error = ((p_est_pre - p_est_cur) ** 2).sum()
            if msd_error < error_tolerance:
                break
        free_energy = (
            -(self._temperature * KB * Quantity(np.log(p_est_cur)))
            .convert_to(default_energy_unit)
            .value
        )
        free_energy[np.isinf(free_energy)] = free_energy[~np.isinf(free_energy)].max()
        # Output
        title = "WHAM result"
        description = {
            "num_trajectories": "The number of trajectories, shape: 1, unit: dimensionless",
            "cv_hist_array": "The histogram of cv for each trajectory, shape: (num_trajectories, num_bins), unit: dimensionless",
            "cv_norm_hist_array": "The normalized histogram of cv for each trajectory, shape: (num_trajectories, num_bins), unit: dimensionless",
            "cv_bin_edges": "The bin edges of cv's histogram, shape: (num_bins+1), unit: the dimension of cv",
            "cv_bin_centers": "The bin centers of cv's histogram, shape: (num_bins), unit: the dimension of cv",
            "cv_bin_width": "The bin width of cv's histogram, shape: 1, unit: the dimension of cv",
            "p_est": "The estimation of probability distribution, shape: (num_bins), unit: dimensionless",
            "free_energy": "The free energy along bin_edge, shape: (num_bins), unit: default_energy_unit",
        }
        data = {
            "num_trajectories": self._num_trajectories,
            "cv_hist_array": cv_hist_array,
            "cv_norm_hist_array": cv_hist_array / num_samples_vec,
            "cv_bin_edges": bin_edges,
            "cv_bin_centers": bin_centers,  # - bin_centers.mean(),
            "cv_bin_width": bin_width,
            "p_est": p_est_cur,
            "free_energy": free_energy - free_energy.min(),
        }
        return AnalyserResult(title=title, description=description, data=data)

    def filter_free_energy(self, free_energy: np.ndarray, is_zero: bool = True):
        res = savgol_filter(free_energy, 15, 3)
        if is_zero:
            return res - res.min()
        return res

    @property
    def cv_trajectory(self) -> np.ndarray:
        return self._cv_trajectory

    @property
    def cv_center(self) -> np.ndarray:
        return self._cv_center

    @property
    def num_trajectories(self) -> int:
        return self._num_trajectories

    @property
    def num_samples_per_trajectories(self) -> int:
        return self._num_samples_per_trajectories


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cv_file_path = os.path.join(
        cur_dir, "./out/04_nvt_sampling_50kcal_start_from_center/new.txt"
    )

    analyzer = Analyzer(
        num_bins=400,
        bin_range=[21.5, 25],
        spring_constant=Quantity(50, kilocalorie_permol / nanometer**2),
        temperature=Quantity(300, kelvin),
        cv_file_path=cv_file_path,
    )
    res = analyzer.analysis()
    # Visualize
    free_energy = (
        Quantity(res.data["free_energy"], default_energy_unit)
        / Quantity(300, kelvin)
        / KB
    ).value
    if not True:
        free_energy = analyzer.filter_free_energy(free_energy)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(res.data["cv_bin_centers"], free_energy)
    ax.set_ylim(-0.1)
    ax_twinx = ax.twinx()
    data = res.data["cv_norm_hist_array"]
    data = np.exp(-data)
    data -= data.min()
    for i in range(res.data["num_trajectories"]):
        ax_twinx.plot(res.data["cv_bin_centers"], data[i, :], "--", lw=0.6)
    plt.show()

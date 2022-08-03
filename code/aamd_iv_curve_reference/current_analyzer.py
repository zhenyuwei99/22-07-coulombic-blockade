#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : current_analyzer.py
created time : 2022/08/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import h5py
import numpy as np
from mdpy.unit import *


class CurrentAnalyzer:
    def __init__(self, data_dir: str, z_length: float) -> None:
        if not os.path.exists(data_dir):
            raise KeyError("%s does not exist" % data_dir)
        self._data_dir = data_dir
        self._z_length = z_length
        unsort_file_list = self._generate_file_list()
        self._ele_list, self._file_list = self._sort_file_list(unsort_file_list)

    def _generate_file_list(self):
        file_list = []
        for i in os.listdir(self._data_dir):
            file_path = os.path.join(self._data_dir, i, "sample")
            if os.path.exists(os.path.join(file_path, "restart.pdb")):
                file_list.append(os.path.join(file_path, "sample.h5"))
        return file_list

    def _sort_file_list(self, file_list: list):
        ele_list = []
        for name in file_list:
            ele = 1 if "plus" in name else -1
            ele *= float(name.split("/")[-3].split("-")[-2])
            ele_list.append(ele)
        return [list(i) for i in zip(*sorted(zip(ele_list, file_list)))]

    def analyze(self):
        current = []
        for index, file_path in enumerate(self._file_list):
            current.append(self._analyze_single_file(file_path))
            print(file_path, current[-1])
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=[16, 8])
        ax.plot(self._ele_list, current)
        fig.tight_layout()
        plt.savefig(os.path.join(cur_dir, "iv.png"))

    def _analyze_single_file(self, file_path: str):
        with h5py.File(file_path, "r") as f:
            num_epochs = f["num_epochs"][()]
            num_particles = f["sample-0/sod-positions"][()].shape[0]
            positions = np.zeros([2, num_particles, num_epochs], np.float32)
            for epoch in range(num_epochs):
                positions[0, :, epoch] = f["sample-%d/sod-positions" % epoch][()][:, 2]
                positions[1, :, epoch] = f["sample-%d/cla-positions" % epoch][()][:, 2]
        scaled = positions / self._z_length
        positions = (scaled - np.round(scaled)) * self._z_length
        index = positions < 0
        positions[index] = 0
        positions[~index] = 1
        current = []
        for interval in range(200, 700):
            num_pass_sod = (
                np.count_nonzero(
                    (positions[0, :, :-interval] == 1)
                    & (positions[0, :, interval:] == 0)
                )
                - np.count_nonzero(
                    (positions[0, :, :-interval] == 0)
                    & (positions[0, :, interval:] == 1)
                )
            ) / (
                num_epochs - interval
            )  # Sum of probability of each particle pass the surface in interval time
            num_pass_cla = (
                np.count_nonzero(
                    (positions[1, :, :-interval] == 1)
                    & (positions[1, :, interval:] == 0)
                )
                - np.count_nonzero(
                    (positions[1, :, :-interval] == 0)
                    & (positions[1, :, interval:] == 1)
                )
            ) / (
                num_epochs - interval
            )  # Sum of probability of each particle pass the surface in interval time
            current.append((num_pass_sod - num_pass_cla) / interval)
        current = (
            Quantity(np.array(current).mean() / 2000, default_charge_unit / femtosecond)
            .convert_to(ampere)
            .value
        )
        return current


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, "out/15.00-nm-pore/out/")
    analyzer = CurrentAnalyzer(data_dir, 229.401)
    analyzer.analyze()

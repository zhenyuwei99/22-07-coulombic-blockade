#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : analyzer.py
created time : 2022/07/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np
import mdpy as md
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from mdpy.unit import *
from mdpy.utils import check_quantity
from job import ION_DICT


class PNPAnalyzer:
    def __init__(self, root_dir: str) -> None:
        self._root_dir = root_dir
        self._voltage, self._target_files = self._get_root_dir_information()
        (
            self._ion_types,
            self._ion_valences,
            self._z_index,
        ) = self._get_grid_information()
        self._num_ion_types = len(self._ion_types)

    def _get_root_dir_information(self):
        target_files = [
            os.path.join(self._root_dir, i, "res.grid")
            for i in os.listdir(self._root_dir)
            if os.path.exists(os.path.join(self._root_dir, i, "res.grid"))
        ]
        voltage_list = [
            float(os.path.dirname(target_file).split("/")[-1].split("V")[0])
            for target_file in target_files
        ]
        voltage_list, target_files = [
            list(i) for i in zip(*sorted(zip(voltage_list, target_files)))
        ]
        return (
            Quantity(voltage_list, volt).convert_to(default_voltage_unit).value,
            target_files,
        )

    def _get_grid_information(self):
        grid = md.io.GridParser(self._target_files[0]).grid
        ion_types = [
            i.split("_")[0] for i in grid.field.__dict__.keys() if "diffusion" in i
        ]
        ion_valences = [ION_DICT[i.lower()]["valence"] for i in ion_types]
        z_index = grid.shape[2] // 2
        return ion_types, ion_valences, z_index

    def analysis(self):
        current_res = [[] for _ in range(self._num_ion_types)]
        for target_file in self._target_files:
            grid = md.io.GridParser(target_file).grid
            for i in range(self._num_ion_types):
                current_res[i].append(
                    self._analysis(
                        grid, self._ion_types[i], self._ion_valences[i], self._z_index
                    )
                )
            print("Finish %s" % target_file.split("/")[-2])
        current_functions = []
        for i in range(self._num_ion_types):
            current_functions.append(self._fit(self._voltage, current_res[i]))
        return current_functions

    def _analysis(
        self,
        grid: md.core.Grid,
        ion_type: str,
        ion_valence: float,
        z_index: int,
        temperature: Quantity = Quantity(300, kelvin),
    ):
        temperature = check_quantity(temperature, kelvin)
        beta = 1 / (temperature * KB).convert_to(default_energy_unit).value
        inv_grid_width = 1 / grid.grid_width[2]  # Z direction
        density = getattr(grid.field, "%s_density" % ion_type)
        diffusion = getattr(grid.field, "%s_diffusion_coefficient" % ion_type)
        energy = grid.field.electric_potential * ion_valence * beta
        channel_mask = 1 - grid.field.channel_shape[1:-1, 1:-1, z_index]
        flux = (
            -(0.5 * inv_grid_width)
            * (diffusion[1:-1, 1:-1, z_index + 1] + diffusion[1:-1, 1:-1, z_index])
            * (
                density[1:-1, 1:-1, z_index + 1]
                - density[1:-1, 1:-1, z_index]
                + 0.5
                * (density[1:-1, 1:-1, z_index + 1] + density[1:-1, 1:-1, z_index])
                * (energy[1:-1, 1:-1, z_index + 1] - energy[1:-1, 1:-1, z_index])
            )
        )
        current = float(
            (
                ion_valence
                * grid.grid_width[0]
                * grid.grid_width[1]
                * (flux * channel_mask).sum()
            ).get()
        )
        return current

    def _fit(self, voltage, current):
        return interpolate.interp1d(voltage, current, "cubic", fill_value="extrapolate")

    @property
    def voltage(self):
        return self._voltage


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    pot_ls, cla_ls = 9.4113406, 95.11422504
    root_dir_name = "out/ls-simulate-annealing/pot-%.4f-cla-%.4f" % (pot_ls, cla_ls)
    root_dir = os.path.join(cur_dir, root_dir_name)
    img_file_path = os.path.join(cur_dir, root_dir_name.split("/")[-1] + ".png")
    print(img_file_path)
    analyzer = PNPAnalyzer(root_dir)
    current_functions = analyzer.analysis()
    voltage = np.linspace(-1, 1, 100, endpoint=True)
    current = np.array([f(voltage) for f in current_functions]).sum(0)
    fig, ax = plt.subplots(1, 1, figsize=[20, 8])
    font_big, font_mid = 20, 15
    ax.plot(
        voltage, current, label="PNP solution", color="navy",
    )
    if True:
        experiment_file = os.path.join(cur_dir, "experiment/4nm-1mol-01.xlsx")
        df = pd.read_excel(experiment_file, header=None)
        voltage_list = np.array(df[0]) / 1000
        current_list = np.array(df[2])
        if True:
            voltage_list *= -1
            current_list *= -1
        ax.plot(voltage_list, current_list, ".-", label="Experiment", color="brown")
    ax.set_xlabel("Voltage (V)", fontsize=font_big)
    ax.set_ylabel("Current (A)", fontsize=font_big)
    ax.tick_params(labelsize=font_mid)
    ax.legend(fontsize=font_big)
    fig.tight_layout()
    plt.savefig(os.path.join(cur_dir, img_file_path))
    plt.close()

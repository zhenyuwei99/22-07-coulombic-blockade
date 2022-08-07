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
from scipy.signal import savgol_filter
from mdpy.unit import *
from mdpy.utils import check_quantity


def analysis(
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


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "out/5a-pore-250a-thick")
    file_name = out_dir.split("/")[-1]
    target_files = [
        os.path.join(out_dir, i, "res.grid")
        for i in os.listdir(out_dir)
        if os.path.exists(os.path.join(out_dir, i, "res.grid"))
    ]
    ele_list = []
    pot_res, cla_res = [], []
    for target_file in target_files:
        factor = -1 if "minus" in target_file else 1
        ele_list.append(factor * float(target_file.split("/")[-2].split("-")[2]))
    ele_list, target_files = [
        list(i) for i in zip(*sorted(zip(ele_list, target_files)))
    ]
    for target_file in target_files:
        index = 256
        grid = md.io.GridParser(target_file).grid
        pot_res.append(analysis(grid, "pot", 1, index))
        cla_res.append(analysis(grid, "cla", -1, index))
        print(target_file)
    pot_res = np.array(pot_res)
    cla_res = np.array(cla_res)
    fig, ax = plt.subplots(1, 1, figsize=[20, 8])
    font_big, font_mid = 20, 15
    #     ax.plot(
    #         ele_list,
    #         Quantity(pot_res, default_charge_unit / default_time_unit)
    #         .convert_to(ampere)
    #         .value,
    #         ".-",
    #         label="Pot solution",
    #     )
    #     ax.plot(
    #         ele_list,
    #         Quantity(cla_res, default_charge_unit / default_time_unit)
    #         .convert_to(ampere)
    #         .value,
    #         ".-",
    #         label="Cla solution",
    #     )
    ax.plot(
        ele_list,
        Quantity(pot_res + cla_res, default_charge_unit / default_time_unit)
        .convert_to(ampere)
        .value,
        ".-",
        label="PNP solution",
        color="navy",
    )
    if True:
        experiment_file = os.path.join(cur_dir, "experiment/1nm-1mol.xlsx")
        df = pd.read_excel(experiment_file, header=None)
        ele_list = np.array(df[0]) / 1000
        current_list = np.array(df[1])
        if False:
            ele_list *= -1
            current_list *= -1
        ax.plot(ele_list, current_list, ".-", label="Experiment", color="brown")
    ax.set_xlabel("Voltage (V)", fontsize=font_big)
    ax.set_ylabel("Current (A)", fontsize=font_big)
    ax.tick_params(labelsize=font_mid)
    ax.legend(fontsize=font_big)
    fig.tight_layout()
    plt.savefig(os.path.join(cur_dir, file_name + ".png"))
    plt.close()

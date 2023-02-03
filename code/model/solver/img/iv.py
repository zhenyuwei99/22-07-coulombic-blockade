#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : iv.py
created time : 2022/12/15
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import numpy as np
import cupy as cp
import mdpy as md

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from solver import *
from solver.analysis import *


def get_voltage_list(grid_dir, name_template):
    grid_file_name_list = [
        i for i in os.listdir(grid_dir) if i.startswith(name_template)
    ]
    i = grid_file_name_list[0]
    voltage_list = [
        float(i.split(".grid")[0].split(name_template)[-1][1:-1])
        for i in grid_file_name_list
    ]
    voltage_list = np.sort(voltage_list)
    return list(voltage_list)


def get_grid_file_info(grid_dir, r0, z0, zs, w0, grid_width, rho, ion_types, is_pnp):
    name_template = generate_name(
        r0=r0,
        z0=z0,
        zs=zs,
        w0=w0,
        grid_width=grid_width,
        rho=rho,
        ion_types=ion_types,
        is_pnp=is_pnp,
        voltage=0,
    )
    name_template = "-".join(name_template.split("-")[:-1])
    voltage_list = get_voltage_list(grid_dir=grid_dir, name_template=name_template)
    grid_file_path_list = []
    for voltage in voltage_list:
        name = generate_name(
            r0=r0,
            z0=z0,
            zs=zs,
            w0=w0,
            grid_width=grid_width,
            rho=rho,
            ion_types=ion_types,
            is_pnp=is_pnp,
            voltage=voltage,
        )
        grid_file_path_list.append(os.path.join(grid_dir, name + ".grid"))
    return grid_file_path_list, voltage_list


def visualize_iv(fig, ax, grid_file_path_list, voltage_list, ion_types):
    current_list = []
    for grid_file_path in grid_file_path_list:
        grid = md.io.GridParser(grid_file_path).grid
        current = 0
        for ion_type in ion_types:
            ion_current = analysis_current(grid=grid, ion_type=ion_type)
            current += ion_current
            # print(ion_type, ion_current)
        current_list.append(-current)

    ax.plot(voltage_list, current_list, ".-")


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    grid_dir = os.path.join(cur_dir, "../out")

    rho = 0.15
    r0 = 10.16
    z0, zs, w0, grid_width = 25, 50, 25, 0.5
    ion_types = ["k", "cl"]
    target_ion_types = ["k"]
    img_file_path = os.path.join(cur_dir, "test.png")

    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    is_pnp = False
    grid_file_path_list, voltage_list = get_grid_file_info(
        grid_dir=grid_dir,
        r0=r0,
        z0=z0,
        zs=zs,
        w0=w0,
        grid_width=grid_width,
        rho=rho,
        ion_types=ion_types,
        is_pnp=is_pnp,
    )
    visualize_iv(
        fig,
        ax,
        grid_file_path_list=grid_file_path_list,
        voltage_list=voltage_list,
        ion_types=target_ion_types,
    )

    is_pnp = True
    grid_file_path_list, voltage_list = get_grid_file_info(
        grid_dir=grid_dir,
        r0=r0,
        z0=z0,
        zs=zs,
        w0=w0,
        grid_width=grid_width,
        rho=rho,
        ion_types=ion_types,
        is_pnp=is_pnp,
    )
    visualize_iv(
        fig,
        ax,
        grid_file_path_list=grid_file_path_list,
        voltage_list=voltage_list,
        ion_types=target_ion_types,
    )
    plt.show()

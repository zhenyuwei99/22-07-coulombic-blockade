#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_vdw.py
created time : 2022/11/14
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np
import mdpy as md
from mdpy.utils import check_quantity_value
from mdpy.unit import *

convert_factor = 2 ** (5 / 6)
VDW_DICT = {
    "c": {
        "sigma": Quantity(1.992 * convert_factor, angstrom),
        "epsilon": Quantity(0.070, kilocalorie_permol),
    },
    "k": {
        "sigma": Quantity(1.764 * convert_factor, angstrom),
        "epsilon": Quantity(0.087, kilocalorie_permol),
    },
    "na": {
        "sigma": Quantity(1.411 * convert_factor, angstrom),
        "epsilon": Quantity(0.047, kilocalorie_permol),
    },
    "ca": {
        "sigma": Quantity(1.367 * convert_factor, angstrom),
        "epsilon": Quantity(0.120, kilocalorie_permol),
    },
    "cl": {
        "sigma": Quantity(2.270 * convert_factor, angstrom),
        "epsilon": Quantity(0.150, kilocalorie_permol),
    },
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, "data")

    pdb = md.io.PDBParser(os.path.join(data_dir, "cnt.pdb"))
    positions = pdb.positions
    pbc = np.diagonal(pdb.pbc_matrix).copy()
    pbc[2] = pbc[2] - 40
    bin_width = 1
    x, y, z = np.meshgrid(
        np.arange(-pbc[0] / 2, pbc[0] / 2 + bin_width, bin_width),
        np.arange(-pbc[1] / 2, pbc[1] / 2 + bin_width, bin_width),
        np.arange(-pbc[2] / 2, pbc[2] / 2 + bin_width, bin_width),
        indexing="ij",
    )
    x_vec = x.reshape([-1, 1])
    dx = (x.reshape([-1, 1]) - positions[:, 0]).reshape(
        list(x.shape) + [positions.shape[0]]
    )
    dy = (y.reshape([-1, 1]) - positions[:, 1]).reshape(
        list(y.shape) + [positions.shape[0]]
    )
    dz = (z.reshape([-1, 1]) - positions[:, 2]).reshape(
        list(z.shape) + [positions.shape[0]]
    )
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    sigma1 = check_quantity_value(VDW_DICT["c"]["sigma"], default_length_unit)
    epsilon1 = check_quantity_value(VDW_DICT["c"]["epsilon"], kilocalorie_permol)
    sigma2 = check_quantity_value(VDW_DICT["k"]["sigma"], default_length_unit)
    epsilon2 = check_quantity_value(VDW_DICT["k"]["epsilon"], kilocalorie_permol)
    sigma = 0.5 * (sigma1 + sigma2)
    epsilon = np.sqrt(epsilon1 * epsilon2)
    scaled_distance = (sigma / (r + 0.01)) ** 6
    vdw = (4 * epsilon * (scaled_distance**2 - scaled_distance)).sum(-1)
    vdw[vdw > 5] = 5

    half_index = x.shape[1] // 2
    fig, ax = plt.subplots(1, 1)
    if not True:
        target_slice = (
            slice(None, None),
            half_index,
            slice(None, None),
        )
        c = ax.contourf(x[target_slice], z[target_slice], vdw[target_slice], 100)
        fig.colorbar(c)
        fig.tight_layout()
    else:
        target_slice = (
            slice(None, None),
            half_index,
            half_index,
        )
        ax.plot(x[target_slice], vdw[target_slice])
    plt.show()

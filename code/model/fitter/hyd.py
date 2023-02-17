#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hyd.py
created time : 2023/02/04
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np
import cupy as cp
import scipy.signal as signal
from mdpy.unit import *
from mdpy.environment import *
from model.potential import HydrationDistributionFunction
from model.utils import get_sigmoid_length


def get_hyd(json_dir: str, ion_type: str):
    n_bulk = (
        (Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton))
        .convert_to(1 / default_length_unit**3)
        .value
    )
    grid_width = 0.5
    targets = ["oxygen", "hydrogen"]
    potential = []
    for target in targets:
        n0 = n_bulk if target == "oxygen" else n_bulk * 2
        pore_file_path = os.path.join(json_dir, "%s-pore.json" % target)
        ion_file_path = os.path.join(json_dir, "%s-%s.json" % (target, ion_type))
        g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
        g_ion = HydrationDistributionFunction(json_file_path=ion_file_path)
        r_cut = get_sigmoid_length(g_ion.bulk_alpha) + g_ion.bulk_rb + 2
        x_ion, y_ion, z_ion = cp.meshgrid(
            cp.arange(-r_cut, r_cut + grid_width, grid_width),
            cp.arange(-r_cut, r_cut + grid_width, grid_width),
            cp.arange(-r_cut, r_cut + grid_width, grid_width),
            indexing="ij",
        )
        x_extend, y_extend, z_extend = cp.meshgrid(
            cp.arange(-10 - r_cut, 10 + r_cut + grid_width, grid_width),
            cp.arange(-10 - r_cut, 10 + r_cut + grid_width, grid_width),
            cp.arange(-r_cut, 20 + r_cut + grid_width, grid_width),
            indexing="ij",
        )
        # Convolve
        pore_distance = z_extend.copy()
        ion_distance = cp.sqrt(x_ion**2 + y_ion**2 + z_ion**2)
        f = g_pore(pore_distance).get()
        g = g_ion(ion_distance)
        print(f.max(), g.max())
        g = g * cp.log(g)
        g *= (Quantity(-300, kelvin) * KB).convert_to(default_energy_unit).value
        g = g.get()
        energy_factor = grid_width**3 * n0
        potential.append((signal.fftconvolve(f, g, "valid") - g.sum()) * energy_factor)
    potential = np.stack(potential).sum(0)
    # potential = potential[1]
    z = cp.arange(0, 20 + grid_width, grid_width)
    return cp.array(potential[:, :, :], CUPY_FLOAT), z.astype(CUPY_FLOAT)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(cur_dir, "../data/hdf")
    hyd, z = get_hyd(json_dir, "pot")
    hyd = Quantity(hyd, default_energy_unit).convert_to(kilocalorie_permol).value

    half_index = hyd.shape[1] // 2
    # c = plt.contourf(hyd[:, half_index, :], 200)
    # plt.colorbar()
    print(hyd[half_index, half_index, :].max())
    plt.plot(z.get(), hyd[half_index, half_index, :], ".-")
    plt.show()

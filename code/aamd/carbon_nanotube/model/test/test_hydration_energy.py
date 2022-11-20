#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_hydration_energy.py
created time : 2022/11/08
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import numpy as np
from scipy import signal
from mdpy.unit import *
from mdpy.utils import check_quantity_value, check_quantity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydration import *
from utils import *


def get_hydration_potential(
    r0: Quantity,
    z0: Quantity,
    n0: Quantity,
    bin_range: np.ndarray,
    bin_width: float,
    pore_file_path: str,
    ion_file_path: str,
):
    r0 = check_quantity_value(r0, angstrom)
    z0 = check_quantity_value(z0, angstrom)
    n0 = check_quantity_value(n0, 1 / angstrom**3)
    g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
    g_ion = HydrationDistributionFunction(json_file_path=ion_file_path)
    r_cut = get_sigmoid_length(g_ion.bulk_alpha) + g_ion.bulk_rb + 4
    print("alpha %.3f, rb %.3f, rcut %.3f" % (g_ion.bulk_alpha, g_ion.bulk_rb, r_cut))
    bin_range = bin_range.copy()
    # Ion coordinate
    x_ion, y_ion, z_ion = np.meshgrid(
        np.arange(-r_cut, r_cut + bin_width, bin_width),
        np.arange(-r_cut, r_cut + bin_width, bin_width),
        np.arange(-r_cut, r_cut + bin_width, bin_width),
        indexing="ij",
    )
    # Origin coordinate
    x, y, z = np.meshgrid(
        np.arange(bin_range[0, 0], bin_range[0, 1] + bin_width, bin_width),
        np.arange(bin_range[1, 0], bin_range[1, 1] + bin_width, bin_width),
        np.arange(bin_range[2, 0], bin_range[2, 1] + bin_width, bin_width),
        indexing="ij",
    )
    # Extend coordinate
    bin_range[:, 0] -= r_cut
    bin_range[:, 1] += r_cut
    x_extend, y_extend, z_extend = np.meshgrid(
        np.arange(bin_range[0, 0], bin_range[0, 1] + bin_width, bin_width),
        np.arange(bin_range[1, 0], bin_range[1, 1] + bin_width, bin_width),
        np.arange(bin_range[2, 0], bin_range[2, 1] + bin_width, bin_width),
        indexing="ij",
    )
    # Convolve
    pore_distance = get_pore_distance(x_extend, y_extend, z_extend, r0=r0, z0=z0)
    ion_distance = get_distance(x_ion, y_ion, z_ion)

    f = g_pore(pore_distance)
    g = g_ion(ion_distance)
    g = g * np.log(g)
    g = -(Quantity(300 * g, kelvin) * KB).convert_to(kilocalorie_permol).value
    energy_factor = bin_width**3 * n0
    print(g.sum() * energy_factor)
    potential = (g.sum() - signal.fftconvolve(f, g, "valid")) * energy_factor

    return x, y, z, potential


def get_nonpolar_potential(
    sigma,
    epsilon,
    r0: Quantity,
    z0: Quantity,
    n0: Quantity,
    bin_range: np.ndarray,
    bin_width: float,
):
    sigma = check_quantity_value(sigma, angstrom)
    epsilon = check_quantity_value(epsilon, kilocalorie_permol)
    r0 = check_quantity_value(r0, angstrom)
    z0 = check_quantity_value(z0, angstrom)
    n0 = check_quantity_value(n0, 1 / angstrom**3)
    x, y, z = np.meshgrid(
        np.arange(bin_range[0, 0], bin_range[0, 1] + bin_width, bin_width),
        np.arange(bin_range[1, 0], bin_range[1, 1] + bin_width, bin_width),
        np.arange(bin_range[2, 0], bin_range[2, 1] + bin_width, bin_width),
        indexing="ij",
    )
    pore_distance = get_pore_distance(x, y, z, r0=r0, z0=z0)
    relative_distance = (sigma / (pore_distance + 0.01)) ** 6
    potential = 4 * epsilon * (relative_distance**2 - relative_distance)
    return x, y, z, potential


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "../out")
    img_file_path = os.path.join(
        os.path.join(out_dir, "image/test_hydration_energy.png")
    )

    sigma = Quantity(1.992 * 2 ** (5 / 6), angstrom)
    epsilon = Quantity(0.070, kilocalorie_permol)
    r0 = Quantity(10, angstrom)
    z0 = Quantity(5, angstrom)
    n0 = Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton)
    bin_width = 0.25
    bin_range = np.array([[-25.0, 25], [-25, 25], [-25, 25]])

    ion = "pot"
    target = "oxygen"
    pore_file_path = os.path.join(out_dir, "%s-pore.json" % target)
    ion_file_path = os.path.join(out_dir, "%s-%s.json" % (target, ion))
    x, y, z, hydration_potential_oxygen = get_hydration_potential(
        r0=r0,
        z0=z0,
        n0=n0,
        bin_range=bin_range,
        bin_width=bin_width,
        pore_file_path=pore_file_path,
        ion_file_path=ion_file_path,
    )
    target = "hydrogen"
    pore_file_path = os.path.join(out_dir, "%s-pore.json" % target)
    ion_file_path = os.path.join(out_dir, "%s-%s.json" % (target, ion))
    n0 *= Quantity(2)
    x, y, z, hydration_potential_hydrogen = get_hydration_potential(
        r0=r0,
        z0=z0,
        n0=n0,
        bin_range=bin_range,
        bin_width=bin_width,
        pore_file_path=pore_file_path,
        ion_file_path=ion_file_path,
    )
    hydration_potential = hydration_potential_oxygen + hydration_potential_hydrogen
    if True:
        x, y, z, nonpolar_potential = get_nonpolar_potential(
            sigma=sigma,
            epsilon=epsilon,
            r0=r0,
            z0=z0,
            n0=n0,
            bin_range=bin_range,
            bin_width=bin_width,
        )
        potential = hydration_potential + nonpolar_potential
    else:
        r = np.sqrt(x**2 + y**2 + z**2)
        r[r <= 0.1] = 0.1
        factor = Quantity(r * 4 * np.pi * 80, angstrom) * EPSILON0
        potential = (
            (Quantity(-1, elementary_charge**2) / factor)
            .convert_to(kilocalorie_permol)
            .value
        )
        potential += hydration_potential
    pore_distance = get_pore_distance(x, y, z, r0=r0.value, z0=z0.value)
    potential += (pore_distance == 0).astype(np.float64) * 100
    potential[potential >= 5] = 5
    fig, ax = plt.subplots(1, 1, figsize=[12, 8])
    half_index = x.shape[1] // 2
    if True:
        target_slice = (
            slice(None, None),
            half_index,
            slice(None, None),
        )
        c = ax.contourf(x[target_slice], z[target_slice], potential[target_slice], 200)
        Ex, Ez = np.gradient(-potential[target_slice])
        # ax.streamplot(
        #     x[:, 0, 0],
        #     z[0, 0, :],
        #     Ex.T,
        #     Ez.T,
        #     linewidth=1,
        #     density=1.5,
        # )
        fig.colorbar(c)
    else:
        target_slice = (
            half_index,
            half_index,
            slice(10, -10),
        )
        ax.plot(z[target_slice], potential[target_slice])
    fig.tight_layout()
    plt.show()
    # fig.savefig(img_file_path)

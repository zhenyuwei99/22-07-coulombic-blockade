#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : vdw.py
created time : 2023/02/13
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import cupy as cp
import numpy as np
from mdpy.utils import check_quantity_value
from model.core import Grid
from model.utils import *
from model.potential.hdf import HydrationDistributionFunction


class VDWPotential:
    def __init__(self, type1: str, type2, r0, z0, rs) -> None:
        self._r0 = r0
        self._z0 = z0
        self._rs = rs
        self._sigma, self._epsilon = self._get_vdw_parameter(type1, type2)

    def _get_vdw_parameter(self, type1: str, type2: str):
        sigma1 = check_quantity_value(VDW_DICT[type1]["sigma"], default_length_unit)
        epsilon1 = check_quantity_value(VDW_DICT[type1]["epsilon"], default_energy_unit)
        sigma2 = check_quantity_value(VDW_DICT[type2]["sigma"], default_length_unit)
        epsilon2 = check_quantity_value(VDW_DICT[type2]["epsilon"], default_energy_unit)
        return (
            CUPY_FLOAT(0.5 * (sigma1 + sigma2)),
            CUPY_FLOAT(np.sqrt(epsilon1 * epsilon2)),
        )

    def __call__(
        self, grid: Grid, threshold=Quantity(1.5, kilocalorie_permol)
    ) -> cp.ndarray:
        threshold = check_quantity_value(threshold, default_energy_unit)
        r = grid.coordinate.r
        z = grid.coordinate.z
        dist = get_pore_distance_cylinder(r, z, r0=self._r0, z0=self._z0, rs=self._rs)
        vdw = grid.zeros_field(CUPY_FLOAT)
        scaled_distance = (self._sigma / (dist + 0.0001)) ** 6
        vdw = 4 * self._epsilon * (scaled_distance**2 - scaled_distance)
        vdw[vdw > threshold] = threshold
        return vdw.astype(CUPY_FLOAT)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    vdw = VDWPotential(r0=15.16, z0=25, rs=2, type1="c", type2="k")

    beta = 1 / (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value

    r0, z0, rs = 8.15, 25, 5
    grid = Grid(grid_width=0.5, r=[0, 50], z=[-100, 100])
    energy = vdw(grid) * beta

    # plt.plot(grid.coordinate.r[:, 0].get(), energy[:, 0].get(), ".-")
    c = plt.contour(grid.coordinate.r.get(), grid.coordinate.z.get(), energy.get(), 200)
    plt.colorbar(c)
    plt.show()

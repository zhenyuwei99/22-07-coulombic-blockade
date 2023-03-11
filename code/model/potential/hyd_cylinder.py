#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hyd_cylinder.py
created time : 2023/03/02
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import cupy as cp
import numba.cuda as cuda
import cupyx.scipy.signal as signal
from mdpy.utils import check_quantity, check_quantity_value
from mdpy.unit import *
from model import *
from model.potential.hdf import HydrationDistributionFunction


HDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/hdf")


class HydrationPotentialCylinder:
    def __init__(
        self,
        r0,
        ion_type,
        r_cut=Quantity(12, angstrom),
        temperature=Quantity(300, kelvin),
        grid_width=Quantity(0.2, angstrom),
        hdf_dir=HDF_DIR,
    ) -> None:
        # Read input
        self._r0 = r0
        self._ion_type = ion_type
        self._r_cut = check_quantity_value(r_cut, default_length_unit)
        self._h = check_quantity_value(grid_width, default_length_unit)
        self._temperature = check_quantity_value(temperature, kelvin)
        self._hdf_dir = hdf_dir
        # Attribute
        self._ion_name = ION_DICT[self._ion_type]["name"]
        self._kbt = CUPY_FLOAT(
            (Quantity(self._temperature, kelvin) * KB)
            .convert_to(default_energy_unit)
            .value
        )
        self._n_bulk = (
            (Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton))
            .convert_to(1 / default_length_unit**3)
            .value
        )
        z = ION_DICT[self._ion_type]["val"].value
        self._target = ["oxygen"] if z < 0 else ["hydrogen"]
        self._target = ["oxygen", "hydrogen"]
        self._r_range = [0, self._r0]
        self._r_extend_range = [0, self._r0 + self._r_cut]
        self._r_ion_range = [0, self._r_cut]
        self._z_range = [-2, 2]
        self._z_extend_range = [-2 - self._r_cut, 2 + self._r_cut]
        self._z_ion_range = [-self._r_cut, self._r_cut]
        # Kernel
        self._assign_data = cuda.jit(
            nb.void(NUMBA_INT[:], NUMBA_FLOAT[:], NUMBA_FLOAT[:], NUMBA_INT[:])
        )(self._assign_data_kernel)

    def evaluate(self):
        # Mesh
        x, y, z = self._get_mesh_points(self._r_range, self._z_range)
        x_extend, y_extend, z_extend = self._get_mesh_points(
            self._r_extend_range, self._z_extend_range
        )
        x_ion, y_ion, z_ion = self._get_mesh_points(
            self._r_ion_range, self._z_ion_range
        )
        # Convolve
        r_xy = cp.sqrt(x_extend**2 + y_extend**2)
        dist_pore1 = self._r0 - r_xy
        dist_pore2 = self._r0 + r_xy
        dist_ion = cp.sqrt(x_ion**2 + y_ion**2 + z_ion**2)
        hyd = cp.zeros_like(x, CUPY_FLOAT)
        for target in self._target:
            n0 = self._n_bulk if target == "oxygen" else self._n_bulk * 2
            factor = CUPY_FLOAT(self._h**3 * n0)
            pore_file_path = os.path.join(self._hdf_dir, "%s-pore.json" % target)
            ion_file_path = os.path.join(
                self._hdf_dir, "%s-%s.json" % (target, self._ion_name)
            )
            g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
            g_ion = HydrationDistributionFunction(json_file_path=ion_file_path)
            f = g_pore(dist_pore1) * g_pore(dist_pore2)
            g = g_ion(dist_ion)
            g = g * cp.log(g) * -self._kbt
            hyd += (signal.fftconvolve(f, g, "valid") - g.sum()) * factor
        # Average
        hyd = hyd.mean(2)  # Average over z
        r = cp.arange(0, self._r0 + self._h, self._h)
        num_bins = r.shape[0]
        index = (
            cp.sqrt(x[:, :, 0] ** 2 + y[:, :, 0] ** 2)
            * CUPY_FLOAT((num_bins - 1) / self._r0)
        ).reshape(-1)
        index = cp.round(index).astype(CUPY_INT)
        index[index >= num_bins] = -1
        data = hyd.reshape(-1).astype(CUPY_FLOAT)
        target = cp.zeros_like(r, CUPY_FLOAT)
        counts = cp.zeros_like(r, CUPY_INT)
        # Assign data
        thread_per_block = 256
        block_per_grid = int(np.ceil(index.shape[0] / thread_per_block))
        self._assign_data[block_per_grid, thread_per_block](index, data, target, counts)
        hyd = (target / counts).astype(CUPY_FLOAT)
        return r, hyd

    def _get_mesh_points(self, r_range, z_range):
        x = cp.arange(0, r_range[1] + self._h, self._h, CUPY_FLOAT)
        x = cp.hstack([-x[::-1][:-1], x])
        z1 = cp.arange(0, z_range[1] + self._h, self._h, CUPY_FLOAT)
        z2 = cp.arange(0, -z_range[0] + self._h, self._h, CUPY_FLOAT)
        z = cp.hstack([-z2[::-1][:-1], z1])
        x, y, z = cp.meshgrid(x, x, z, indexing="ij")
        return x.astype(CUPY_FLOAT), y.astype(CUPY_FLOAT), z.astype(CUPY_FLOAT)

    @staticmethod
    def _assign_data_kernel(index, data, target, counts):
        thread_id = cuda.grid(1)
        if thread_id >= index.shape[0]:
            return
        i = index[thread_id]
        if i != -1:
            cuda.atomic.add(target, (i), data[thread_id])
            cuda.atomic.add(counts, (i), NUMBA_INT(1))


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    cc_bond_length = 1.418
    r0_list = [i * cc_bond_length * 3 / (2 * np.pi) for i in range(5, 15)]

    for r0 in r0_list:
        hyd = HydrationPotentialCylinder(r0=r0, ion_type="sod")
        r, hyd = hyd.evaluate()
        r, hyd = r.get(), hyd.get()
        hyd *= (Quantity(1, default_energy_unit) / Quantity(300, kelvin) / KB).value

        ratio = np.exp(-hyd)
        factor = 2 * np.pi * r
        print(
            r0,
            np.sum(factor),
            np.sum(factor * ratio),
            (np.sum(factor) - np.sum(factor * ratio)) / np.sum(factor),
        )
        # plt.plot(r.get(), , ".-")
        # plt.plot(r.get(), hyd.get() * convert.value, ".-")

        plt.show()

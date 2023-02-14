#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hyd.py
created time : 2023/02/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.signal as signal
import cupyx.scipy.interpolate as interpolate
from model.core import Grid
from model.utils import *
from model.potential.hdf import HydrationDistributionFunction

RCUT = 12
HDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/hdf")


class HydrationPotential:
    def __init__(
        self,
        ion_type: str,
        r0,
        z0,
        rs,
        grid_width=0.5,
        r_cut=RCUT,
        temperature=300,
        hdf_dir: str = HDF_DIR,
    ) -> None:
        self._ion_type = ion_type
        self._r0 = r0
        self._z0 = z0
        self._rs = rs
        self._grid_width = grid_width
        self._r_cut = r_cut
        self._temperature = temperature
        self._hdf_dir = hdf_dir
        # Other attribute
        self._r_range = [-self._r0 - rs - 2, self._r0 + rs + 2]
        self._r_extend_range = [
            self._r_range[0] - self._r_cut,
            self._r_range[1] + self._r_cut,
        ]
        self._z_range = [0, self._z0 + self._r_cut * 2]
        self._z_extend_range = [
            self._z_range[0] - self._r_cut,
            self._z_range[1] + self._r_cut,
        ]
        self._hyd_fun_range = []
        self._target = ["oxygen", "hydrogen"]
        self._target = ["oxygen"]
        self._n_bulk = (
            (Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton))
            .convert_to(1 / default_length_unit**3)
            .value
        )
        self._hyd_fun = self._get_interpolate()

    def _get_mesh_points(self, r_range, z_range, grid_width):
        x = cp.arange(0, r_range[1] + grid_width, grid_width, CUPY_FLOAT)
        x = cp.hstack([-x[::-1][:-1], x])
        z1 = cp.arange(0, z_range[1] + grid_width, grid_width, CUPY_FLOAT)
        z2 = cp.arange(0, -z_range[0] + grid_width, grid_width, CUPY_FLOAT)
        z = cp.hstack([-z2[::-1][:-1], z1])
        x, y, z = cp.meshgrid(x, x, z, indexing="ij")
        return x.astype(CUPY_FLOAT), y.astype(CUPY_FLOAT), z.astype(CUPY_FLOAT)

    def _get_interpolate(self):
        kBT = CUPY_FLOAT(
            (Quantity(-self._temperature, kelvin) * KB)
            .convert_to(default_energy_unit)
            .value
        )
        x, y, z = self._get_mesh_points(self._r_range, self._z_range, self._grid_width)
        x_extend, y_extend, z_extend = self._get_mesh_points(
            self._r_extend_range, self._z_extend_range, self._grid_width
        )
        dist1, dist2 = get_double_pore_distance_cartesian(
            x_extend, y_extend, z_extend, self._r0, self._z0, self._rs
        )

        x_ion = cp.arange(0, self._r_cut + self._grid_width, self._grid_width)
        x_ion = cp.hstack([-x_ion[::-1][:-1], x_ion])
        x_ion, y_ion, z_ion = cp.meshgrid(x_ion, x_ion, x_ion, indexing="ij")
        dist_ion = cp.sqrt(x_ion**2 + y_ion**2 + z_ion**2)
        hyd = cp.zeros_like(x, CUPY_FLOAT)
        for target in self._target:
            n0 = self._n_bulk if target == "oxygen" else self._n_bulk * 2
            factor = CUPY_FLOAT(self._grid_width**3 * n0)
            pore_file_path = os.path.join(self._hdf_dir, "%s-pore.json" % target)
            ion_file_path = os.path.join(
                self._hdf_dir, "%s-%s.json" % (target, self._ion_type)
            )
            g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
            g_ion = HydrationDistributionFunction(json_file_path=ion_file_path)
            f = g_pore(dist1) * g_pore(dist2)
            g = g_ion(dist_ion)
            g = g * cp.log(g) * kBT
            hyd += (signal.fftconvolve(f, g, "valid") - g.sum()) * factor
        import matplotlib.pyplot as plt

        half_index = x_extend.shape[1] // 2
        plt.contour(
            x_extend[:, half_index, :].get(),
            z_extend[:, half_index, :].get(),
            f[:, half_index, :].get(),
        )
        plt.show()
        half_index = int((hyd.shape[1] - 1) / 2)
        target_slice = (
            slice(half_index, None),
            slice(half_index - 2, half_index + 3),
            slice(None, None),
        )
        r = cp.sqrt(x[target_slice] ** 2 + y[target_slice] ** 2).astype(CUPY_FLOAT)
        self._hyd_fun_range = [float(r.min()), float(r.max())]
        # r = cp.array(x[target_slice], CUPY_FLOAT)
        z = cp.array(z[target_slice], CUPY_FLOAT)
        hyd = cp.array(hyd[target_slice], CUPY_FLOAT)
        y = cp.hstack([r.reshape([-1, 1]), z.reshape([-1, 1])])
        d = hyd.reshape([-1, 1])
        index = cp.unique(y[:, 0] + y[:, 1] * 1j, return_index=True)[1]
        hyd_fun = interpolate.RBFInterpolator(y=y[index], d=d[index])
        return hyd_fun
        x, y, z = self._get_mesh_points(self._r_range, self._z_range, 0.2)
        half_index = int((hyd.shape[1] - 1) / 2)
        r = cp.array(x[:, half_index, :], CUPY_FLOAT)
        z = cp.array(z[:, half_index, :], CUPY_FLOAT)
        y = cp.hstack([r.reshape([-1, 1]), z.reshape([-1, 1])])
        return (r, z, cp.array(hyd_fun(y).reshape(r.shape)))

    def __call__(self, grid: Grid) -> cp.ndarray:
        r = grid.coordinate.r
        z = grid.coordinate.z
        # Inner point
        hyd = grid.zeros_field(CUPY_FLOAT)
        area1 = (
            (r <= self._hyd_fun_range[1])
            & (r >= self._hyd_fun_range[0])
            & (z <= self._z_range[1])
            & (z >= self._z_range[0])
        )
        y = cp.hstack([r[area1].reshape([-1, 1]), z[area1].reshape([-1, 1])])
        hyd[area1] = self._hyd_fun(y).reshape(r[area1].shape)
        area2 = (
            (r <= self._hyd_fun_range[1])
            & (r >= self._hyd_fun_range[0])
            & (z >= -self._z_range[1])
            & (z <= -self._z_range[0])
        )
        y = cp.hstack([r[area2].reshape([-1, 1]), -z[area2].reshape([-1, 1])])
        hyd[area2] = self._hyd_fun(y).reshape(r[area2].shape)
        # Add beyond value
        index = cp.argwhere(area1)
        r_max_index = index[:, 0].max()
        r_min_index = index[:, 0].min()
        z_max_index = index[:, 1].max()
        index = cp.argwhere(area2)
        z_min_index = index[:, 1].min()
        print(z_min_index, z_max_index)
        hyd[: r_max_index + 1, z_max_index:] = hyd[
            : r_max_index + 1, z_max_index : z_max_index + 1
        ]
        hyd[: r_max_index + 1, :z_min_index] = hyd[
            : r_max_index + 1, z_min_index : z_min_index + 1
        ]
        hyd[r_max_index + 1 :, :] = hyd[r_max_index : r_max_index + 1, :]
        return cp.array(hyd, CUPY_FLOAT)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hyd = HydrationPotential(r0=15.16, z0=25, rs=2, ion_type="pot")

    beta = 1 / (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    # hyd *= beta

    r0, z0, rs = 8.15, 25, 5
    grid = Grid(grid_width=0.5, r=[0, 50], z=[-100, 100])
    energy = hyd(grid)

    # plt.plot(grid.coordinate.r[:, 0].get(), energy[:, 0].get(), ".-")
    c = plt.contour(grid.coordinate.r.get(), grid.coordinate.z.get(), energy.get(), 200)
    plt.colorbar(c)
    plt.show()

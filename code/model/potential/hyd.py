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
import cupyx.scipy.signal as signal
import cupyx.scipy.interpolate as interpolate
import torch as tc
import torch.nn as nn
import torch.optim as optim
import numba as nb
import numba.cuda as cuda
from model.core import Grid, Net
from model.utils import *
from model.potential.hdf import HydrationDistributionFunction

RCUT = 12
HDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/hdf")


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


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
        self._z_range = [0, self._z0 + self._r_cut * 3]
        self._z_extend_range = [
            self._z_range[0] - self._r_cut,
            self._z_range[1] + self._r_cut,
        ]
        self._hyd_r_range = []
        self._hyd_z_range = []
        self._target = ["oxygen", "hydrogen"]
        self._target = ["oxygen"]
        self._n_bulk = (
            (Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton))
            .convert_to(1 / default_length_unit**3)
            .value
        )
        self._assign_data = cuda.jit(
            nb.void(
                NUMBA_INT[:, :], NUMBA_FLOAT[:, :], NUMBA_FLOAT[:, :], NUMBA_FLOAT[:, :]
            )
        )(self._assign_data_kernel)
        self._hyd_fun = self._get_fun()

    def _get_mesh_points(self, r_range, z_range, grid_width):
        x = cp.arange(0, r_range[1] + grid_width, grid_width, CUPY_FLOAT)
        x = cp.hstack([-x[::-1][:-1], x])
        z1 = cp.arange(0, z_range[1] + grid_width, grid_width, CUPY_FLOAT)
        z2 = cp.arange(0, -z_range[0] + grid_width, grid_width, CUPY_FLOAT)
        z = cp.hstack([-z2[::-1][:-1], z1])
        x, y, z = cp.meshgrid(x, x, z, indexing="ij")
        return x.astype(CUPY_FLOAT), y.astype(CUPY_FLOAT), z.astype(CUPY_FLOAT)

    def _get_fun(self):
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
            f = g_pore(dist1)  # * g_pore(dist2)
            g = g_ion(dist_ion)
            g = g * cp.log(g) * kBT
            hyd += (signal.fftconvolve(f, g, "valid") - g.sum()) * factor

        r = cp.sqrt(x**2 + y**2).astype(CUPY_FLOAT)
        x = cp.hstack([r.reshape(-1, 1), z.reshape(-1, 1)]).astype(CUPY_FLOAT)
        y = hyd.reshape(-1, 1).astype(CUPY_FLOAT)
        self._hyd_r_range = [0, r.max().get()]
        self._hyd_z_range = [0, self._z0 + 2 * self._r_cut]
        r, z = cp.meshgrid(
            cp.arange(
                self._hyd_r_range[0],
                self._hyd_r_range[1] + self._grid_width,
                self._grid_width,
            ),
            cp.arange(
                self._hyd_z_range[0],
                self._hyd_z_range[1] + self._grid_width,
                self._grid_width,
            ),
            indexing="ij",
        )
        num_r, num_z = r.shape
        index = x / cp.array([self._hyd_r_range[1], self._hyd_z_range[1]])
        index *= cp.array([num_r, num_z]) - 1
        index = cp.round(index).astype(CUPY_INT)
        index[:, 0][index[:, 0] >= num_r] = -1
        index[:, 1][index[:, 1] >= num_z] = -1
        hyd = cp.zeros_like(r, CUPY_FLOAT)
        counts = cp.zeros_like(r, CUPY_FLOAT)

        thread_per_block = 256
        block_per_grid = int(np.ceil(x.shape[0] / thread_per_block))
        self._assign_data[block_per_grid, thread_per_block](index, y, hyd, counts)
        hyd = hyd / counts

        # dist = get_pore_distance_cylinder(r, z, self._r0, self._z0, self._rs)
        # g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
        # hyd = g_pore(dist)
        # hyd = -(
        #     cp.log(hyd)
        #     * (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
        # )
        # c = plt.contour(r.get(), z.get(), hyd.get(), 200)
        # plt.colorbar(c)
        # plt.show()

        x = cp.hstack([r.reshape([-1, 1]), z.reshape([-1, 1])])
        y = hyd.reshape([-1, 1])
        hyd_fun = self._train_net(x, y)

        pred = hyd_fun(x)
        c = plt.contour(
            r.get(), z.get(), pred.detach().cpu().numpy().reshape(r.shape), 200
        )
        # plt.colorbar(c)
        plt.show()
        return hyd_fun

        y = cp.hstack([r.reshape([-1, 1]), z.reshape([-1, 1])])
        d = hyd.reshape([-1, 1])
        index = cp.unique(y[:, 0] + y[:, 1] * 1j, return_index=True)[1]
        hyd_fun = interpolate.RBFInterpolator(y=y[index], d=d[index])
        return hyd_fun

    @staticmethod
    def _assign_data_kernel(index, data, target, counts):
        thread_id = cuda.grid(1)
        if thread_id >= index.shape[0]:
            return
        i = index[thread_id, 0]
        j = index[thread_id, 1]
        if i != -1 and j != -1:
            cuda.atomic.add(target, (i, j), data[thread_id, 0])
            cuda.atomic.add(counts, (i, j), NUMBA_FLOAT(1))

    def _train_net(self, x, y):
        device = tc.device("cuda")
        x = tc.tensor(x.get(), dtype=TORCH_FLOAT, device=device)
        y = tc.tensor(y.get(), dtype=TORCH_FLOAT, device=device)
        x_norm, x_std = x.mean(), x.std()
        y_norm, y_std = y.mean(), y.std()
        x = (x - x_norm) / x_std
        y = (y - y_norm) / y_std
        net = Net(2, [16, 64, 128, 256, 128, 64, 16], 1, device=device)
        net.apply(weights_init)
        optimizer = optim.Adam(net.parameters(), lr=5e-3)
        for epoch in range(2000):
            optimizer.zero_grad()
            loss = ((net(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            print(loss)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        for epoch in range(2000):
            optimizer.zero_grad()
            loss = ((net(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            print(loss)

        def hyd_fun(x):
            x = tc.tensor(x.get(), dtype=TORCH_FLOAT, device=device)
            x = (x - x_norm) / x_std
            y = net(x)
            return y * y_std + y_norm

        return hyd_fun

    def __call__(self, grid: Grid) -> cp.ndarray:
        r = grid.coordinate.r
        z = grid.coordinate.z
        # Inner point
        hyd = grid.zeros_field(CUPY_FLOAT)
        area1 = (
            (r <= self._hyd_r_range[1])
            & (r >= self._hyd_r_range[0])
            & (z <= self._hyd_z_range[1])
            & (z >= self._hyd_z_range[0])
        )
        y = cp.hstack([r[area1].reshape([-1, 1]), z[area1].reshape([-1, 1])])
        hyd[area1] = self._hyd_fun(y).reshape(r[area1].shape)
        area2 = (
            (r <= self._hyd_r_range[1])
            & (r >= self._hyd_r_range[0])
            & (z >= -self._hyd_z_range[1])
            & (z <= -self._hyd_z_range[0])
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

    hyd = HydrationPotential(r0=25.16, z0=25, rs=2, ion_type="pot")

    beta = 1 / (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    # hyd *= beta

    grid = Grid(grid_width=0.5, r=[0, 50], z=[-100, 100])
    energy = hyd(grid)

    half_index = grid.shape[1] // 2
    plt.plot(grid.coordinate.r[:, half_index].get(), energy[:, half_index].get(), ".-")
    # plt.plot(grid.coordinate.z[50, :].get(), energy[50, :].get(), ".-")
    # c = plt.contourf(
    #     grid.coordinate.r.get(), grid.coordinate.z.get(), energy.get(), 200
    # )
    # plt.colorbar(c)
    plt.show()

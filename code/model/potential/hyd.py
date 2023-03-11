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
import torch as tc
from torch.autograd import grad
from mdpy.utils import check_quantity
from mdpy.unit import *
from model.core import Grid, Net
from model.utils import *


HYD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/hyd")


class HydrationPotential:
    def __init__(
        self,
        ion_type: str,
        temperature=300,
        hyd_dir: str = HYD_DIR,
    ) -> None:
        self._ion_type = ion_type
        self._temperature = check_quantity(temperature, kelvin)
        self._hyd_dir = hyd_dir

        # Attribute
        self._net: Net = tc.load(os.path.join(HYD_DIR, "hyd_%s.pkl" % ion_type))
        self._convert = (self._temperature * KB).convert_to(default_energy_unit).value

    def __call__(self, grid: Grid, dist_fun, require_first: True, require_second: True):
        x = []
        for coordinate in grid.coordinate.__dict__.keys():
            if not "_SubGrid" in coordinate:
                x.append(getattr(grid.coordinate, coordinate).reshape(-1, 1))
        x = cp.hstack(x).astype(CUPY_FLOAT).get()
        dim = x.shape[1]
        x = tc.tensor(x, device=self._net.device, dtype=TORCH_FLOAT, requires_grad=True)
        dist = dist_fun(x)
        hyd = self._net(dist) * self._convert
        return_list = [hyd]

        if require_first:
            dhyd = grad(hyd.sum(), x, create_graph=True, retain_graph=True)[0]
            return_list.append([dhyd[:, i] for i in range(dim)])
            if require_second:
                dhyd2_list = []
                for i in range(dim):
                    dhyd2_list.append(
                        grad(dhyd[:, i].sum(), x, retain_graph=True)[0][:, i]
                    )
                return_list.append(dhyd2_list)
        else:
            if require_second:
                raise ValueError(
                    "First order derivative is compulsory for require second order derivative"
                )
        return_list[0] = cp.array(
            return_list[0].detach().cpu().numpy(), CUPY_FLOAT
        ).reshape(grid.shape)
        if require_first:
            return_list[1] = [
                cp.array(i.detach().cpu().numpy(), CUPY_FLOAT).reshape(grid.shape)
                for i in return_list[1]
            ]
            for i in return_list[1]:
                index = cp.argwhere(cp.isnan(i))
                self_index = tuple([index[:, i] for i in range(dim)])
                i[self_index] = CUPY_FLOAT(0.5) * (
                    i[tuple([index[:, i] + 1 for i in range(dim)])]
                    + i[tuple([index[:, i] - 1 for i in range(dim)])]
                )
        if require_second:
            return_list[2] = [
                cp.array(i.detach().cpu().numpy(), CUPY_FLOAT).reshape(grid.shape)
                for i in return_list[2]
            ]
            for i in return_list[2]:
                index = cp.argwhere(cp.isnan(i))
                self_index = tuple([index[:, i] for i in range(dim)])
                i[self_index] = CUPY_FLOAT(0.5) * (
                    i[tuple([index[:, i] + 1 for i in range(dim)])]
                    + i[tuple([index[:, i] - 1 for i in range(dim)])]
                )
        return tuple(return_list)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hyd = HydrationPotential(ion_type="pot")
    r0, z0, rs = 12, 25, 5

    def dist_fun(x: tc.Tensor, r0=r0, z0=z0, rs=rs):
        device = x.device
        r, z = x[:, 0], x[:, 1]
        r0s = r0 + rs
        z0s = z0 - rs
        dist = tc.zeros_like(r, device=device, dtype=TORCH_FLOAT)
        z_abs = tc.abs(z)
        area1 = (z_abs < z0s) & (r < r0)  # In pore
        area2 = (r > r0s) & (z_abs > z0)  # In bulk
        area3 = (z_abs >= z0s) & (r <= r0s)  # In pore-bulk
        dist[area1] = r0 - r[area1]
        dist[area2] = z_abs[area2] - z0
        dist[area3] = tc.sqrt((z_abs[area3] - z0s) ** 2 + (r[area3] - r0s) ** 2)
        dist[area3] -= rs
        dist[dist <= 0] = 0
        return dist.reshape(-1, 1)

    # hyd = HydrationPotential(r0=25.16, z0=25, rs=2, ion_type="pot")

    grid = Grid(grid_width=0.5, r=[0, 50], z=[-100, 100])
    res = hyd(grid, dist_fun, require_first=True, require_second=True)
    energy = res[0]
    d_energy_dx = res[1]
    d_energy_dx2 = res[2]

    half_index = grid.shape[1] // 2
    # plt.plot(grid.coordinate.r[:, half_index].get(), energy[:, half_index].get(), ".-")
    # plt.plot(grid.coordinate.z[50, :].get(), energy[50, :].get(), ".-")
    c = plt.contour(
        grid.coordinate.r.get(),
        grid.coordinate.z.get(),
        d_energy_dx2[1].get(),
        200,
    )
    plt.colorbar(c)
    plt.show()

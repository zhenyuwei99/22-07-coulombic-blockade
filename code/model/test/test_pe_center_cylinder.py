#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_pe_center_cylinder.py
created time : 2023/02/23
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""


import os
import torch as tc
import cupy as cp
from torch.autograd import grad
from mdpy.unit import *
from model import *
from model.solver import PECenterCylinderSolver
from model.core import Grid, GridWriter, GridParser
from model.utils import *
from model.test import PE_CYLINDER_GRID_FILE_PATH, NPE_CYLINDER_GRID_FILE_PATH


def get_phi(grid: Grid):
    phi = grid.empty_variable()
    voltage = (
        Quantity(-10, volt * elementary_charge).convert_to(default_energy_unit).value
    )
    field = (grid.zeros_field() - 1).astype(CUPY_INT)
    value = grid.zeros_field().astype(CUPY_FLOAT)
    direction = grid.zeros_field().astype(CUPY_INT)
    vector = cp.zeros(grid.shape + [2], CUPY_FLOAT)
    # 0: inner;
    # 1: dirichlet;
    # 2: axial-symmetry-boundary;
    # 3: r-no-gradient-boundary
    # Inner
    field[1:-1, 1:-1] = 0
    index = cp.argwhere(field).astype(CUPY_INT)
    # Dirichlet
    field[:, 0] = 1  # down
    value[:, 0] = voltage * -0.5
    field[:, -1] = 1  # up
    value[:, -1] = voltage * 0.5
    # axial symmetry
    field[0, 1:-1] = 2  # left
    # r-no-gradient
    field[-1, 1:-1] = 3  # right
    direction[-1, 1:-1] = -1
    # Register
    index = cp.argwhere(field == 0).astype(CUPY_INT)
    phi.register_points(
        type="inner",
        index=index,
    )
    index = cp.argwhere(field == 1).astype(CUPY_INT)
    phi.register_points(
        type="dirichlet",
        index=index,
        value=value[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 2).astype(CUPY_INT)
    phi.register_points(type="axial-symmetry-boundary", index=index)
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    phi.register_points(
        type="r-no-gradient-boundary",
        index=index,
    )
    return phi


def get_rho(grid: Grid, r0=40, z0=30):
    rho = grid.zeros_field()
    charge = -1 / grid.grid_width**grid.num_dimensions
    dist = (grid.coordinate.r - r0) ** 2 + (grid.coordinate.z - z0) ** 2
    sigma = grid.grid_width
    rho = cp.exp(-dist / (2 * sigma**2)) * charge
    rho = grid.zeros_field()
    return rho


def get_epsilon(grid: Grid, r0, z0, rs):
    x = cp.hstack([grid.coordinate.r.reshape(-1, 1), grid.coordinate.z.reshape(-1, 1)])
    x = tc.tensor(x.get(), requires_grad=True).cuda()
    r0s = r0 + rs
    z0s = z0 - rs
    alpha = reasoning_alpha(2)
    r = x[:, 0]
    z = x[:, 1]
    dist = tc.zeros_like(r)
    z_abs = tc.abs(z)
    area1 = (z_abs < z0s) & (r < r0s)  # In pore
    area2 = (r > r0s) & (z_abs > z0s)  # In bulk
    area3 = (z_abs >= z0s) & (r <= r0s)  # In pore-bulk
    dist[area1] = r0s - r[area1]
    dist[area2] = z_abs[area2] - z0s
    dist[area3] = tc.sqrt((z_abs[area3] - z0s) ** 2 + (r[area3] - r0s) ** 2)
    dist -= rs  # r0, z0 dist=0
    epsilon = 1.0 / (1.0 + tc.exp(-alpha * dist))
    epsilon *= 78.0
    epsilon += 2.0
    depsilon_dx = grad(epsilon.sum(), x)[0].cpu().numpy()
    depsilon_dx[np.isnan(depsilon_dx)] = 0
    epsilon = epsilon.detach().cpu().numpy().reshape(grid.shape)
    depsilon_dr = depsilon_dx[:, 0].reshape(grid.shape)
    depsilon_dz = depsilon_dx[:, 1].reshape(grid.shape)
    return epsilon, depsilon_dr, depsilon_dz


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    r0, z0, rs = 10, 25, 5
    grid = Grid(grid_width=0.5, r=[0, 75], z=[-100, 100])
    epsilon, depsilon_dr, depsilon_dz = get_epsilon(grid, r0, z0, rs)
    print(epsilon)
    solver = PECenterCylinderSolver(grid=grid)
    grid.add_variable("phi", get_phi(grid))
    grid.add_field("rho", get_rho(grid))
    grid.add_field("epsilon", cp.array(epsilon, CUPY_FLOAT))
    grid.add_field("depsilon_dr", cp.array(depsilon_dr, CUPY_FLOAT))
    grid.add_field("depsilon_dz", cp.array(depsilon_dz, CUPY_FLOAT))

    solver.iterate(5000)
    phi = grid.variable.phi.value.get()
    phi = (
        Quantity(phi, default_energy_unit / default_charge_unit).convert_to(volt).value
    )

    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    threshold = 200
    c = ax.contour(
        grid.coordinate.r.get()[1:-1, 1:-1],
        grid.coordinate.z.get()[1:-1, 1:-1],
        phi[1:-1, 1:-1],
        # solver._upwind_direction_z.get()[1:-1, 1:-1],
        200,
        cmap="RdBu",
    )
    fig.colorbar(c)
    plt.show()

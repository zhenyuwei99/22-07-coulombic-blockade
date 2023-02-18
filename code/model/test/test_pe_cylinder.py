#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_pe_cylinder.py
created time : 2023/02/18
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import cupy as cp
from mdpy.unit import *
from model import *
from model.solver import PECylinderSolver
from model.core import Grid, GridWriter
from model.utils import *

PE_CYLINDER_GRID_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "./data/pe_cylinder.grid"
)


def get_phi(grid: Grid):
    phi = grid.empty_variable()
    voltage = (
        Quantity(5, volt * elementary_charge).convert_to(default_energy_unit).value
    )
    field = (grid.zeros_field() - 1).astype(CUPY_INT)
    value = grid.zeros_field().astype(CUPY_FLOAT)
    direction = grid.zeros_field().astype(CUPY_INT)
    # 0: inner; 1: dirichlet; 2: axial-symmetry;
    # 3: r-no-gradient; 4: z-no-gradient
    # Inner
    field[1:-1, 1:-1] = 0
    index = cp.argwhere(field).astype(CUPY_INT)
    # Dirichlet
    field[:, 0] = 1  # down
    value[:, 0] = voltage * -0.5
    field[:, -1] = 1  # up
    value[:, -1] = voltage * 0.5
    # r-no-gradient
    field[-1, 1:-1] = 3  # right
    direction[-1, 1:-1] = -1
    # axial symmetry
    field[0, 1:-1] = 2  # left
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
    phi.register_points(type="axial-symmetry", index=index)
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    phi.register_points(
        type="r-no-gradient",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 4).astype(CUPY_INT)
    phi.register_points(
        type="z-no-gradient",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    return phi


def get_rho(grid: Grid, r0=2, z0=0):
    rho = grid.zeros_field()
    charge = -1 / grid.grid_width**grid.num_dimensions
    dist = (grid.coordinate.r - r0) ** 2 + (grid.coordinate.z - z0) ** 2
    sigma = grid.grid_width / 3
    rho = cp.exp(-dist / (2 * sigma**2)) * charge
    rho = grid.zeros_field()
    return rho


def get_epsilon(grid: Grid, r0, z0):
    dist = get_pore_distance_cylinder(grid.coordinate.r, grid.coordinate.z, r0, z0, 5)
    alpha = reasoning_alpha(5)
    epsilon = CUPY_FLOAT(1) / (1 + cp.exp(-alpha * dist))
    epsilon -= 0.5
    epsilon *= 2
    epsilon *= CUPY_FLOAT(78)
    epsilon += CUPY_FLOAT(2)
    epsilon[dist == -1] = 2

    epsilon = grid.ones_field() * CUPY_FLOAT(78)
    epsilon[(grid.coordinate.r >= r0) & (cp.abs(grid.coordinate.z) <= z0)] = 2
    return epsilon.astype(CUPY_FLOAT)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    r0, z0 = 5, 25
    grid = Grid(grid_width=0.25, r=[0, 50], z=[-100, 100])
    solver = PECylinderSolver(grid=grid)
    grid.add_variable("phi", get_phi(grid))
    grid.add_field("rho", get_rho(grid))
    grid.add_field("epsilon", get_epsilon(grid, r0, z0))

    s = time.time()
    solver.iterate(5000)
    phi = grid.variable.phi.value.get()
    phi = (
        Quantity(phi, default_energy_unit / default_charge_unit).convert_to(volt).value
    )
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    writer = GridWriter(PE_CYLINDER_GRID_FILE_PATH)
    writer.write(grid)

    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    threshold = 200
    phi[phi >= threshold] = threshold
    phi[phi <= -threshold] = -threshold
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

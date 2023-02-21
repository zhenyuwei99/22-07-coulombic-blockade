#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_npe_cylinder.py
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
from model.utils import *
from model.solver import NPECylinderSolver
from model.core import Grid, GridParser, GridWriter
from model.test import PE_CYLINDER_GRID_FILE_PATH, NPE_CYLINDER_GRID_FILE_PATH


def get_rho(grid: Grid, dist, unit_vec):
    density = Quantity(0.15, mol / decimeter**3) * NA
    density = check_quantity_value(density, 1 / default_length_unit**3)
    rho = grid.empty_variable()
    field = grid.zeros_field().astype(CUPY_INT) - 1
    value = grid.zeros_field().astype(CUPY_FLOAT)
    direction = grid.zeros_field().astype(CUPY_INT)
    vector = cp.zeros(grid.shape + [2], CUPY_FLOAT)
    # 0: inner; 1: dirichlet; 3: axial-symmetry; 4: r-no-flux
    # Inner
    field[1:-1, 1:-1] = 0
    # dirichlet
    field[:, [0, -1]] = 1
    value[:, 0] = density
    value[:, -1] = density
    # axial-symmetry
    field[0, 1:-1] = 2
    # no-flux
    index = dist == 0
    field[index] = 4
    vector[index, 0] = unit_vec[index, 0]
    vector[index, 1] = unit_vec[index, 1]
    # r-no-flux
    field[-1, 1:-1] = 3
    direction[-1, 1:-1] = -1

    index = dist == -1
    field[index] = 1
    value[index] = 0

    # Register
    index = cp.argwhere(field == 0).astype(CUPY_INT)
    rho.register_points(
        type="inner",
        index=index,
    )
    index = cp.argwhere(field == 1).astype(CUPY_INT)
    rho.register_points(
        type="dirichlet",
        index=index,
        value=value[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 2).astype(CUPY_INT)
    rho.register_points(type="axial-symmetry", index=index)
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    rho.register_points(
        type="r-no-flux",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 4).astype(CUPY_INT)
    rho.register_points(
        type="no-flux-inner",
        index=index,
        unit_vec=vector[index[:, 0], index[:, 1]],
    )
    return rho


def get_u():
    grid = GridParser(PE_CYLINDER_GRID_FILE_PATH).grid
    # return grid.zeros_field(CUPY_FLOAT)
    return -grid.variable.phi.value


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    r0, z0, rs = 5, 25, 5
    beta = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    beta = 1 / beta

    grid = Grid(grid_width=0.5, r=[0, 150], z=[-100, 100])
    dist, unit_vec = get_pore_distance_and_vector(
        grid.coordinate.r, grid.coordinate.z, r0, z0, rs
    )
    solver = NPECylinderSolver(grid=grid, ion_type="k")
    grid.add_variable("rho_k", get_rho(grid, dist, unit_vec))
    grid.add_field("u_k", get_u())
    grid.add_constant("beta", beta)

    s = time.time()
    solver.iterate(10000)
    print(solver.residual)
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    writer = GridWriter(NPE_CYLINDER_GRID_FILE_PATH)
    writer.write(grid)

    rho = grid.variable.rho_k.value.get()
    rho = (
        (Quantity(rho, 1 / default_length_unit**3) / NA)
        .convert_to(mol / decimeter**3)
        .value
    )

    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    c = ax.contour(
        grid.coordinate.r.get(),
        grid.coordinate.z.get(),
        rho,
        20,
    )
    fig.colorbar(c)
    plt.show()

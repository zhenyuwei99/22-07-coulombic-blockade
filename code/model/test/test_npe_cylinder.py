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
from model.solver import NPECylinderSolver
from model.core import Grid, GridParser, GridWriter
from model.test import PE_CYLINDER_GRID_FILE_PATH, NPE_CYLINDER_GRID_FILE_PATH


def get_rho(grid: Grid, r0, z0):
    density = Quantity(0.15, mol / decimeter**3) * NA
    density = check_quantity_value(density, 1 / default_length_unit**3)
    rho = grid.empty_variable()
    field = grid.zeros_field().astype(CUPY_INT) - 1
    value = grid.zeros_field().astype(CUPY_FLOAT)
    direction = grid.zeros_field().astype(CUPY_INT)
    # 0: inner; 1: dirichlet; 3: axial-symmetry; 4: r-no-flux
    # Inner
    field[1:-1, 1:-1] = 0
    # dirichlet
    field[:, [0, -1]] = 1
    value[:, 0] = density
    value[:, -1] = density
    # r-no-flux
    field[-1, 1:-1] = 3
    direction[-1, 1:-1] = -1
    # axial-symmetry
    field[0, 1:-1] = 2

    index = (grid.coordinate.r >= r0) & (cp.abs(grid.coordinate.z) <= z0)
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
    return rho


def get_u():
    grid = GridParser(PE_CYLINDER_GRID_FILE_PATH).grid
    # return grid.zeros_field(CUPY_FLOAT)
    return grid.variable.phi.value


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    r0, z0 = 5, 25
    beta = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    beta = 1 / beta

    grid = Grid(grid_width=0.5, r=[0, 50], z=[-100, 100])
    solver = NPECylinderSolver(grid=grid, ion_type="k")
    grid.add_variable("rho_k", get_rho(grid, r0, z0))
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

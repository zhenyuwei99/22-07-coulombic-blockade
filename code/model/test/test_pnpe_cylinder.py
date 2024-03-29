#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_pnpe_cylinder.py
created time : 2023/02/19
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import cupy as cp
from mdpy.utils import check_quantity_value, check_quantity
from mdpy.unit import *

from model import *
from model.core import Grid, GridWriter
from model.utils import *
from model.solver import PNPECylinderSolver
from model.solver.utils import *


def get_phi(grid: Grid, voltage, dist, unit_vec):
    voltage = check_quantity_value(voltage, volt)
    voltage = (
        Quantity(voltage, volt * elementary_charge)
        .convert_to(default_energy_unit)
        .value
    )
    phi = grid.empty_variable()
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
    # no-gradient inner
    index = dist == 0
    index[-1, :] = False
    field[index] = 5  # right
    vector[index, 0] = unit_vec[index, 0]
    vector[index, 1] = unit_vec[index, 1]
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
    # index = cp.argwhere(field == 4).astype(CUPY_INT)
    # phi.register_points(
    #     type="z-no-gradient",
    #     index=index,
    #     direction=direction[index[:, 0], index[:, 1]],
    # )
    index = cp.argwhere(field == 5).astype(CUPY_INT)
    phi.register_points(
        type="no-gradient-inner",
        index=index,
        unit_vector=vector[index[:, 0], index[:, 1]],
    )
    return phi


def get_rho(grid: Grid, ion_type, density, dist, vector):
    density = check_quantity(density, mol / decimeter**3) * NA
    density = check_quantity_value(density, 1 / default_length_unit**3)
    rho = grid.empty_variable()
    field = grid.zeros_field().astype(CUPY_INT) - 1
    value = grid.zeros_field().astype(CUPY_FLOAT)
    direction = grid.zeros_field().astype(CUPY_INT)
    unit_vec = cp.zeros(grid.shape + [2], CUPY_FLOAT)
    # 0: inner; 1: dirichlet; 2: axial-symmetry; 3: no-flux-inner;
    # 4: r-no-flux; 5: z-no-gradient
    r = grid.coordinate.r
    z = grid.coordinate.z
    index = cp.argwhere((r > r0) & (z < z0) & (z > -z0))
    # Inner
    field[1:-1, 1:-1] = 0

    # stern
    r_stern = 2.0
    # index = dist == r_stern
    # field[index] = 3
    # unit_vec[index, 0] = -vector[index, 0]
    # unit_vec[index, 1] = -vector[index, 1]
    # index = (dist < r_stern) & (dist >= 0)
    # field[index] = 1
    # value[index] = density * 0.5
    # value[index] = (density * 0.5 / r_stern) * dist[index]

    # no-flux
    index = dist == 0
    field[index] = 3
    unit_vec[index, 0] = vector[index, 0]
    unit_vec[index, 1] = vector[index, 1]

    # dirichlet and no gradient
    if not True:
        val = ION_DICT[ion_type]["val"].value
        if val < 0:
            dirichlet_index = 0
            no_gradient_index = -1
        else:
            dirichlet_index = -1
            no_gradient_index = 0
        field[:, dirichlet_index] = 1
        value[:, dirichlet_index] = density
        field[:, no_gradient_index] = 5
        direction[:, no_gradient_index] = 1 if no_gradient_index == 0 else -1
    else:
        field[:, [0, -1]] = 1
        value[:, [0, -1]] = density
    # axial-symmetry
    field[0, 1:-1] = 2
    direction[0, 1:-1] = 1

    # r-no-flux
    if True:
        field[-1, 1:-1] = 4
        direction[-1, 1:-1] = -1
    else:
        index = cp.argwhere(dist == 0)[:, 1]
        z_min, z_max = index.min(), index.max()
        # field[-1, 1 : z_min + 1] = 1
        # value[-1, 1 : z_min + 1] = density
        # field[-1, z_max:-1] = 1
        # value[-1, z_max:-1] = density

        field[-1, 1 : z_min + 1] = 6
        direction[-1, 1 : z_min + 1] = 1
        field[-1, z_max:-1] = 6
        direction[-1, z_max:-1] = -1

    # Dirichlet
    index = dist == -1
    field[index] = 1
    value[index] = 0

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
        type="no-flux-inner",
        index=index,
        unit_vec=unit_vec[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 4).astype(CUPY_INT)
    rho.register_points(
        type="r-no-flux",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    # index = cp.argwhere(field == 5).astype(CUPY_INT)
    # rho.register_points(
    #     type="z-no-gradient",
    #     index=index,
    #     direction=direction[index[:, 0], index[:, 1]],
    # )
    index = cp.argwhere(field == 6).astype(CUPY_INT)
    rho.register_points(
        type="z-no-flux",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    # index = cp.argwhere(field == 7).astype(CUPY_INT)
    # rho.register_points(
    #     type="stern",
    #     index=index,
    #     density=value[index[:, 0], index[:, 1]],
    # )
    return rho


def get_epsilon(grid: Grid, dist):
    epsilon = grid.ones_field() * CUPY_FLOAT(78)
    epsilon[dist == -1] = 2
    return epsilon.astype(CUPY_FLOAT)


def get_rho_fixed(grid: Grid, r0=16, z0=0):
    rho = grid.zeros_field()
    charge = 1 / grid.grid_width**grid.num_dimensions
    dist = (grid.coordinate.r - r0) ** 2 + (grid.coordinate.z - z0) ** 2
    sigma = grid.grid_width
    rho = cp.exp(-dist / (2 * sigma**2)) * charge
    rho = grid.zeros_field()
    return rho


if __name__ == "__main__":
    r0, z0, rs = 15, 25, 2.5
    voltage = Quantity(1.5, volt)
    density = Quantity(0.15, mol / decimeter**3)
    beta = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    beta = 1 / beta
    ion_types = ["cl"]
    ion_types = ["k", "cl"]
    grid = Grid(grid_width=0.5, r=[0, 150], z=[-100, 100])
    dist, vector = get_pore_distance_and_vector(
        grid.coordinate.r, grid.coordinate.z, r0, z0, rs
    )

    solver = PNPECylinderSolver(grid=grid, ion_types=ion_types)
    solver.npe_solver_list[0].is_inverse = True
    grid.add_variable("phi", get_phi(grid, voltage=voltage, dist=dist, unit_vec=vector))
    grid.add_field("epsilon", get_epsilon(grid, dist))
    grid.add_field("rho", grid.zeros_field(CUPY_FLOAT))
    grid.add_field("rho_fixed", get_rho_fixed(grid))
    grid.add_field("u_s", grid.zeros_field(CUPY_FLOAT))
    for ion_type in ion_types:
        grid.add_variable(
            "rho_%s" % ion_type, get_rho(grid, ion_type, density, dist, vector)
        )
        grid.add_field("u_%s" % ion_type, grid.zeros_field(CUPY_FLOAT))
    grid.add_constant("beta", beta)

    solver.iterate(5, 5000, is_restart=True)
    # solver.iterate(10, 2000, is_restart=True)
    # solver.iterate(10, 1000, is_restart=True)
    # visualize_concentration(grid, ion_types=ion_types, name="rho-test")
    # visualize_flux(grid, pnpe_solver=solver, ion_types=ion_types, name="flux-test")

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = check_dir(os.path.join(cur_dir, "./out"))
    writer = GridWriter(os.path.join(out_dir, "pnpe_cylinder.grid"))
    writer.write(grid)
    visualize_concentration(grid, ion_types=ion_types, is_save=False)
    plt.show()
    # for i in range(100):
    #     print("Iteration", i)
    #     solver.iterate(10, 5000, is_restart=True)
    #     visualize_concentration(grid, ion_types=ion_types, iteration=i)
    #     visualize_flux(grid, pnpe_solver=solver, ion_types=ion_types, iteration=i)

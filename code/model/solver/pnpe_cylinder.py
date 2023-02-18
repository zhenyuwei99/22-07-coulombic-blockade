#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pnpe_cylinder.py
created time : 2023/01/27
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np
import cupy as cp
from mdpy.utils import check_quantity_value, check_quantity
from mdpy.unit import *

from model import *
from model.core import Grid, GridWriter
from model.utils import *
from model.solver.utils import *
from model.solver.pe_cylinder import PECylinderSolver
from model.solver.npe_cylinder import NPECylinderSolver


class PNPECylinderSolver:
    def __init__(self, grid: Grid, ion_types: list[str]) -> None:
        """All grid and constant in default unit
        ### Variable:
        - phi: Electric potential
            - inner: Inner points
            - dirichlet: Dirichlet point
                - `index`, `value` required
            - no-gradient: dphi/dz = 0
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor
        - rho_[ion]: Number density of [ion]
            - inner: Inner points
            - dirichlet: Dirichlet point, constant density
                - `index`, `value` required
            - no-flux: no flux boundary
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor

        ### Field:
        - epsilon: Relative permittivity
        - rho: Fixed charge density
        - u_[ion]: External potential of [ion]
        - u_s: Steric potential

        ### Constant:
        - beta: 1/kBT
        - epsilon0 (added): Vacuum permittivity
        - r_[ion] (added): Radius of [ion]
        - z_[ion] (added): Valence of [ion]
        - d_[ion] (added): Diffusion coefficient of [ion]
        """
        # Input
        self._grid = grid
        self._ion_types = ion_types
        self._grid.add_requirement("field", "u_s")
        for ion_type in ion_types:
            self._grid.add_requirement("constant", "r_%s" % ion_type)
            self._grid.add_requirement("constant", "z_%s" % ion_type)
            # Radius
            self._grid.add_constant(
                "r_%s" % ion_type,
                check_quantity_value(VDW_DICT[ion_type]["sigma"], default_length_unit),
            )
            # Valence
            self._grid.add_constant(
                "z_%s" % ion_type,
                check_quantity_value(ION_DICT[ion_type]["val"], VAL_UNIT),
            )
        # Create solver
        self._pe_solver = PECylinderSolver(grid=self._grid)
        self._npe_solver_list = [
            NPECylinderSolver(self._grid, ion_type=i) for i in self._ion_types
        ]
        # Create res list
        self._pre_res = []

    def _update_rho(self):
        self._grid.field.rho = self._grid.zeros_field(CUPY_FLOAT)
        for ion_type in self._ion_types:
            z = getattr(self._grid.constant, "z_%s" % ion_type)  # valence
            self._grid.field.rho += (
                CUPY_FLOAT(z) * getattr(self.grid.variable, "rho_%s" % ion_type).value
            )

    def _update_u_s(self):
        self._grid.field.u_s = self._grid.zeros_field(CUPY_FLOAT)
        for ion_type in self._ion_types:
            r = getattr(self._grid.constant, "r_%s" % ion_type)
            v = CUPY_FLOAT(4 / 3 * np.pi * r**3)
            self._grid.field.u_s -= (
                v * getattr(self._grid.variable, "rho_%s" % ion_type).value
            )
        self._grid.field.u_s[self._grid.field.u_s <= 1e-5] = 1e-5
        self._grid.field.u_s[self._grid.field.u_s >= 1] = 1
        self._grid.field.u_s = -cp.log(self._grid.field.u_s)
        self._grid.field.u_s *= CUPY_FLOAT(1 / self.grid.constant.beta)

    def _update_u_ion(self, ion_type: str):
        z = getattr(self._grid.constant, "z_%s" % ion_type)  # valence
        u = self._grid.zeros_field(CUPY_FLOAT)
        # Electric energy
        u += CUPY_FLOAT(z) * self._grid.variable.phi.value
        # Steric energy
        # u += self._grid.field.u_s
        setattr(self._grid.field, "u_%s" % ion_type, u.astype(CUPY_FLOAT))

    def iterate(self, num_iterations, num_sub_iterations=100, is_restart=False):
        self._grid.check_requirement()
        for iterations in range(num_iterations):
            self._pre_res = []
            self._pre_res.append(self._grid.variable.phi.value)
            for index, ion_type in enumerate(self._ion_types):
                self._update_rho()
                self._update_u_s()
                self._pe_solver.iterate(
                    num_iterations=num_sub_iterations, is_restart=is_restart
                )
                self._update_u_ion(ion_type=ion_type)
                self._pre_res.append(
                    getattr(self._grid.variable, "rho_%s" % ion_type).value
                )
                self._npe_solver_list[index].iterate(
                    num_iterations=num_sub_iterations, is_restart=is_restart
                )
            print(self.residual)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def ion_types(self) -> list[str]:
        return self._ion_types

    @property
    def pe_solver(self) -> PECylinderSolver:
        return self._pe_solver

    @property
    def npe_solver_list(self) -> list[NPECylinderSolver]:
        return self._npe_solver_list

    @property
    def residual(self):
        residual = 0
        self._cur_res = [self._grid.variable.phi.value] + [
            getattr(self._grid.variable, "rho_%s" % ion_type).value
            for ion_type in self._ion_types
        ]
        for i, j in zip(self._pre_res, self._cur_res):
            residual += cp.abs(i - j).mean()
        return residual


def get_phi(grid: Grid, voltage):
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

    # no-flux
    index = dist == 0
    field[index] = 3
    unit_vec[index, 0] = vector[index, 0]
    unit_vec[index, 1] = vector[index, 1]
    # stern
    # r_stern = 2.0
    # index = dist == r_stern
    # field[index] = 3
    # unit_vec[index, 0] = -vector[index, 0]
    # unit_vec[index, 1] = -vector[index, 1]
    # index = (dist < r_stern) & (dist >= 0)
    # field[index] = 1
    # value[index] = density * 0.5
    # value[index] = (density * 0.5 / r_stern) * dist[index]

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
    # Dirichlet
    index = dist == -1
    field[index] = 1
    value[index] = 0

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
    index = cp.argwhere(field == 5).astype(CUPY_INT)
    rho.register_points(
        type="z-no-gradient",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 6).astype(CUPY_INT)
    rho.register_points(
        type="z-no-flux",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 7).astype(CUPY_INT)
    rho.register_points(
        type="stern",
        index=index,
        density=value[index[:, 0], index[:, 1]],
    )
    return rho


def get_epsilon(grid: Grid, dist):
    epsilon = grid.ones_field() * CUPY_FLOAT(78)
    epsilon[dist == -1] = 2
    return epsilon.astype(CUPY_FLOAT)


if __name__ == "__main__":
    r0, z0, rs = 15, 25, 2.5
    voltage = Quantity(5.0, volt)
    density = Quantity(0.15, mol / decimeter**3)
    beta = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    beta = 1 / beta
    ion_types = ["cl", "k"]
    grid = Grid(grid_width=0.5, r=[0, 75], z=[-150, 150])
    dist, vector = get_pore_distance_and_vector(
        grid.coordinate.r, grid.coordinate.z, r0, z0, rs
    )

    solver = PNPECylinderSolver(grid=grid, ion_types=ion_types)
    solver.npe_solver_list[0].is_inverse = True
    grid.add_variable("phi", get_phi(grid, voltage=voltage))
    grid.add_field("epsilon", get_epsilon(grid, dist))
    grid.add_field("rho", grid.zeros_field(CUPY_FLOAT))
    grid.add_field("u_s", grid.zeros_field(CUPY_FLOAT))
    for ion_type in ion_types:
        grid.add_variable(
            "rho_%s" % ion_type, get_rho(grid, ion_type, density, dist, vector)
        )
        grid.add_field("u_%s" % ion_type, grid.zeros_field(CUPY_FLOAT))
    grid.add_constant("beta", beta)

    solver.iterate(10, 2000, is_restart=True)
    visualize_concentration(grid, ion_types=ion_types, name="rho-test")
    visualize_flux(grid, pnpe_solver=solver, ion_types=ion_types, name="flux-test")

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = check_dir(os.path.join(cur_dir, "../out/solver/pnpe"))
    writer = GridWriter(os.path.join(out_dir, "test.grid"))
    writer.write(grid)
    # for i in range(100):
    #     print("Iteration", i)
    #     solver.iterate(10, 5000, is_restart=True)
    #     visualize_concentration(grid, ion_types=ion_types, iteration=i)
    #     visualize_flux(grid, pnpe_solver=solver, ion_types=ion_types, iteration=i)

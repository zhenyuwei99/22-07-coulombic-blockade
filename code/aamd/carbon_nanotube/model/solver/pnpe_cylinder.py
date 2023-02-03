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
import sys
import numpy as np
import cupy as cp
import numba.cuda as cuda
import mdpy as md
from mdpy.core import Grid
from mdpy.utils import check_quantity_value, check_quantity
from mdpy.environment import *
from mdpy.unit import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import *
from hydration import *
from analysis_cylinder import *
from pe_cylinder import PECylinderSolver
from npe_cylinder import NPECylinderSolver


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
        u += self._grid.field.u_s
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


def get_distance_and_vector(r, z, r0, z0, rs):
    r0s = r0 + rs
    z0s = z0 - rs
    r = grid.coordinate.r
    z = grid.coordinate.z
    dist = grid.zeros_field(CUPY_FLOAT) - 1
    vector = cp.zeros(grid.shape + [2], CUPY_FLOAT)
    # In pore
    index = (cp.abs(z) <= z0s) & (r <= r0)
    dist[index] = r0 - r[index]
    vector[index, 0] = 1
    vector[index, 1] = 0
    # Out pore
    index = (z >= z0) & (r >= r0s)
    dist[index] = z[index] - z0
    vector[index, 0] = 0
    vector[index, 1] = -1
    index = (z <= -z0) & (r >= r0s)
    dist[index] = -(z[index] + z0)
    vector[index, 0] = 0
    vector[index, 1] = 1
    # Sphere part
    index = (z > z0s) & (r < r0s)
    temp = cp.sqrt((z[index] - z0s) ** 2 + (r[index] - r0s) ** 2) - rs
    temp[temp < 0] = -1
    dist[index] = temp
    vector[index, 0] = z[index] - z0s
    vector[index, 1] = r[index] - r0s
    index = (z < -z0s) & (r < r0s)
    temp = cp.sqrt((z[index] + z0s) ** 2 + (r[index] - r0s) ** 2) - rs
    temp[temp < 0] = -1
    dist[index] = temp
    vector[index, 0] = z[index] + z0s
    vector[index, 1] = r[index] - r0s
    # Norm
    norm = cp.sqrt(vector[:, :, 0] ** 2 + vector[:, :, 1] ** 2)
    vector[:, :, 0] /= norm
    vector[:, :, 1] /= norm
    return dist.astype(CUPY_FLOAT), vector.astype(CUPY_FLOAT)


def get_phi(grid: Grid, voltage):
    phi = grid.empty_variable()
    voltage = check_quantity_value(voltage, volt)
    voltage = (
        Quantity(voltage, volt * elementary_charge)
        .convert_to(default_energy_unit)
        .value
    )
    field = (grid.zeros_field() - 1).astype(CUPY_INT)
    value = grid.zeros_field().astype(CUPY_FLOAT)
    dimension = grid.zeros_field().astype(CUPY_INT)
    direction = grid.zeros_field().astype(CUPY_INT)
    # 0: inner; 1: dirichlet; 2: no-gradient; 3: axial-symmetry
    # Inner
    field[1:-1, 1:-1] = 0
    index = cp.argwhere(field).astype(CUPY_INT)
    # Dirichlet
    field[:, 0] = 1  # down
    value[:, 0] = voltage * -0.5
    field[:, -1] = 1  # up
    value[:, -1] = voltage * 0.5
    # no-gradient
    field[-1, 1:-1] = 2  # right
    dimension[-1, 1:-1] = 0
    direction[-1, 1:-1] = -1
    # axial symmetry
    field[0, 1:-1] = 3  # left
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
    phi.register_points(
        type="no-gradient",
        index=index,
        dimension=dimension[index[:, 0], index[:, 1]],
        direction=direction[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    phi.register_points(type="axial-symmetry", index=index)
    return phi


def get_rho(grid: Grid, density, dist, vector):
    density = check_quantity(density, mol / decimeter**3) * NA
    density = check_quantity_value(density, 1 / default_length_unit**3)
    rho = grid.empty_variable()
    field = grid.zeros_field().astype(CUPY_INT) - 1
    value = grid.zeros_field().astype(CUPY_FLOAT)
    direction = grid.zeros_field().astype(CUPY_INT)
    unit_vec = cp.zeros(grid.shape + [2], CUPY_FLOAT)
    # 0: inner; 1: dirichlet; 2: axial-symmetry; 3: z-no-flux;
    # 4: r-no-flux; 5: no-flux; 6: r-no-flux-inner
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

    # dirichlet
    field[:, [0, -1]] = 1
    value[:, [0, -1]] = density
    index = dist == -1
    field[index] = 1
    value[index] = 0

    # axial-symmetry
    field[0, 1:-1] = 2
    direction[0, 1:-1] = 1

    # r-no-flux
    field[-1, 1:-1] = 4
    direction[-1, 1:-1] = -1

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
    return rho


def get_epsilon(grid: Grid, dist):
    epsilon = grid.ones_field() * CUPY_FLOAT(78)
    epsilon[dist == -1] = 2
    return epsilon.astype(CUPY_FLOAT)


if __name__ == "__main__":
    r0, z0, rs = 8, 25, 5
    voltage = Quantity(1.0, volt)
    density = Quantity(0.15, mol / decimeter**3)
    beta = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    beta = 1 / beta
    ion_types = ["cl", "k"]
    grid = Grid(grid_width=0.5, r=[0, 50], z=[-150, 150])
    dist, vector = get_distance_and_vector(
        grid.coordinate.r, grid.coordinate.z, r0, z0, rs
    )

    solver = PNPECylinderSolver(grid=grid, ion_types=ion_types)
    solver.npe_solver_list[0].is_inverse = True
    grid.add_variable("phi", get_phi(grid, voltage=voltage))
    grid.add_field("epsilon", get_epsilon(grid, dist))
    grid.add_field("rho", grid.zeros_field(CUPY_FLOAT))
    grid.add_field("u_s", grid.zeros_field(CUPY_FLOAT))
    for ion_type in ion_types:
        grid.add_variable("rho_%s" % ion_type, get_rho(grid, density, dist, vector))
        grid.add_field("u_%s" % ion_type, grid.zeros_field(CUPY_FLOAT))
    grid.add_constant("beta", beta)

    solver.iterate(5, 5000, is_restart=True)
    visualize_concentration(grid, ion_types=ion_types, iteration="test")
    visualize_flux(grid, pnpe_solver=solver, ion_types=ion_types, iteration="test")
    # for i in range(100):
    #     print("Iteration", i)
    #     solver.iterate(10, 5000, is_restart=True)
    #     visualize_concentration(grid, ion_types=ion_types, iteration=i)
    #     visualize_flux(grid, pnpe_solver=solver, ion_types=ion_types, iteration=i)

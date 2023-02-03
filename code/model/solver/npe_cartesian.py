#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : npe_cartesian.py
created time : 2023/01/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import numpy as np
import cupy as cp
import numba.cuda as cuda
from mdpy.core import Grid
from mdpy.utils import check_quantity_value
from mdpy.environment import *
from mdpy.unit import *

from model import *


class NPECartesianSolver:
    def __init__(self, grid: Grid, ion_type: str) -> None:
        """All grid and constant in default unit
        ### Variable:
        - rho_[ion]: Number density of [ion]
            - dirichlet: Dirichlet boundary condition
                - index, value required
            - no-flux: No flux boundary condition
                - self_index, direction, value required

        ### Field:
        - u_[ion]: External potential of [ion]

        ### Constant:
        - beta: 1/kBT
        - r_[ion] (added): Radius of [ion]
        - z_[ion] (added): Valence of [ion]
        - d_[ion] (added): Diffusion coefficient of [ion]
        """
        self._grid = grid
        if not ion_type in ION_DICT.keys():
            raise KeyError("Ion %s is not supported" % ion_type)
        self._ion_type = ion_type
        # Requirement
        self._grid.add_requirement("variable", "rho_%s" % self._ion_type)
        self._grid.add_requirement("field", "u_%s" % self._ion_type)
        self._grid.add_requirement("constant", "beta")
        self._grid.add_requirement("constant", "r_%s" % self._ion_type)
        self._grid.add_requirement("constant", "z_%s" % self._ion_type)
        self._grid.add_requirement("constant", "d_%s" % self._ion_type)
        # Radius
        self._grid.add_constant(
            "r_%s" % self._ion_type,
            check_quantity_value(VDW_DICT[ion_type]["sigma"], default_length_unit),
        )
        # Diffusion coefficient
        self._grid.add_constant(
            "d_%s" % ion_type,
            check_quantity_value(ION_DICT[ion_type]["d"], DIFFUSION_UNIT),
        )
        # Valence
        self._grid.add_constant(
            "z_%s" % self._ion_type,
            check_quantity_value(ION_DICT[ion_type]["val"], VAL_UNIT),
        )

    def _get_delta_u(self):
        # 1: for plus and :-1 for minus
        shape = [self._grid.num_dimensions] + [i + 1 for i in self._grid.inner_shape]
        delta_u = cp.zeros(shape, CUPY_FLOAT)
        u = getattr(self._grid.field, "u_%s" % self._ion_type)
        # X
        delta_u[0, :, :-1, :-1] = u[1:, 1:-1, 1:-1] - u[:-1, 1:-1, 1:-1]
        # Y
        delta_u[1, :-1, :, :-1] = u[1:-1, 1:, 1:-1] - u[1:-1, :-1, 1:-1]
        # Z
        delta_u[2, :-1, :-1, :] = u[1:-1, 1:-1, 1:] - u[1:-1, 1:-1, :-1]
        delta_u *= CUPY_FLOAT(self._grid.constant.beta)
        delta_u[(delta_u < 1e-5) & (delta_u > 0)] = 1e-5
        delta_u[(delta_u > -1e-5) & (delta_u <= 0)] = -1e-5
        return delta_u.astype(CUPY_FLOAT)

    def _get_coefficient(self, delta_u):
        # Pre factor and inv_denominator
        factor = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        inv_denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        # Factor
        # f(x) = x / exp(x) - 1
        # f(-u_i) = u_i - f(u_i)
        exp_delta_u = delta_u / (cp.exp(delta_u) - CUPY_FLOAT(1))
        factor[0, 0] = exp_delta_u[0, 1:, :-1, :-1]
        factor[0, 1] = exp_delta_u[0, :-1, :-1, :-1]
        factor[1, 0] = exp_delta_u[1, :-1, 1:, :-1]
        factor[1, 1] = exp_delta_u[1, :-1, :-1, :-1]
        factor[2, 0] = exp_delta_u[2, :-1, :-1, 1:]
        factor[2, 1] = exp_delta_u[2, :-1, :-1, :-1]
        # X
        inv_denominator += factor[0, 0]
        inv_denominator += delta_u[0, :-1, :-1, :-1] + factor[0, 1]
        factor[0, 0] += delta_u[0, 1:, :-1, :-1]
        # Y
        inv_denominator += factor[1, 0]
        inv_denominator += delta_u[1, :-1, :-1, :-1] + factor[1, 1]
        factor[1, 0] += delta_u[1, :-1, 1:, :-1]
        # Z
        inv_denominator += factor[2, 0]
        inv_denominator += delta_u[2, :-1, :-1, :-1] + factor[2, 1]
        factor[1, 0] += delta_u[2, :-1, :-1, 1:]
        return factor, inv_denominator

    def _update_boundary_point(self, rho, u):
        for boundary_type, boundary_data in rho.boundary.items():
            boundary_type = boundary_type.lower()
            if boundary_type == "dirichlet":
                index = boundary_data["index"]
                value = boundary_data["value"]
                rho.value[index[:, 0], index[:, 1], index[:, 2]] = value
            elif boundary_type == "no-flux":
                self_index = boundary_data["self_index"]
                neighbor_index = boundary_data["neighbor_index"]
                value = boundary_data["value"]
                factor = (neighbor_index - self_index).sum(1).astype(CUPY_FLOAT)
                self_slice = (self_index[:, 0], self_index[:, 1], self_index[:, 2])
                neighbor_slice = (
                    neighbor_index[:, 0],
                    neighbor_index[:, 1],
                    neighbor_index[:, 2],
                )
                # rho_i = exp(du_{i+1}) * u_{i+1}
                # rho_i = exp(-du_{i-1}) * u_{i-1}
                exp = cp.exp(factor * u[neighbor_slice])
                rho.value[self_slice] = rho.value[neighbor_slice] * exp
            else:
                raise KeyError(
                    "Only dirichlet and no-flux boundary condition supported, while %s provided"
                    % boundary_type
                )

    def _update_inner_point(self, rho, factor, inv_denominator, soa_factor):
        factor_a, factor_b = CUPY_FLOAT(soa_factor), CUPY_FLOAT(1 - soa_factor)
        nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        # X
        nominator += factor[0, 0] * rho.value[2:, 1:-1, 1:-1]
        nominator += factor[0, 1] * rho.value[:-2, 1:-1, 1:-1]
        # Y
        nominator += factor[1, 0] * rho.value[1:-1, 2:, 1:-1]
        nominator += factor[1, 1] * rho.value[1:-1, :-2, 1:-1]
        # Z
        nominator += factor[2, 0] * rho.value[1:-1, 1:-1, 2:]
        nominator += factor[2, 1] * rho.value[1:-1, 1:-1, :-2]
        # Update
        new = (
            factor_a * rho.value[1:-1, 1:-1, 1:-1]
            + factor_b * nominator * inv_denominator
        )
        new[new >= NP_DENSITY_UPPER_THRESHOLD] = NP_DENSITY_UPPER_THRESHOLD
        new[new <= NP_DENSITY_LOWER_THRESHOLD] = NP_DENSITY_LOWER_THRESHOLD
        rho.value[1:-1, 1:-1, 1:-1] = new

    def iterate(self, num_iterations, soa_factor=0.01):
        self._grid.check_requirement()
        delta_u = self._get_delta_u()
        factor, inv_denominator = self._get_coefficient(delta_u=delta_u)
        rho = getattr(self._grid.variable, "rho_%s" % self._ion_type)
        u = getattr(self._grid.field, "u_%s" % self._ion_type)
        for iteration in range(num_iterations):
            self._update_boundary_point(rho=rho, u=u)
            self._update_inner_point(
                rho=rho,
                factor=factor,
                inv_denominator=inv_denominator,
                soa_factor=soa_factor,
            )

    def get_flux(self, direction: int):
        rho = getattr(self._grid.variable, "rho_%s" % self._ion_type)
        d = getattr(self._grid.constant, "d_%s" % self._ion_type)
        u_slice = [slice(None, -1) for i in range(self._grid.num_dimensions)]
        u_slice[direction] = slice(1, None)
        u_slice = [direction] + u_slice
        delta_u = self._get_delta_u()[tuple(u_slice)]
        exp_delta_u = cp.exp(delta_u)
        rho_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        rho_slice[direction] = slice(2, None)
        flux = exp_delta_u * rho.value[2:, 1:-1, 1:-1]
        flux -= rho.value[1:-1, 1:-1, 1:-1]
        flux /= exp_delta_u - CUPY_FLOAT(1)
        flux *= delta_u * d
        return flux

    @property
    def ion_type(self) -> str:
        return self._ion_type

    @property
    def grid(self) -> Grid:
        return self._grid


def get_u(grid: Grid, height, mu=0, sigma=2):
    height = check_quantity_value(height, default_energy_unit)
    mu = check_quantity_value(mu, default_length_unit)
    sigma = check_quantity_value(sigma, default_length_unit)
    r = cp.sqrt(
        grid.coordinate.x**2 + grid.coordinate.y**2 + grid.coordinate.z**2
    ).astype(CUPY_FLOAT)
    u = height * cp.exp(-((r - CUPY_FLOAT(mu)) ** 2) / CUPY_FLOAT(2 * sigma**2))
    return u


def get_rho(grid: Grid):
    rho = grid.empty_variable()
    rho.value[1:-1, 1:-1, 1:-1] = 0.001
    boundary_type = "no-flux"
    # boundary_type = "dirichlet"
    boundary_data = {}
    boundary_points = (
        grid.inner_shape[0] * grid.inner_shape[1]
        + grid.inner_shape[0] * grid.inner_shape[2]
        + grid.inner_shape[1] * grid.inner_shape[2]
    ) * 2
    boundary_self_index = cp.zeros([boundary_points, 3], CUPY_INT)
    boundary_neighbor_index = cp.zeros([boundary_points, 3], CUPY_INT)
    field = grid.zeros_field().astype(CUPY_INT)
    num_added_points = 0
    for i in range(3):
        target_slice = [slice(1, -1) for i in range(3)]
        target_slice[i] = [0, -1]
        field[tuple(target_slice)] = 1
        index = cp.argwhere(field).astype(CUPY_INT)
        num_points = index.shape[0]
        boundary_self_index[num_added_points : num_added_points + num_points, :] = index
        field[tuple(target_slice)] = 0
        target_slice[i] = [1, -2]
        field[tuple(target_slice)] = 1
        index = cp.argwhere(field).astype(CUPY_INT)
        boundary_neighbor_index[
            num_added_points : num_added_points + num_points, :
        ] = index
        field[tuple(target_slice)] = 0
        num_added_points += num_points
    boundary_data["value"] = cp.zeros([boundary_self_index.shape[0]], CUPY_FLOAT)
    if boundary_type == "no-flux":
        boundary_data["self_index"] = boundary_self_index
        boundary_data["neighbor_index"] = boundary_neighbor_index
    elif boundary_type == "dirichlet":
        boundary_data["index"] = boundary_self_index
        bulk_rho = (
            (Quantity(0.15, mol / decimeter**3) * NA)
            .convert_to(1 / default_length_unit**3)
            .value
        )
        boundary_data["value"] += bulk_rho
    rho.add_boundary(boundary_type=boundary_type, boundary_data=boundary_data)
    return rho


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    beta = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    beta = 1 / beta
    r0, z0 = 30, 5
    grid = Grid(grid_width=0.25, x=[-10, 10], y=[-10, 10], z=[-10, 10])
    solver = NPECartesianSolver(grid=grid, ion_type="k")
    grid.add_variable("rho_k", get_rho(grid))
    grid.add_field(
        "u_k", get_u(grid, height=Quantity(10.5, kilocalorie_permol), sigma=2)
    )
    grid.add_constant("beta", beta)
    solver.iterate(1000)

    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    half_index = grid.coordinate.x.shape[1] // 2
    target_slice = (
        slice(None, None),
        half_index,
        slice(None, None),
    )

    rho = grid.variable.rho_k.value.get()
    rho = rho = (
        (Quantity(rho, 1 / default_length_unit**3) / NA)
        .convert_to(mol / decimeter**3)
        .value
    )
    flux = np.zeros(grid.shape)
    flux[1:-1, 1:-1, 1:-1] = solver.get_flux(2).get()
    # c = ax.contour(
    #     grid.coordinate.x[target_slice].get(),
    #     grid.coordinate.z[target_slice].get(),
    #     flux[target_slice],
    #     100,
    # )

    flux_x = solver.get_flux(0)
    flux_y = solver.get_flux(1)
    flux_z = solver.get_flux(2)
    net_flux = flux_x[2:, 1:-1, 1:-1] - flux_x[1:-1, 1:-1, 1:-1]
    net_flux += flux_y[1:-1, 2:, 1:-1] - flux_y[1:-1, 1:-1, 1:-1]
    net_flux += flux_z[1:-1, 1:-1, 2:] - flux_z[1:-1, 1:-1, 1:-1]
    for i in [10, 20, 40, -20, -10]:
        half_slice = tuple([i // 2 for i in grid.inner_shape])
        print(net_flux[half_slice], net_flux[1:-1, 1:-1, i].sum())

    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    c = ax.contour(
        grid.coordinate.x[2:-2, half_index, 2:-2].get(),
        grid.coordinate.z[2:-2, half_index, 2:-2].get(),
        net_flux[:, half_index, :].get(),
        100,
    )
    fig.colorbar(c)
    plt.show()

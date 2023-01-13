#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pe_cartesian.py
created time : 2022/11/13
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import cupy as cp
import numba.cuda
from mdpy.core import Grid
from mdpy.environment import *
from mdpy.unit import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydration import get_pore_distance


class PECartesianSolver:
    def __init__(self, grid: Grid) -> None:
        """All grid and constant in default unit
        ### Variable:
        - phi: Electric potential
            - dirichlet: Dirichlet boundary condition
                - index, value required
            - neumann: Neumann boundary condition
                - self_index, target_index, value required
        ### Field:
        - epsilon: Relative permittivity
        - rho: charge density

        ### Constant:
        - epsilon0 (added): Vacuum permittivity
        """
        self._grid = grid
        self._grid.add_requirement("variable", "phi")
        self._grid.add_requirement("field", "epsilon")
        self._grid.add_requirement("field", "rho")
        self._grid.add_requirement("constant", "epsilon0")
        epsilon0 = EPSILON0.convert_to(
            default_charge_unit**2 / (default_energy_unit * default_length_unit)
        ).value
        self._grid.add_constant("epsilon0", epsilon0)

    def _get_coefficient(self):
        pre_factor = NUMPY_FLOAT(0.5 / self._grid.grid_width**2)
        # 1: for plus and :-1 for minus
        shape = [self._grid.num_dimensions] + [i + 1 for i in self._grid.inner_shape]
        factor = cp.zeros(shape, CUPY_FLOAT)
        inv_denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        # X
        factor[0, :, :-1, :-1] = pre_factor * (
            self._grid.field.epsilon[1:, 1:-1, 1:-1]
            + self._grid.field.epsilon[:-1, 1:-1, 1:-1]
        )
        inv_denominator += factor[0, 1:, :-1, :-1] + factor[0, :-1, :-1, :-1]
        # Y
        factor[1, :-1, :, :-1] = pre_factor * (
            self._grid.field.epsilon[1:-1, 1:, 1:-1]
            + self._grid.field.epsilon[1:-1, :-1, 1:-1]
        )
        inv_denominator += factor[1, :-1, 1:, :-1] + factor[1, :-1, :-1, :-1]
        # Z
        factor[2, :-1, :-1, :] = pre_factor * (
            self._grid.field.epsilon[1:-1, 1:-1, 1:]
            + self._grid.field.epsilon[1:-1, 1:-1, :-1]
        )
        inv_denominator += factor[2, :-1, :-1, 1:] + factor[2, :-1, :-1, :-1]
        inv_denominator = CUPY_FLOAT(1) / inv_denominator
        return factor, inv_denominator

    def _update_boundary_point(self, phi):
        for boundary_type, boundary_data in phi.boundary.items():
            boundary_type = boundary_type.lower()
            if boundary_type == "dirichlet":
                index = boundary_data["index"]
                value = boundary_data["value"]
                phi.value[index[:, 0], index[:, 1], index[:, 2]] = value
            elif boundary_type == "neumann":
                self_index = boundary_data["self_index"]
                neighbor_index = boundary_data["neighbor_index"]
                value = boundary_data["value"]
                dist = (neighbor_index - self_index).sum(1).astype(CUPY_FLOAT)
                dist *= self._grid.grid_width
                phi.value[self_index[:, 0], self_index[:, 1], self_index[:, 2]] = (
                    phi.value[
                        neighbor_index[:, 0], neighbor_index[:, 1], neighbor_index[:, 2]
                    ]
                    - dist * value
                )
            else:
                raise KeyError(
                    "Only dirichlet and neumann boundary condition supported, while %s provided"
                    % boundary_type
                )

    def _update_inner_point(self, phi, factor, inv_denominator, scaled_rho, soa_factor):
        factor_a, factor_b = CUPY_FLOAT(soa_factor), CUPY_FLOAT(1 - soa_factor)
        nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        # X Neumann condition
        nominator += factor[0, 1:, :-1, :-1] * phi.value[2:, 1:-1, 1:-1]
        nominator += factor[0, :-1, :-1, :-1] * phi.value[:-2, 1:-1, 1:-1]
        # Y Neumann
        nominator += factor[1, :-1, 1:, :-1] * phi.value[1:-1, 2:, 1:-1]
        nominator += factor[1, :-1, :-1, :-1] * phi.value[1:-1, :-2, 1:-1]
        # Z Neumann
        nominator += factor[2, :-1, :-1, 1:] * phi.value[1:-1, 1:-1, 2:]
        nominator += factor[2, :-1, :-1, :-1] * phi.value[1:-1, 1:-1, :-2]
        # Add charge
        nominator += scaled_rho
        # Update
        phi.value[1:-1, 1:-1, 1:-1] = (
            factor_a * phi.value[1:-1, 1:-1, 1:-1]
            + factor_b * nominator * inv_denominator
        )

    def iterate(self, num_iterations, soa_factor=0.01):
        self._grid.check_requirement()
        factor, inv_denominator = self._get_coefficient()
        inv_epsilon0 = NUMPY_FLOAT(1 / self._grid.constant.epsilon0)
        scaled_rho = self._grid.field.rho[1:-1, 1:-1, 1:-1] * inv_epsilon0
        for iteration in range(num_iterations):
            self._update_boundary_point(self._grid.variable.phi)
            self._update_inner_point(
                phi=self._grid.variable.phi,
                factor=factor,
                inv_denominator=inv_denominator,
                scaled_rho=scaled_rho,
                soa_factor=soa_factor,
            )


def get_rho(grid: Grid):
    rho = grid.zeros_field()
    charge = -1 / grid.grid_width**grid.num_dimensions
    index = [i // 2 for i in grid.coordinate.x.shape]
    index[2] = 5
    num_points = 1 + grid.num_dimensions * 2
    charge /= num_points
    for i in range(grid.num_dimensions):
        index[i] += 1
        rho[tuple(index)] = charge
        index[i] -= 2
        rho[tuple(index)] = charge
        index[i] += 1

    charge = 1 / grid.grid_width**grid.num_dimensions
    index = [i // 2 for i in grid.coordinate.x.shape]
    index[2] = -5
    index[0] += 20
    num_points = 1 + grid.num_dimensions * 2
    charge /= num_points
    for i in range(grid.num_dimensions):
        index[i] += 1
        rho[tuple(index)] += charge
        index[i] -= 2
        rho[tuple(index)] += charge
        index[i] += 1
    return rho


def get_epsilon(grid: Grid, r0, z0):
    dist = get_pore_distance(
        grid.coordinate.x,
        grid.coordinate.y,
        grid.coordinate.z,
        r0=r0,
        z0=z0,
        thickness=0,
    )
    epsilon = cp.zeros_like(dist)
    epsilon[dist <= 0] = 2
    epsilon[dist > 0] = 78
    return epsilon


def get_phi(grid: Grid):
    phi = grid.empty_variable()
    boundary_type = "neumann"
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
    # boundary_data["index"] = boundary_self_index
    boundary_data["self_index"] = boundary_self_index
    boundary_data["neighbor_index"] = boundary_neighbor_index
    boundary_data["value"] = cp.zeros([boundary_self_index.shape[0]], CUPY_FLOAT)
    phi.add_boundary(boundary_type=boundary_type, boundary_data=boundary_data)
    return phi


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r0, z0 = 30, 5
    grid = Grid(grid_width=0.25, x=[-50, 50], y=[-50, 50], z=[-50, 50])
    solver = PECartesianSolver(grid=grid)
    grid.add_variable("phi", get_phi(grid))
    grid.add_field("rho", get_rho(grid))
    grid.add_field("epsilon", get_epsilon(grid, r0, z0))
    solver.iterate(100)
    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    half_index = grid.coordinate.x.shape[1] // 2
    target_slice = (
        slice(None, None),
        half_index,
        slice(None, None),
    )

    phi = grid.variable.phi.value.get()
    phi = Quantity(phi, default_energy_unit).convert_to(kilocalorie_permol).value
    dist = get_pore_distance(
        grid.coordinate.x,
        grid.coordinate.y,
        grid.coordinate.z,
        r0=r0,
        z0=z0,
        thickness=0,
    )
    phi += (dist == 0).get() * 100
    phi[phi >= 5] = 5
    threshold = 5
    phi[phi <= -threshold] = -threshold
    c = ax.contour(
        grid.coordinate.x[target_slice].get(),
        grid.coordinate.z[target_slice].get(),
        phi[target_slice],
        100,
    )
    fig.colorbar(c)
    plt.show()

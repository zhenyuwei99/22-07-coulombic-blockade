#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pe.py
created time : 2022/11/13
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import cupy as cp
import numba.cuda as cuda
from mdpy.core import Grid
from mdpy.environment import *
from mdpy.unit import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydration import get_pore_distance

THREAD_PER_BLOCK = 128


class PESolver:
    def __init__(self, grid: Grid) -> None:
        """All grid and constant in default unit
        Variable:
        - phi: Electric potential

        Field:
        - epsilon: Relative permittivity
        - rho: charge density

        Constant:
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
        # self._update_boundary_point = cuda.jit(
        #     nb.void(
        #         NUMBA_INT[:, :, ::1],  # boundary_index
        #         NUMBA_INT[::1],  # boundary_type
        #         NUMBA_FLOAT[::1],  # boundary_value
        #         NUMBA_FLOAT[:, :, ::1],  # value
        #     ),
        #     fastmath=True,
        # )(self._update_boundary_point_kernel)

    def _generate_coefficient(self):
        factor = NUMPY_FLOAT(0.5 / self._grid.grid_width**2)
        pre_factor = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        inv_denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        target_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                # 0 for plus and 1 for minus
                target_slice[i] = slice(2, None) if j == 0 else slice(0, -2)
                pre_factor[i, j] = factor * (
                    self._grid.field.epsilon[tuple(target_slice)]
                    + self._grid.field.epsilon[1:-1, 1:-1, 1:-1]
                )
                inv_denominator += pre_factor[i, j]
            target_slice[i] = slice(1, -1)
        inv_denominator = CUPY_FLOAT(1) / inv_denominator
        return pre_factor, inv_denominator

    @staticmethod
    def _update_boundary_point_kernel(
        boundary_index, boundary_type, boundary_value, value
    ):
        point_id = cuda.grid(1)
        num_points = boundary_index.shape[0]
        if point_id >= num_points:
            return None
        local_boundary_index = cuda.local.array((3,), NUMBA_INT)
        local_boundary_type = boundary_type[point_id]
        local_boundary_value = boundary_value[point_id]
        for i in range(3):
            local_boundary_index[i] = boundary_index[point_id, 0, i]
        if local_boundary_type == 0:  # Dirichlet boundary:
            value[
                local_boundary_index[0],
                local_boundary_index[1],
                local_boundary_index[2],
            ] = local_boundary_value
        if local_boundary_type == 1:  # Neumann boundary:
            local_neighbor_index = cuda.local.array((3,), NUMBA_INT)
            dist = NUMBA_INT(0)
            for i in range(3):
                local_neighbor_index[i] = boundary_index[point_id, 1, i]
                dist += local_neighbor_index[i] - local_boundary_index[i]
            neighbor_value = value[
                local_neighbor_index[0],
                local_neighbor_index[1],
                local_neighbor_index[2],
            ]
            value[
                local_boundary_index[0],
                local_boundary_index[1],
                local_boundary_index[2],
            ] = (
                neighbor_value - NUMBA_FLOAT(dist) * local_boundary_value
            )
        local_value = cuda.local.array((2), NUMBA_FLOAT)
        for i in range(2):
            local_value[i] = value[
                local_boundary_index[0],
                local_boundary_index[1],
                local_boundary_index[2],
            ]

    def _update_boundary_point(self, variable):
        boundary_index = variable.boundary_index[:, 0, :]
        neighbor_index = variable.boundary_index[:, 1, :]
        dist = (neighbor_index - boundary_index).sum(1).astype(CUPY_FLOAT)

        variable.value[
            boundary_index[:, 0], boundary_index[:, 1], boundary_index[:, 2]
        ] = (
            variable.value[
                neighbor_index[:, 0], neighbor_index[:, 1], neighbor_index[:, 2]
            ]
            - dist * variable.boundary_value
        )

    def _update_inner_point(self, pre_factor, inv_denominator, scaled_rho, soa_factor):
        factor_a, factor_b = CUPY_FLOAT(soa_factor), CUPY_FLOAT(1 - soa_factor)
        nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        # X Neumann condition
        nominator += pre_factor[0, 0] * self._grid.variable.phi.value[2:, 1:-1, 1:-1]
        nominator += pre_factor[0, 1] * self._grid.variable.phi.value[:-2, 1:-1, 1:-1]
        # Y Neumann
        nominator += pre_factor[1, 0] * self._grid.variable.phi.value[1:-1, 2:, 1:-1]
        nominator += pre_factor[1, 1] * self._grid.variable.phi.value[1:-1, :-2, 1:-1]
        # Z Neumann
        nominator += pre_factor[2, 0] * self._grid.variable.phi.value[1:-1, 1:-1, 2:]
        nominator += pre_factor[2, 1] * self._grid.variable.phi.value[1:-1, 1:-1, :-2]
        # Add charge
        nominator += scaled_rho
        self._grid.variable.phi.value[1:-1, 1:-1, 1:-1] = (
            factor_a * self._grid.variable.phi.value[1:-1, 1:-1, 1:-1]
            + factor_b * nominator * inv_denominator
        )

    def iterate(self, num_iterations, soa_factor=0.01):
        self._grid.check_requirement()
        block_per_grid = int(
            np.ceil(self._grid.variable.phi.boundary_type.shape[0] / THREAD_PER_BLOCK)
        )
        pre_factor, inv_denominator = self._generate_coefficient()
        inv_epsilon0 = NUMPY_FLOAT(1 / self._grid.constant.epsilon0)
        scaled_rho = grid.field.rho[1:-1, 1:-1, 1:-1] * inv_epsilon0
        for iteration in range(num_iterations):
            # self._update_boundary_point[block_per_grid, THREAD_PER_BLOCK](
            #     self._grid.variable.phi.boundary_index,
            #     self._grid.variable.phi.boundary_type,
            #     self._grid.variable.phi.boundary_value,
            #     self._grid.variable.phi.value,
            # )
            # self._update_boundary_point(self._grid.variable.phi)
            self._update_inner_point(
                pre_factor=pre_factor,
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

    # charge = 1 / grid.grid_width**grid.num_dimensions
    # index = [i // 2 for i in grid.coordinate.x.shape]
    # index[2] = 5
    # index[0] += 20
    # num_points = 1 + grid.num_dimensions * 2
    # charge /= num_points
    # for i in range(grid.num_dimensions):
    #     index[i] += 1
    #     rho[tuple(index)] += charge
    #     index[i] -= 2
    #     rho[tuple(index)] += charge
    #     index[i] += 1
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r0, z0 = 30, 5
    grid = Grid(grid_width=0.25, x=[-50, 50], y=[-50, 50], z=[-50, 50])
    solver = PESolver(grid=grid)
    # Variable
    phi = grid.empty_variable()
    boundary_points = (
        grid.inner_shape[0] * grid.inner_shape[1]
        + grid.inner_shape[0] * grid.inner_shape[2]
        + grid.inner_shape[1] * grid.inner_shape[2]
    ) * 2
    phi.boundary_type = cp.ones([boundary_points], CUPY_INT)
    phi.boundary_value = cp.zeros([boundary_points], CUPY_FLOAT)
    boundary_index = cp.zeros([boundary_points, 2, 3], CUPY_INT)
    field = grid.zeros_field().astype(CUPY_INT)
    num_added_points = 0
    for i in range(3):
        target_slice = [slice(1, -1) for i in range(3)]
        target_slice[i] = [0, -1]
        field[tuple(target_slice)] = 1
        index = cp.argwhere(field).astype(CUPY_INT)
        num_points = index.shape[0]
        boundary_index[num_added_points : num_added_points + num_points, 0, :] = index
        field[tuple(target_slice)] = 0
        target_slice[i] = [1, -2]
        field[tuple(target_slice)] = 1
        index = cp.argwhere(field).astype(CUPY_INT)
        boundary_index[num_added_points : num_added_points + num_points, 1, :] = index
        field[tuple(target_slice)] = 0
        num_added_points += num_points
    phi.boundary_index = boundary_index.copy()
    grid.add_variable("phi", phi)
    grid.add_field("rho", get_rho(grid))
    grid.add_field("epsilon", get_epsilon(grid, r0, z0))
    solver.iterate(200)
    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    half_index = grid.coordinate.x.shape[1] // 2
    target_slice = (
        slice(None, None),
        half_index,
        slice(None, None),
    )

    phi = grid.variable.phi.value.get()
    # phi = Quantity(phi, default_energy_unit).convert_to(kilocalorie_permol).value
    # dist = get_pore_distance(
    #     grid.coordinate.x,
    #     grid.coordinate.y,
    #     grid.coordinate.z,
    #     r0=r0,
    #     z0=z0,
    #     thickness=0,
    # )
    # phi += (dist == 0).get() * 100
    # phi[phi >= 5] = 5
    # threshold = 5
    # phi[phi <= -threshold] = -threshold
    # c = ax.contour(
    #     grid.coordinate.x[target_slice].get(),
    #     grid.coordinate.z[target_slice].get(),
    #     phi[target_slice],
    #     200,
    # )
    # fig.colorbar(c)
    # plt.show()

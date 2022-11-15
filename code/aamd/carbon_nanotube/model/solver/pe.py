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
from mdpy.core import Grid
from mdpy.environment import *
from mdpy.unit import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydration import get_pore_distance


class PESolver:
    def __init__(self, grid: Grid) -> None:
        """All grid and constant in default unit

        Field:
        - phi: Electric potential
        - epsilon: Relative permittivity
        - rho: charge density

        Constant:
        - epsilon0 (added): Vacuum permittivity
        """
        self._grid = grid
        field_name_list = [
            "phi",
            "epsilon",
            "rho",
        ]
        constant_name_list = ["epsilon0"]
        self._grid.set_requirement(
            field_name_list=field_name_list,
            constant_name_list=constant_name_list,
        )
        epsilon0 = EPSILON0.convert_to(
            default_charge_unit**2 / (default_energy_unit * default_length_unit)
        ).value
        self._grid.add_constant("epsilon0", epsilon0)

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

    def solve(self, soa_factor=0.01):
        pre_factor, inv_denominator = self._generate_coefficient()
        inv_epsilon0 = NUMPY_FLOAT(1 / self._grid.constant.epsilon0)
        scaled_rho = grid.field.rho[1:-1, 1:-1, 1:-1] * inv_epsilon0
        for iteration in range(500):
            nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            # X Neumann condition
            nominator[:-1, :, :] += (
                pre_factor[0, 0, :-1, :, :] * self._grid.field.phi[2:-1, 1:-1, 1:-1]
            )
            nominator[-1, :, :] += pre_factor[0, 0, -1, :, :] * (
                self._grid.field.phi[-2, 1:-1, 1:-1]
                + self._grid.field.phi[-1, 1:-1, 1:-1] * self._grid.grid_width
            )
            nominator[1:, :, :] += (
                pre_factor[0, 1, 1:, :, :] * self._grid.field.phi[1:-2, 1:-1, 1:-1]
            )
            nominator[0, :, :] += pre_factor[0, 1, 1, :, :] * (
                self._grid.field.phi[1, 1:-1, 1:-1]
                - self._grid.field.phi[0, 1:-1, 1:-1] * self._grid.grid_width
            )
            # Y Neumann
            nominator[:, :-1, :] += (
                pre_factor[1, 0, :, :-1, :] * self._grid.field.phi[1:-1, 2:-1, 1:-1]
            )
            nominator[:, -1, :] += pre_factor[1, 0, :, -1, :] * (
                self._grid.field.phi[1:-1, -2, 1:-1]
                + self._grid.field.phi[1:-1, -1, 1:-1] * self._grid.grid_width
            )
            nominator[:, 1:, :] += (
                pre_factor[1, 1, :, 1:, :] * self._grid.field.phi[1:-1, 1:-2, 1:-1]
            )
            nominator[:, 0, :] += pre_factor[1, 1, :, -1, :] * (
                self._grid.field.phi[1:-1, 1, 1:-1]
                - self._grid.field.phi[1:-1, 0, 1:-1] * self._grid.grid_width
            )
            # Z Neumann
            nominator[:, :, :-1] += (
                pre_factor[2, 0, :, :, :-1] * self._grid.field.phi[1:-1, 1:-1, 2:-1]
            )
            nominator[:, :, -1] += pre_factor[2, 0, :, :, -1] * (
                self._grid.field.phi[1:-1, 1:-1, -2]
                + self._grid.field.phi[1:-1, 1:-1, -1] * self._grid.grid_width
            )
            nominator[:, :, 1:] += (
                pre_factor[2, 1, :, :, 1:] * self._grid.field.phi[1:-1, 1:-1, 1:-2]
            )
            nominator[:, :, 0] += pre_factor[2, 1, :, :, -1] * (
                self._grid.field.phi[1:-1, 1:-1, 1]
                - self._grid.field.phi[1:-1, 1:-1, 0] * self._grid.grid_width
            )
            # Add charge
            nominator += scaled_rho
            self._grid.field.phi[1:-1, 1:-1, 1:-1] = (
                soa_factor * self._grid.field.phi[1:-1, 1:-1, 1:-1]
                + (1 - soa_factor) * nominator * inv_denominator
            )
            print(iteration)


def get_rho(grid: Grid):
    charge = -1 / grid.grid_width**grid.num_dimensions
    rho = grid.zeros_field()
    index = [i // 2 for i in grid.coordinate.x.shape]
    num_points = 1 + grid.num_dimensions * 2
    charge /= num_points
    for i in range(grid.num_dimensions):
        index[i] += 1
        rho[tuple(index)] = charge
        index[i] -= 2
        rho[tuple(index)] = charge
        index[i] += 1
    return rho


def get_epsilon(grid: Grid, r0, z0):
    dist = get_pore_distance(
        grid.coordinate.x,
        grid.coordinate.y,
        grid.coordinate.z,
        r0=r0,
        z0=z0,
        threshold=0,
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
    grid.add_field("phi", grid.zeros_field())
    grid.add_field("rho", get_rho(grid))
    grid.add_field("epsilon", get_epsilon(grid, r0, z0))
    solver.solve()
    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    half_index = grid.coordinate.x.shape[1] // 2
    target_slice = (
        slice(None, None),
        half_index,
        slice(None, None),
    )

    phi = grid.field.phi.get()
    phi = Quantity(phi, default_energy_unit).convert_to(kilocalorie_permol).value
    dist = get_pore_distance(
        grid.coordinate.x,
        grid.coordinate.y,
        grid.coordinate.z,
        r0=r0,
        z0=z0,
        threshold=0,
    )
    phi += (dist == 0).get() * 100
    phi[phi >= 5] = 5
    threshold = 38
    phi[phi <= -threshold] = -threshold
    c = ax.contour(
        grid.coordinate.x[target_slice].get(),
        grid.coordinate.z[target_slice].get(),
        phi[target_slice],
        200,
    )
    fig.colorbar(c)
    plt.show()

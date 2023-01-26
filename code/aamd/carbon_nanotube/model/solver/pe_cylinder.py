#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pe_cylinder.py
created time : 2023/01/13
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


class PECylinderSolver:
    def __init__(self, grid: Grid) -> None:
        """All grid and constant in default unit
        ### Coordinate:
        - r: radius
        - z: z

        ### Variable:
        - phi: Electric potential
            - dirichlet: Dirichlet boundary condition
                - index, value required
            - z-no-gradient: dphi/dz = 0
                - index, direction required. Direction is the index difference between neighbor
            - r-symmetry: dphi/dr = 0
                - index, direction required.
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
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2)
        pre_factor = CUPY_FLOAT(1 / 4 / self._grid.grid_width**2)
        epsilon = self._grid.field.epsilon
        epsilon_h2 = inv_h2 * epsilon[1:-1, 1:-1]
        scaled_hr = (
            CUPY_FLOAT(1 / self._grid.grid_width)
            / self._grid.coordinate.r[1:-1, 1:-1]
            * epsilon[1:-1, 1:-1]
        ).astype(CUPY_FLOAT)
        inv_denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        # 0 for plus and 1 for minus
        shape = [self._grid.num_dimensions, 2] + self._grid.inner_shape
        factor = cp.zeros(shape, CUPY_FLOAT)
        # r
        delta_epsilon = inv_h2 * (epsilon[1:-1, 1:-1] - epsilon[:-2, 1:-1])
        factor[0, 0] = epsilon_h2
        factor[0, 1] = epsilon_h2 - delta_epsilon - scaled_hr
        inv_denominator -= scaled_hr + delta_epsilon
        # z
        delta_epsilon = inv_h2 * (epsilon[1:-1, 1:-1] - epsilon[1:-1, :-2])
        factor[1, 0] = epsilon_h2
        factor[1, 1] = epsilon_h2 - delta_epsilon
        inv_denominator -= delta_epsilon
        inv_denominator += epsilon_h2 * CUPY_FLOAT(4)
        inv_denominator = CUPY_FLOAT(1) / inv_denominator
        return factor.astype(CUPY_FLOAT), inv_denominator.astype(CUPY_FLOAT)

    def _update_boundary_point(self, phi):
        for boundary_type, boundary_data in phi.boundary.items():
            boundary_type = boundary_type.lower()
            if boundary_type == "dirichlet":
                index = boundary_data["index"]
                value = boundary_data["value"]
                phi.value[index[:, 0], index[:, 1]] = value
            elif boundary_type == "z-no-gradient":
                index = boundary_data["index"]
                direction = boundary_data["direction"]
                z_index = (index[:, 0], index[:, 1])
                z_plus1_index = (index[:, 0], index[:, 1] + direction)
                z_plus2_index = (index[:, 0], index[:, 1] + direction + direction)
                phi.value[z_index] = CUPY_FLOAT(1 / 3) * (
                    CUPY_FLOAT(4) * phi.value[z_plus1_index] - phi.value[z_plus2_index]
                )
            elif boundary_type == "r-symmetry":
                index = boundary_data["index"]
                direction = boundary_data["direction"]
                r_index = (index[:, 0], index[:, 1])
                r_plus1_index = (index[:, 0] + direction, index[:, 1])
                r_plus2_index = (index[:, 0] + direction + direction, index[:, 1])
                phi.value[r_index] = CUPY_FLOAT(1 / 3) * (
                    CUPY_FLOAT(4) * phi.value[r_plus1_index] - phi.value[r_plus2_index]
                )
            else:
                raise KeyError(
                    "Only dirichlet and neumann boundary condition supported, while %s provided"
                    % boundary_type
                )

    def _update_inner_point(self, phi, factor, inv_denominator, scaled_rho, soa_factor):
        factor_a, factor_b = CUPY_FLOAT(soa_factor), CUPY_FLOAT(1 - soa_factor)
        nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        # r
        nominator += factor[0, 0] * phi.value[2:, 1:-1]
        nominator += factor[0, 1] * phi.value[:-2, 1:-1]
        # z
        nominator += factor[1, 0] * phi.value[1:-1, 2:]
        nominator += factor[1, 1] * phi.value[1:-1, :-2]
        # Add charge
        nominator += scaled_rho
        # Update
        phi.value[1:-1, 1:-1] = (
            factor_a * phi.value[1:-1, 1:-1] + factor_b * nominator * inv_denominator
        )

    def iterate(self, num_iterations, soa_factor=0.01):
        self._grid.check_requirement()
        factor, inv_denominator = self._get_coefficient()
        inv_epsilon0 = NUMPY_FLOAT(1 / self._grid.constant.epsilon0)
        scaled_rho = self._grid.field.rho[1:-1, 1:-1] * inv_epsilon0
        print(scaled_rho)
        for iteration in range(num_iterations):
            self._update_boundary_point(self._grid.variable.phi)
            self._update_inner_point(
                phi=self._grid.variable.phi,
                factor=factor,
                inv_denominator=inv_denominator,
                scaled_rho=scaled_rho,
                soa_factor=soa_factor,
            )


def get_phi(grid: Grid):
    phi = grid.empty_variable()
    # R symmetry
    field = grid.zeros_field().astype(CUPY_INT)
    field[0, 1:-1] = 1
    boundary_index = cp.argwhere(field).astype(CUPY_INT)
    direction = cp.ones([boundary_index.shape[0]], CUPY_INT)
    phi.add_boundary(
        boundary_type="r-symmetry",
        boundary_data={
            "index": boundary_index,
            "direction": direction.astype(CUPY_INT),
        },
    )
    field = grid.zeros_field().astype(CUPY_INT)
    field[-1, 1:-1] = 1
    direction *= -1
    boundary_index = cp.argwhere(field).astype(CUPY_INT)
    phi.add_boundary(
        boundary_type="r-symmetry",
        boundary_data={
            "index": boundary_index,
            "direction": direction.astype(CUPY_INT),
        },
    )
    # Z boundary
    # field = grid.zeros_field().astype(CUPY_INT)
    # field[:, 0] = 1
    # boundary_index = cp.argwhere(field).astype(CUPY_INT)
    # direction = cp.ones([boundary_index.shape[0]], CUPY_INT)
    # phi.add_boundary(
    #     boundary_type="z-no-gradient",
    #     boundary_data={
    #         "index": boundary_index,
    #         "direction": direction.astype(CUPY_INT),
    #     },
    # )
    # field = grid.zeros_field().astype(CUPY_INT)
    # field[:, -1] = 1
    # direction *= -1
    # boundary_index = cp.argwhere(field).astype(CUPY_INT)
    # phi.add_boundary(
    #     boundary_type="z-no-gradient",
    #     boundary_data={
    #         "index": boundary_index,
    #         "direction": direction.astype(CUPY_INT),
    #     },
    # )

    field = grid.zeros_field().astype(CUPY_INT)
    field[:, 0] = 1
    boundary_index = cp.argwhere(field).astype(CUPY_INT)
    value = (
        cp.zeros([boundary_index.shape[0]])
        + Quantity(1, volt).convert_to(default_energy_unit / default_charge_unit).value
    )
    phi.add_boundary(
        boundary_type="dirichlet",
        boundary_data={
            "index": boundary_index,
            "value": value.astype(CUPY_FLOAT),
        },
    )
    field = grid.zeros_field().astype(CUPY_INT)
    field[:, -1] = 1
    value = cp.zeros([boundary_index.shape[0]])
    boundary_index = cp.argwhere(field).astype(CUPY_INT)
    phi.add_boundary(
        boundary_type="dirichlet",
        boundary_data={
            "index": boundary_index,
            "value": value.astype(CUPY_FLOAT),
        },
    )
    return phi


def get_rho(grid: Grid, r0=5, z0=0):
    rho = grid.zeros_field()
    charge = -1 / grid.grid_width**grid.num_dimensions
    dist = (grid.coordinate.r - r0) ** 2 + (grid.coordinate.z - z0) ** 2
    sigma = grid.grid_width / 3
    rho = cp.exp(-dist / (2 * sigma**2)) * charge
    return rho


def get_epsilon(grid: Grid, r0, z0):
    epsilon = grid.ones_field() * CUPY_FLOAT(78)
    epsilon[(grid.coordinate.r >= r0) & (cp.abs(grid.coordinate.z) <= z0)] = 2
    return epsilon.astype(CUPY_FLOAT)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r0, z0 = 10, 50
    grid = Grid(grid_width=0.25, r=[0, 50], z=[-100, 100])
    solver = PECylinderSolver(grid=grid)
    grid.add_variable("phi", get_phi(grid))
    grid.add_field("rho", get_rho(grid))
    grid.add_field("epsilon", get_epsilon(grid, r0, z0))
    solver.iterate(10000)

    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    phi = grid.variable.phi.value.get()
    phi = Quantity(phi, default_energy_unit).convert_to(kilocalorie_permol).value
    print(phi)
    threshold = 100
    phi[phi >= threshold] = threshold
    phi[phi <= -threshold] = -threshold
    c = ax.contour(
        grid.coordinate.r.get(),
        grid.coordinate.z.get(),
        phi,
        200,
    )
    fig.colorbar(c)
    plt.show()

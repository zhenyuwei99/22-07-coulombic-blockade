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
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as spl
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
            - inner: Inner points
            - dirichlet: Dirichlet point
                - `index`, `value` required
            - no-gradient: dphi/dz = 0
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor

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
        # Mapping of function
        self._func_map = {
            "inner": self._get_inner_points,
            "dirichlet": self._get_dirichlet_points,
            "no-gradient": self._get_no_gradient_points,
        }

    def _get_equation(self, phi):
        data, row, col = [], [], []
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        for key, val in phi.points.items():
            cur_data, cur_row, cur_col, cur_vector = self._func_map[key](**val)
            data.append(cur_data)
            row.append(cur_row)
            col.append(cur_col)
            vector += cur_vector
        # Matrix
        data = cp.hstack(data).astype(CUPY_FLOAT)
        row = cp.hstack(row).astype(CUPY_INT)
        col = cp.hstack(col).astype(CUPY_INT)
        matrix = sp.coo_matrix(
            (data, (row, col)),
            shape=(self._grid.num_points, self._grid.num_points),
            dtype=CUPY_FLOAT,
        )
        # Return
        return matrix.tocsr(), vector.astype(CUPY_FLOAT)

    def _get_inner_points(self, index):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        epsilon = self._grid.field.epsilon
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2)
        index_tuple = (index[:, 0], index[:, 1])
        epsilon_h2 = inv_h2 * epsilon[index_tuple]
        scaled_hr = (
            CUPY_FLOAT(1 / self._grid.grid_width)
            / self._grid.coordinate.r[index_tuple]
            * epsilon[index_tuple]
        ).astype(CUPY_FLOAT)
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        # r+1
        delta_epsilon_r = inv_h2 * (
            epsilon[index[:, 0] + 1, index[:, 1]] - epsilon[index_tuple]
        )
        data.append(epsilon_h2 + delta_epsilon_r + scaled_hr)
        col.append(row_index + z_shape)
        # r-1
        data.append(epsilon_h2)
        col.append(row_index - z_shape)
        # z+1
        delta_epsilon_z = inv_h2 * (
            epsilon[index[:, 0], index[:, 1] + 1] - epsilon[index_tuple]
        )
        data.append(epsilon_h2 + delta_epsilon_z)
        col.append(row_index + 1)
        # z-1
        data.append(epsilon_h2)
        col.append(row_index - 1)
        # Self
        data.append(
            -delta_epsilon_r - scaled_hr - delta_epsilon_z - epsilon_h2 * CUPY_FLOAT(4)
        )
        col.append(row_index)
        # Vector
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        inv_epsilon = NUMPY_FLOAT(1 / self._grid.constant.epsilon0)
        vector[row_index] = -self._grid.field.rho[index_tuple] * inv_epsilon
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_dirichlet_points(self, index, value):
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        size = CUPY_INT(index.shape[0])
        data = cp.ones(size, CUPY_FLOAT)
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        vector[row_index] = value
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row_index).astype(CUPY_INT),
            cp.hstack(row_index).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_no_gradient_points(self, index, dimension, direction):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        size = CUPY_INT(index.shape[0])
        # dimension = 0 radius offset need multiple z_shape
        offset = (direction * (CUPY_INT(1) - dimension) * z_shape).astype(CUPY_INT)
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        # +1
        data.append(cp.zeros(size) + 4)
        col.append(row_index + offset)
        row.append(row_index)
        # +2
        data.append(cp.zeros(size) - 1)
        col.append(row_index + offset + offset)
        row.append(row_index)
        # self
        data.append(cp.zeros(size) - 3)
        col.append(row_index)
        row.append(row_index)
        # Vector
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def iterate(self, num_iterations, is_restart=True):
        self._grid.check_requirement()
        if is_restart:
            self._matrix, self._vector = self._get_equation(self._grid.variable.phi)
        x0 = self._grid.variable.phi.value.reshape(self._grid.num_points)
        res, info = spl.gmres(
            self._matrix,
            self._vector,
            x0=x0,
            maxiter=num_iterations,
            restart=200,
        )
        self._grid.variable.phi.value = res.reshape(self._grid.shape)


def get_phi(grid: Grid):
    phi = grid.empty_variable()
    # Inner
    field = grid.zeros_field().astype(CUPY_INT)
    field[1:-1, 1:-1] = 1
    index = cp.argwhere(field).astype(CUPY_INT)
    phi.register_points(
        type="inner",
        index=index,
    )
    # no-gradient
    field = grid.zeros_field(CUPY_INT)
    dimension = grid.zeros_field(CUPY_INT)
    direction = grid.zeros_field(CUPY_INT)
    # left
    field[0, 1:-1] = 1
    dimension[0, 1:-1] = 0
    direction[0, 1:-1] = 1
    # right
    field[-1, 1:-1] = 2
    dimension[-1, 1:-1] = 0
    direction[-1, 1:-1] = -1
    # down
    field[:, 0] = 3  # down
    dimension[:, 0] = 1
    direction[:, 0] = 1
    # up
    field[:, -1] = 4  # down
    dimension[:, -1] = 1
    direction[:, -1] = -1
    for i in [1, 2, 3, 4]:
        index = cp.argwhere(field == i).astype(CUPY_INT)
        index_tuple = (index[:, 0], index[:, 1])
        phi.register_points(
            type="no-gradient",
            index=index,
            dimension=dimension[index_tuple].astype(CUPY_INT),
            direction=direction[index_tuple].astype(CUPY_INT),
        )

    # field = grid.zeros_field().astype(CUPY_INT)
    # field[:, 0] = 1
    # index = cp.argwhere(field).astype(CUPY_INT)
    # value = (
    #     cp.zeros([index.shape[0]])
    #     + Quantity(1, volt).convert_to(default_energy_unit / default_charge_unit).value
    # )
    # phi.register_points(type="dirichlet", index=index, value=value.astype(CUPY_FLOAT))
    # field = grid.zeros_field().astype(CUPY_INT)
    # field[:, -1] = 1
    # index = cp.argwhere(field).astype(CUPY_INT)
    # value = cp.zeros([index.shape[0]])
    # phi.register_points(type="dirichlet", index=index, value=value.astype(CUPY_FLOAT))
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
    import time
    import matplotlib.pyplot as plt

    r0, z0 = 10, 25
    grid = Grid(grid_width=0.25, r=[0, 50], z=[-100, 100])
    solver = PECylinderSolver(grid=grid)
    grid.add_variable("phi", get_phi(grid))
    grid.add_field("rho", get_rho(grid))
    grid.add_field("epsilon", get_epsilon(grid, r0, z0))

    s = time.time()
    solver.iterate(6000)
    phi = grid.variable.phi.value.get()
    phi = Quantity(phi, default_energy_unit).convert_to(kilocalorie_permol).value
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    s = time.time()
    solver.iterate(6000, False)
    phi = grid.variable.phi.value.get()
    phi = Quantity(phi, default_energy_unit).convert_to(kilocalorie_permol).value
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    threshold = 200
    phi[phi >= threshold] = threshold
    phi[phi <= -threshold] = -threshold
    c = ax.contour(
        grid.coordinate.r.get(),
        grid.coordinate.z.get(),
        phi,
        500,
    )
    fig.colorbar(c)
    plt.show()

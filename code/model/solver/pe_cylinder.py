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

import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as spl
from mdpy.unit import *

from model import *
from model.core import Grid
from model.utils import *


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
            - axial-symmetry: boundary point for r=0
                - `index` required

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
            "axial-symmetry": self._get_axial_symmetry_points,
        }

    def _get_equation(self, phi):
        data, row, col = [], [], []
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        epsilon_h2 = CUPY_FLOAT(1 / self._grid.grid_width) * self._grid.field.epsilon
        for key, val in phi.points.items():
            cur_data, cur_row, cur_col, cur_vector = self._func_map[key](
                epsilon_h2=epsilon_h2, **val
            )
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

    def _get_inner_points(self, epsilon_h2, index):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        epsilon = self._grid.field.epsilon
        self_index = (index[:, 0], index[:, 1])
        # scaled_hr = (
        #     CUPY_FLOAT(1 / self._grid.grid_width)
        #     / self._grid.coordinate.r[self_index]
        #     * epsilon[self_index]
        # ).astype(CUPY_FLOAT)
        scaled_hr = (epsilon[self_index] / self._grid.coordinate.r[self_index]).astype(
            CUPY_FLOAT
        )
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)

        # direction = (index[:, 1] >= self._grid.inner_shape[1] // 2).astype(CUPY_INT)
        # plus_direction = direction.copy()
        # minus_direction = (direction - 1).astype(CUPY_INT)
        # direction[direction == 0] = -1

        # # r+1
        # data.append(epsilon_h2[index[:, 0] + 1, index[:, 1]] + scaled_hr)
        # col.append(row_index + z_shape)
        # # r-1
        # data.append(epsilon_h2[self_index])
        # col.append(row_index - z_shape)
        # # z+1
        # data.append(epsilon_h2[index[:, 0], index[:, 1] + plus_direction])
        # col.append(row_index + 1)
        # # z-1
        # data.append(epsilon_h2[index[:, 0], index[:, 1] + minus_direction])
        # col.append(row_index - 1)
        # # Self
        # data.append(
        #     -epsilon_h2[index[:, 0] + 1, index[:, 1]]
        #     - epsilon_h2[index[:, 0], index[:, 1] + direction]
        #     - scaled_hr
        #     - epsilon_h2[self_index] * CUPY_FLOAT(2)
        # )
        # col.append(row_index)
        # # Vector
        # vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        # inv_epsilon0 = NUMPY_FLOAT(1 / self._grid.constant.epsilon0 / (-4 * np.pi))
        # vector[row_index] = self._grid.field.rho[self_index] * inv_epsilon0
        # # Return
        # return (
        #     cp.hstack(data).astype(CUPY_FLOAT),
        #     cp.hstack(row).astype(CUPY_INT),
        #     cp.hstack(col).astype(CUPY_INT),
        #     vector.astype(CUPY_FLOAT),
        # )

        # r+1
        data.append(epsilon_h2[index[:, 0] + 1, index[:, 1]] + scaled_hr)
        col.append(row_index + z_shape)
        # r-1
        data.append(epsilon_h2[self_index])
        col.append(row_index - z_shape)
        # z+1
        data.append(epsilon_h2[index[:, 0], index[:, 1] + 1])
        col.append(row_index + 1)
        # z-1
        data.append(epsilon_h2[self_index])
        col.append(row_index - 1)
        # Self
        data.append(
            -epsilon_h2[index[:, 0] + 1, index[:, 1]]
            - epsilon_h2[index[:, 0], index[:, 1] + 1]
            - scaled_hr
            - epsilon_h2[self_index] * CUPY_FLOAT(2)
        )
        col.append(row_index)
        # Vector
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        inv_epsilon0 = NUMPY_FLOAT(1 / self._grid.constant.epsilon0 / (-4 * np.pi))
        vector[row_index] = self._grid.field.rho[self_index] * inv_epsilon0
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_dirichlet_points(self, epsilon_h2, index, value):
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

    def _get_no_gradient_points(self, epsilon_h2, index, dimension, direction):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        # +1 offset in the same direction. 0 offset z_shape, 1 offset 1
        offset = (CUPY_INT(1) - dimension) * z_shape + dimension
        offset = (offset * direction).astype(CUPY_INT)
        data.append(epsilon_h2[self_index] * CUPY_FLOAT(4))
        col.append(row_index + offset)
        # +2
        data.append(epsilon_h2[self_index] * CUPY_FLOAT(-0.5))
        col.append(row_index + offset + offset)
        # +1 offset in the different direction. 0 offset 1, 1 offset z_shape
        offset = (CUPY_INT(1) - dimension + dimension * z_shape).astype(CUPY_INT)
        # if dimension = 0 index[:, 0] unchanged but index[:, 1] change
        neighbor_index = (index[:, 0] + dimension, index[:, 1] + (1 - dimension))
        data.append(epsilon_h2[neighbor_index])
        col.append(row_index + offset)
        # -1
        data.append(epsilon_h2[self_index])
        col.append(row_index - offset)
        # self
        data.append(
            epsilon_h2[self_index] * CUPY_FLOAT(-4.5) - epsilon_h2[neighbor_index]
        )
        col.append(row_index)
        # Vector
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        inv_epsilon0 = NUMPY_FLOAT(1 / self._grid.constant.epsilon0 / (-4 * np.pi))
        vector[row_index] = self._grid.field.rho[self_index] * inv_epsilon0
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_axial_symmetry_points(self, epsilon_h2, index):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        # r+1
        epsilon_h2x8 = epsilon_h2[self_index] * CUPY_FLOAT(8)
        data.append(epsilon_h2x8)
        col.append(row_index + z_shape)
        # r+2
        data.append(-epsilon_h2[self_index])
        col.append(row_index + CUPY_INT(2 * z_shape))
        # z+1
        data.append(epsilon_h2[index[:, 0], index[:, 1] + CUPY_INT(1)])
        col.append(row_index + 1)
        # z-1
        data.append(epsilon_h2[self_index])
        col.append(row_index - 1)
        # Self
        data.append(-epsilon_h2[index[:, 0], index[:, 1] + CUPY_INT(1)] - epsilon_h2x8)
        col.append(row_index)
        # Vector
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        inv_epsilon0 = NUMPY_FLOAT(1 / self._grid.constant.epsilon0 / (-4 * np.pi))
        vector[row_index] = self._grid.field.rho[self_index] * inv_epsilon0
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def iterate(self, num_iterations, is_restart=False):
        self._grid.check_requirement()
        self._matrix, self._vector = self._get_equation(self._grid.variable.phi)
        # if is_restart:
        #     x0 = self._grid.variable.phi.value.reshape(self._grid.num_points)
        #     res, info = spl.gmres(
        #         self._matrix,
        #         self._vector,
        #         x0=x0,
        #         maxiter=num_iterations,
        #         restart=100,
        #     )
        # else:
        #     res, info = spl.gmres(
        #         self._matrix,
        #         self._vector,
        #         maxiter=num_iterations,
        #         restart=100,
        #     )
        res = spl.spsolve(self._matrix, self._vector)
        self._grid.variable.phi.value = res.reshape(self._grid.shape)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def residual(self):
        res = self._grid.variable.phi.value.reshape(self._grid.num_points)
        residual = (cp.abs(self._matrix.dot(res) - self._vector)).mean()
        return residual


def get_phi(grid: Grid):
    phi = grid.empty_variable()
    voltage = (
        Quantity(0.5, volt * elementary_charge).convert_to(default_energy_unit).value
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


def get_rho(grid: Grid, r0=5, z0=0):
    rho = grid.zeros_field()
    charge = -1 / grid.grid_width**grid.num_dimensions
    dist = (grid.coordinate.r - r0) ** 2 + (grid.coordinate.z - z0) ** 2
    sigma = grid.grid_width / 3
    rho = cp.exp(-dist / (2 * sigma**2)) * charge
    rho = grid.zeros_field()
    return rho


def get_distance(grid: Grid, r0, z0, rs):
    r0s = r0 + rs
    z0s = z0 - rs
    r = grid.coordinate.r
    z = grid.coordinate.z
    dist = grid.zeros_field(CUPY_FLOAT) - 1
    # In pore
    index = (cp.abs(z) <= z0s) & (r <= r0)
    dist[index] = r0 - r[index]
    # Out pore
    index = (z >= z0) & (r >= r0s)
    dist[index] = z[index] - z0
    index = (z <= -z0) & (r >= r0s)
    dist[index] = -(z[index] + z0)
    # Sphere part
    index = (z > z0s) & (r < r0s)
    temp = cp.sqrt((z[index] - z0s) ** 2 + (r[index] - r0s) ** 2) - rs
    temp[temp < 0] = -1
    dist[index] = temp
    index = (z < -z0s) & (r < r0s)
    temp = cp.sqrt((z[index] + z0s) ** 2 + (r[index] - r0s) ** 2) - rs
    temp[temp < 0] = -1
    dist[index] = temp
    return dist.astype(CUPY_FLOAT)


def get_epsilon(grid: Grid, dist, rs=4):
    # epsilon = grid.ones_field() * CUPY_FLOAT(78)
    # epsilon[dist == -1] = 2
    alpha = reasoning_alpha(rs)
    epsilon = CUPY_FLOAT(1) / (1 + cp.exp(-alpha * dist))
    epsilon *= CUPY_FLOAT(78)
    epsilon += CUPY_FLOAT(2)
    epsilon[dist == -1] = 2
    return epsilon.astype(CUPY_FLOAT)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    r0, z0, rs = 5, 25, 5
    grid = Grid(grid_width=0.5, r=[0, 50], z=[-100, 100])
    dist = get_distance(grid, r0, z0, rs)
    solver = PECylinderSolver(grid=grid)
    grid.add_variable("phi", get_phi(grid))
    grid.add_field("rho", get_rho(grid))
    grid.add_field("epsilon", get_epsilon(grid, dist))

    import matplotlib.pyplot as plt

    epsilon = get_epsilon(grid, dist)
    c = plt.contourf(
        grid.coordinate.r.get(), grid.coordinate.z.get(), epsilon.get(), 200
    )
    plt.colorbar()
    plt.show()

    s = time.time()
    solver.iterate(5000)
    phi = grid.variable.phi.value.get()
    phi = Quantity(phi, default_energy_unit).convert_to(kilocalorie_permol).value
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    s = time.time()
    solver.iterate(5000, False)
    print(solver.residual)
    phi = grid.variable.phi.value.get()
    phi = Quantity(phi, default_energy_unit).convert_to(kilocalorie_permol).value
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    threshold = 200
    phi[phi >= threshold] = threshold
    phi[phi <= -threshold] = -threshold
    c = ax.contour(
        grid.coordinate.r.get()[1:-1, 1:-1],
        grid.coordinate.z.get()[1:-1, 1:-1],
        phi[1:-1, 1:-1],
        200,
        cmap="RdBu",
    )
    fig.colorbar(c)
    plt.show()

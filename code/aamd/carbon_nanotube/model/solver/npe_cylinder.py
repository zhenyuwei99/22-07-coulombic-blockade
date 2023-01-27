#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : npe_cylinder.py
created time : 2023/01/27
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
from solver import *


class NPECylinderSolver:
    def __init__(self, grid: Grid, ion_type: str) -> None:
        """All grid and constant in default unit
        ### Coordinate:
        - r: radius
        - z: z

        ### Variable:
        - rho_[ion]: Number density of [ion]
            - inner: Inner points
            - dirichlet: Dirichlet point, constant density
                - `index`, `value` required
            - no-flux: no flux boundary
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor

        ### Field:
        - u_[ion]: External potential of [ion]

        ### Constant:
        - beta: 1/kBT
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
        self._grid.add_requirement("constant", "d_%s" % self._ion_type)
        # Diffusion coefficient
        self._grid.add_constant(
            "d_%s" % ion_type,
            check_quantity_value(ION_DICT[ion_type]["d"], DIFFUSION_UNIT),
        )
        # Mapping of function
        self._func_map = {
            "inner": self._get_inner_points,
            "dirichlet": self._get_dirichlet_points,
            "no-flux": self._get_no_flux_points,
        }

    def _get_equation(self, rho):
        data, row, col = [], [], []
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        for key, val in rho.points.items():
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
        index_tuple = (index[:, 0], index[:, 1])
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2)
        inv_rh = (
            CUPY_FLOAT(1 / self._grid.grid_width) / self._grid.coordinate.r[index_tuple]
        ).astype(CUPY_FLOAT)
        z_shape = CUPY_INT(self._grid.shape[1])
        size = CUPY_INT(index.shape[0])
        u = (
            getattr(self._grid.field, "u_%s" % self._ion_type)
            * self._grid.constant.beta
        )
        delta_u_r = u[index[:, 0] + 1, index[:, 1]] - u[index_tuple]
        delta_u_z = u[index[:, 0], index[:, 1] + 1] - u[index_tuple]
        delta_u_r_rh = delta_u_r * inv_rh
        delta_u_r_h2 = delta_u_r * inv_h2
        delta_u_z_h2 = delta_u_z * inv_h2
        curv_u_r = inv_h2 * (
            u[index[:, 0] + 1, index[:, 1]]
            - CUPY_FLOAT(2) * u[index_tuple]
            + u[index[:, 0] - 1, index[:, 1]]
        )
        curv_u_z = inv_h2 * (
            u[index[:, 0], index[:, 1] + 1]
            - CUPY_FLOAT(2) * u[index_tuple]
            + u[index[:, 0], index[:, 1] - 1]
        )
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        # r+1
        inv_h2 += cp.zeros(size, CUPY_FLOAT)
        data.append(inv_h2 + inv_rh + delta_u_r_h2)
        col.append(row_index + z_shape)
        # r-1
        data.append(inv_h2)
        col.append(row_index - z_shape)
        # z+1
        data.append(inv_h2 + delta_u_z_h2)
        col.append(row_index + 1)
        # z-1
        data.append(inv_h2)
        col.append(row_index - 1)
        # Self
        data.append(
            curv_u_r
            + delta_u_r_rh
            + curv_u_z
            - inv_h2 * CUPY_FLOAT(4)
            - inv_rh
            - delta_u_r_h2
            - delta_u_z_h2
        )
        col.append(row_index)
        # Vector
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
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

    def _get_no_flux_points(self, index, dimension, direction):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        size = CUPY_INT(index.shape[0])
        inv_h = CUPY_FLOAT(1 / self._grid.grid_width)
        # dimension = 0 radius offset need multiple z_shape
        offset = (direction * (CUPY_INT(1) - dimension) * z_shape).astype(CUPY_INT)
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(3):
            row.append(row_index)
        # +1
        temp = cp.zeros(size, CUPY_FLOAT)
        data.append(temp + CUPY_FLOAT(2 * inv_h))
        col.append(row_index + offset)
        # +2
        data.append(temp + CUPY_FLOAT(-0.5 * inv_h))
        col.append(row_index + offset + offset)
        # self
        u = (
            getattr(self._grid.field, "u_%s" % self._ion_type)
            * self._grid.constant.beta
        )
        offset = cp.zeros_like(index, CUPY_INT)
        offset[:, 0] += direction * (CUPY_INT(1) - dimension)
        offset[:, 1] += direction * dimension
        index_plus_1 = index + offset
        index_plus_2 = index_plus_1 + offset
        data.append(
            CUPY_FLOAT(0.5 * inv_h)
            * (
                CUPY_FLOAT(4) * u[index_plus_1[:, 0], index_plus_1[:, 1]]
                - CUPY_FLOAT(3) * u[index[:, 0], index[:, 1]]
                - u[index_plus_2[:, 0], index_plus_2[:, 1]]
                - CUPY_FLOAT(3)
            )
        )
        col.append(row_index)
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
        rho = getattr(self._grid.variable, "rho_%s" % self._ion_type)
        if is_restart:
            self._matrix, self._vector = self._get_equation(rho)
        x0 = rho.value.reshape(self._grid.num_points)
        res, info = spl.gmres(
            self._matrix,
            self._vector,
            x0=x0,
            maxiter=num_iterations,
            restart=500,
        )
        rho.value = res.reshape(self._grid.shape)

    def get_flux(self, dimension: int):
        rho = getattr(self._grid.variable, "rho_%s" % self._ion_type).value
        d = getattr(self._grid.constant, "d_%s" % self._ion_type)
        u = (
            getattr(self._grid.field, "u_%s" % self._ion_type)
            * self._grid.constant.beta
        )
        inv_h = CUPY_FLOAT(1 / self._grid.grid_width)
        slice_plus = [slice(1, -1) for i in range(2)]
        slice_plus[dimension] = slice(2, None)
        slice_plus = tuple(slice_plus)
        flux = rho[slice_plus] - rho[1:-1, 1:-1]
        flux += rho[1:-1, 1:-1] * (u[slice_plus] - u[1:-1, 1:-1])
        flux *= inv_h * d
        return flux


def get_rho(grid: Grid):
    rho = grid.empty_variable()
    # Inner
    field = grid.zeros_field().astype(CUPY_INT)
    field[1:-1, 1:-1] = 1
    index = cp.argwhere(field).astype(CUPY_INT)
    rho.register_points(
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
        if i == 3 or i == 4:
            density = (
                (Quantity(0.15, mol / decimeter**3) * NA)
                .convert_to(1 / default_length_unit**3)
                .value
            )
            rho.register_points(
                type="dirichlet",
                index=index,
                value=cp.zeros(index.shape[0], CUPY_FLOAT) + density,
            )
            continue
        index_tuple = (index[:, 0], index[:, 1])
        rho.register_points(
            type="no-flux",
            index=index,
            dimension=dimension[index_tuple].astype(CUPY_INT),
            direction=direction[index_tuple].astype(CUPY_INT),
        )
    return rho


def get_u(grid: Grid, r0=10, z0=5):
    sigma = grid.grid_width * 10
    height = Quantity(0.5, kilocalorie_permol).convert_to(default_energy_unit).value
    dist = (grid.coordinate.r - r0) ** 2 + (grid.coordinate.z - z0) ** 2
    u = cp.exp(-dist / (2 * sigma**2)) * height

    # dist = (grid.coordinate.r - r0 - 10) ** 2 + (grid.coordinate.z - z0) ** 2
    # u += cp.exp(-dist / (2 * sigma**2)) * -height
    return u.astype(CUPY_FLOAT)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    beta = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    beta = 1 / beta

    grid = Grid(grid_width=0.25, r=[0, 50], z=[-100, 100])
    solver = NPECylinderSolver(grid=grid, ion_type="k")
    grid.add_variable("rho_k", get_rho(grid))
    grid.add_field("u_k", get_u(grid))
    grid.add_constant("beta", beta)

    s = time.time()
    solver.iterate(6000)
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    s = time.time()
    solver.iterate(6000, False)
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    rho = grid.variable.rho_k.value[1:-1, 1:-1].get()
    rho = (
        (Quantity(rho, 1 / default_length_unit**3) / NA)
        .convert_to(mol / decimeter**3)
        .value
    )

    flux = solver.get_flux(0).get()
    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    threshold = 200
    c = ax.contour(
        grid.coordinate.r[1:-1, 1:-1].get(),
        grid.coordinate.z[1:-1, 1:-1].get(),
        rho,
        200,
    )
    fig.colorbar(c)
    plt.show()

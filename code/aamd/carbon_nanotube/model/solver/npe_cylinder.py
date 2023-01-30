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
            - no-gradient: gradient of c equals to 0
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor
            - no-flux: no flux boundary
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor
            - axial-symmetry: boundary point for r=0
                - `index` required

        ### Field:
        - u_[ion]: External potential of [ion]

        ### Constant:
        - beta: 1/kBT
        - rho_bulk_[ion]: Bulk density of [ion]
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
            "no-gradient": self._get_no_gradient_points,
            "z-no-flux": self._get_z_no_flux_points,
            "r-no-flux": self._get_r_no_flux_points,
            "no-flux": self._get_no_flux_points,
            "axial-symmetry": self._get_axial_symmetry_points,
        }

    def _get_equation(self, rho):
        data, row, col = [], [], []
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        factor = CUPY_FLOAT(self._grid.constant.beta / self._grid.grid_width**2)
        scaled_u = getattr(self._grid.field, "u_%s" % self._ion_type) * factor
        for key, val in rho.points.items():
            cur_data, cur_row, cur_col, cur_vector = self._func_map[key](
                scaled_u=scaled_u, **val
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

    def _get_inner_points(self, scaled_u, index):
        data, row, col = [], [], []
        self_index = (index[:, 0], index[:, 1])
        inv_rh = (
            CUPY_FLOAT(1 / self._grid.grid_width) / self._grid.coordinate.r[self_index]
        ).astype(CUPY_FLOAT)
        z_shape = CUPY_INT(self._grid.shape[1])
        size = CUPY_INT(index.shape[0])
        delta_u_r_rh = (
            (scaled_u[index[:, 0] + 1, index[:, 1]] - scaled_u[self_index])
            * inv_rh
            * CUPY_FLOAT(self._grid.grid_width**2)
        )
        delta_u_r_h2 = scaled_u[index[:, 0] + 1, index[:, 1]] - scaled_u[self_index]
        delta_u_z_h2 = scaled_u[index[:, 0], index[:, 1] + 1] - scaled_u[self_index]
        curv_u_r = (
            scaled_u[index[:, 0] + 1, index[:, 1]]
            - CUPY_FLOAT(2) * scaled_u[self_index]
            + scaled_u[index[:, 0] - 1, index[:, 1]]
        )
        curv_u_z = (
            scaled_u[index[:, 0], index[:, 1] + 1]
            - CUPY_FLOAT(2) * scaled_u[self_index]
            + scaled_u[index[:, 0], index[:, 1] - 1]
        )
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        # r+1
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2) + cp.zeros(size, CUPY_FLOAT)
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

    def _get_dirichlet_points(self, scaled_u, index, value):
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

    def _get_no_gradient_points(self, scaled_u, index, dimension, direction):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2)
        inv_h2 += cp.zeros(index.shape[0], CUPY_FLOAT)
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        self_index = (index[:, 0], index[:, 1])
        # if dimension = 0 index[:, 0] change but index[:, 1] unchanged
        r_plus_1 = (index[:, 0] + (1 - dimension), index[:, 1] + dimension)
        r_plus_2 = (
            index[:, 0] + (2 - dimension * CUPY_INT(2)),
            index[:, 1] + dimension,
        )
        # if dimension = 0 index[:, 0] unchanged but index[:, 1] change
        z_plus_1 = (index[:, 0] + dimension, index[:, 1] + (1 - dimension))
        z_minus_1 = (index[:, 0] - dimension, index[:, 1] - (1 - dimension))
        for i in range(5):
            row.append(row_index)
        # +1 offset in the same direction. 0 offset z_shape, 1 offset 1
        offset = (CUPY_INT(1) - dimension) * z_shape + dimension
        offset = (offset * direction).astype(CUPY_INT)
        data.append(inv_h2 * CUPY_FLOAT(4))
        col.append(row_index + offset)
        # +2
        data.append(inv_h2 * CUPY_FLOAT(-0.5))
        col.append(row_index + offset + offset)
        # +1 offset in the different direction. 0 offset 1, 1 offset z_shape
        offset = (CUPY_INT(1) - dimension + dimension * z_shape).astype(CUPY_INT)
        data.append(inv_h2 + scaled_u[z_plus_1] - scaled_u[self_index])
        col.append(row_index + offset)
        # -1
        data.append(inv_h2)
        col.append(row_index - offset)
        # self
        data.append(
            CUPY_FLOAT(4) * scaled_u[r_plus_1]
            + CUPY_FLOAT(-0.5) * scaled_u[r_plus_2]
            + CUPY_FLOAT(-4.5) * scaled_u[self_index]
            + scaled_u[z_minus_1]
            + CUPY_FLOAT(-5.5) * inv_h2
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

    def _get_r_no_flux_points(self, scaled_u, index, direction):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        self_index = (index[:, 0], index[:, 1])
        r_plus_1 = (index[:, 0] + direction, index[:, 1])
        r_plus_2 = (index[:, 0] + direction + direction, index[:, 1])
        z_plus_1 = (index[:, 0], index[:, 1] + 1)
        z_minus_1 = (index[:, 0], index[:, 1] - 1)
        for i in range(5):
            row.append(row_index)
        # r+1
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2)
        inv_h2 += cp.zeros(index.shape[0], CUPY_FLOAT)
        offset = z_shape * direction
        data.append(inv_h2 * CUPY_FLOAT(4))
        col.append(row_index + offset)
        # r+2
        data.append(inv_h2 * CUPY_FLOAT(-0.5))
        col.append(row_index + offset + offset)
        # z+1
        data.append(inv_h2 + scaled_u[z_plus_1] - scaled_u[self_index])
        col.append(row_index + 1)
        # z-1
        data.append(inv_h2)
        col.append(row_index - 1)
        # self
        factor = CUPY_FLOAT(0.5 * self._grid.grid_width) * (
            CUPY_FLOAT(4) * scaled_u[r_plus_1]
            - scaled_u[r_plus_2]
            - CUPY_FLOAT(3) * scaled_u[self_index]
        )
        factor = factor**2
        data.append(
            CUPY_FLOAT(4) * scaled_u[r_plus_1]
            + CUPY_FLOAT(-0.5) * scaled_u[r_plus_2]
            + CUPY_FLOAT(-4.5) * scaled_u[self_index]
            + scaled_u[z_minus_1]
            + CUPY_FLOAT(-5.5) * inv_h2
            + factor
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

    def _get_z_no_flux_points(self, scaled_u, index, direction):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        self_index = (index[:, 0], index[:, 1])
        z_plus_1 = (index[:, 0], index[:, 1] + direction)
        z_plus_2 = (index[:, 0], index[:, 1] + direction + direction)
        r_plus_1 = (index[:, 0] + 1, index[:, 1])
        r_minus_1 = (index[:, 0] - 1, index[:, 1])
        inv_rh = (
            CUPY_FLOAT(1 / self._grid.grid_width) / self._grid.coordinate.r[self_index]
        ).astype(CUPY_FLOAT)
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2) + cp.zeros(
            index.shape[0], CUPY_FLOAT
        )
        z_shape = CUPY_INT(self._grid.shape[1])
        for i in range(5):
            row.append(row_index)
        # z+1
        data.append(inv_h2 * CUPY_FLOAT(4))
        col.append(row_index + direction)
        # z+2
        data.append(inv_h2 * CUPY_FLOAT(-0.5))
        col.append(row_index + direction + direction)
        # r+1
        data.append(inv_h2 + inv_rh + scaled_u[r_plus_1] - scaled_u[self_index])
        col.append(row_index + z_shape)
        # r-1
        data.append(inv_h2)
        col.append(row_index - z_shape)
        # self
        delta_u_r_rh = (
            (scaled_u[r_plus_1] - scaled_u[self_index])
            * inv_rh
            * CUPY_FLOAT(self._grid.grid_width**2)
        )
        factor = CUPY_FLOAT(0.5 * self._grid.grid_width) * (
            CUPY_FLOAT(4) * scaled_u[z_plus_1]
            - scaled_u[z_plus_2]
            - CUPY_FLOAT(3) * scaled_u[self_index]
        )
        factor = factor**2
        data.append(
            CUPY_FLOAT(4) * scaled_u[z_plus_1]
            + CUPY_FLOAT(-0.5) * scaled_u[z_plus_2]
            + CUPY_FLOAT(-4.5) * scaled_u[self_index]
            + scaled_u[r_minus_1]
            + factor
            + delta_u_r_rh
            + CUPY_FLOAT(-5.5) * inv_h2
            - inv_rh
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

    def _get_no_flux_points(self, index, r_direction, z_direction):
        pass

    def _get_axial_symmetry_points(self, scaled_u, index):
        data, row, col = [], [], []
        self_index = (index[:, 0], index[:, 1])
        z_shape = CUPY_INT(self._grid.shape[1])
        size = CUPY_INT(index.shape[0])
        delta_u_z_h2 = scaled_u[index[:, 0], index[:, 1] + 1] - scaled_u[self_index]
        curv_u_r = (
            CUPY_FLOAT(8) * scaled_u[index[:, 0] + 1, index[:, 1]]
            - scaled_u[index[:, 0] + 2, index[:, 1]]
            - CUPY_FLOAT(7) * scaled_u[self_index]
        )
        curv_u_z = (
            scaled_u[index[:, 0], index[:, 1] + 1]
            - CUPY_FLOAT(2) * scaled_u[self_index]
            + scaled_u[index[:, 0], index[:, 1] - 1]
        )
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        # r+1
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2) + cp.zeros(size, CUPY_FLOAT)
        data.append(inv_h2 * CUPY_FLOAT(8))
        col.append(row_index + z_shape)
        # r+2
        data.append(-inv_h2)
        col.append(row_index + z_shape + z_shape)
        # z+1
        data.append(inv_h2 + delta_u_z_h2)
        col.append(row_index + 1)
        # z-1
        data.append(inv_h2)
        col.append(row_index - 1)
        # Self
        data.append(curv_u_r + curv_u_z - delta_u_z_h2 - inv_h2 * CUPY_FLOAT(9))
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

    def _guess_solution(self, rho, bulk_density):
        dirichlet = rho.points["dirichlet"]
        inner = rho.points["inner"]
        x0 = self._grid.zeros_field(CUPY_FLOAT)
        # dirichlet
        index = (dirichlet["index"][:, 0], dirichlet["index"][:, 1])
        value = dirichlet["value"]
        x0[index] = value
        # inner
        index = (inner["index"][:, 0], inner["index"][:, 1])
        factor = CUPY_FLOAT(-self._grid.constant.beta)
        factor = getattr(self._grid.field, "u_%s" % self._ion_type)[index] * factor
        x0[index] = bulk_density * cp.exp(factor) / 10
        return x0.reshape(self._grid.num_points)

    def iterate(self, num_iterations, is_restart=False):
        self._grid.check_requirement()
        rho = getattr(self._grid.variable, "rho_%s" % self._ion_type)
        rho_bulk = getattr(self._grid.constant, "rho_bulk_%s" % self._ion_type)
        self._matrix, self._vector = self._get_equation(rho)
        if is_restart:
            x0 = rho.value.reshape(self._grid.num_points)
            res, info = spl.gmres(
                self._matrix,
                self._vector,
                x0=self._guess_solution(rho=rho, bulk_density=rho_bulk),
                maxiter=num_iterations,
                restart=100,
            )
        else:
            res, info = spl.gmres(
                self._matrix,
                self._vector,
                maxiter=num_iterations,
                restart=100,
            )
        # res = spl.lsqr(self._matrix, self._vector)[0]
        # res[res < 0] = 0
        rho.value = res.reshape(self._grid.shape)

    def get_flux(self, dimension: int):
        rho = getattr(self._grid.variable, "rho_%s" % self._ion_type).value
        d = getattr(self._grid.constant, "d_%s" % self._ion_type)
        z = getattr(self._grid.constant, "z_%s" % self._ion_type)
        scaled_u = (
            getattr(self._grid.field, "u_%s" % self._ion_type)
            * self._grid.constant.beta
        )
        inv_h = CUPY_FLOAT(1 / self._grid.grid_width)
        slice_plus = [slice(1, -1) for i in range(2)]
        slice_plus[dimension] = slice(2, None)
        slice_plus = tuple(slice_plus)
        flux = rho[slice_plus] - rho[1:-1, 1:-1]
        flux += rho[1:-1, 1:-1] * (scaled_u[slice_plus] - scaled_u[1:-1, 1:-1])
        flux *= inv_h * d * z
        return flux

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def residual(self):
        rho = getattr(self._grid.variable, "rho_%s" % self._ion_type).value
        res = rho.reshape(self._grid.num_points)
        residual = (cp.abs(self._matrix.dot(res) - self._vector)).mean()
        return residual


def get_rho(grid: Grid):
    density = Quantity(0.15, mol / decimeter**3) * NA
    density = check_quantity_value(density, 1 / default_length_unit**3)
    rho = grid.empty_variable()
    field = grid.zeros_field().astype(CUPY_INT) - 1
    value = grid.zeros_field().astype(CUPY_FLOAT)
    dimension = grid.zeros_field().astype(CUPY_INT)
    direction = grid.zeros_field().astype(CUPY_INT)
    # 0: inner; 1: dirichlet; 2: no-gradient; 3: axial-symmetry
    # Inner
    field[1:-1, 1:-1] = 0
    # dirichlet
    field[:, [0, -1]] = 1
    value[:, 0] = density
    value[:, -1] = 0
    # z-no-flux
    r_min_index = 100
    min_index, max_index = 100, 400
    field[r_min_index:-1, min_index] = 2
    direction[r_min_index:-1, min_index] = 1
    field[r_min_index:-1, max_index] = 2
    direction[r_min_index:-1, max_index] = -1
    # r-no-flux
    field[r_min_index, min_index:max_index] = 3
    direction[r_min_index, min_index:max_index] = 1
    field[-1, 1:-1] = 3
    direction[-1, 1:-1] = -1
    # axial-symmetry
    field[0, 1:-1] = 4

    field[r_min_index + 1 :, min_index + 1 : max_index] = 1
    value[r_min_index + 1 :, min_index + 1 : max_index] = 0
    # Register
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
    rho.register_points(
        type="z-no-flux",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    rho.register_points(
        type="r-no-flux",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 4).astype(CUPY_INT)
    rho.register_points(type="axial-symmetry", index=index)
    return rho


def get_u(grid: Grid, r0=10, z0=5):
    sigma = grid.grid_width * 10
    height = Quantity(0.5, kilocalorie_permol).convert_to(default_energy_unit).value
    dist = (grid.coordinate.r - r0) ** 2 + (grid.coordinate.z - z0) ** 2
    u = cp.exp(-dist / (2 * sigma**2)) * height

    # dist = (grid.coordinate.r - r0 - 10) ** 2 + (grid.coordinate.z - z0) ** 2
    # u += cp.exp(-dist / (2 * sigma**2)) * -height
    u = grid.zeros_field(CUPY_FLOAT)
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
    solver.iterate(1000)
    print(solver.residual)
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    s = time.time()
    solver.iterate(1000, False)
    e = time.time()
    print("Run xxx for %s s" % (e - s))

    rho = grid.variable.rho_k.value.get()
    rho = (
        (Quantity(rho, 1 / default_length_unit**3) / NA)
        .convert_to(mol / decimeter**3)
        .value
    )

    # flux = solver.get_flux(0).get()
    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    threshold = 200
    c = ax.contour(
        grid.coordinate.r.get(),
        grid.coordinate.z.get(),
        rho,
        50,
    )
    fig.colorbar(c)
    plt.show()

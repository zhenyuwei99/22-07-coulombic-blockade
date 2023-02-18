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

import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as spl
from mdpy.unit import *

from model import *
from model.core import Grid


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
            "axial-symmetry": self._get_axial_symmetry_points,
            "r-no-flux": self._get_r_no_flux_points,
            "z-no-flux": self._get_z_no_flux_points,
            "no-flux-inner": self._get_no_flux_inner_points,
        }
        # Attribute
        # self._u_h2
        # self._delta_u
        # self._upwind

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

    def _update_factor(self):
        pass

    def _get_inner_points(self, scaled_u, index):
        data, row, col = [], [], []
        self_index = (index[:, 0], index[:, 1])
        z_shape = CUPY_INT(self._grid.shape[1])
        size = CUPY_INT(index.shape[0])
        self_index = (index[:, 0], index[:, 1])
        # Upwind direction
        r_direction = (
            scaled_u[index[:, 0] + 1, index[:, 1]] <= scaled_u[self_index]
        ).astype(CUPY_INT)
        r_direction[r_direction == 0] = CUPY_INT(-1)
        z_direction = (
            scaled_u[index[:, 0], index[:, 1] + 1] <= scaled_u[self_index]
        ).astype(CUPY_INT)
        z_direction[z_direction == 0] = CUPY_INT(-1)
        # Index
        r_plus = (index[:, 0] + r_direction, index[:, 1])
        r_minus = (index[:, 0] - r_direction, index[:, 1])
        z_plus = (index[:, 0], index[:, 1] + z_direction)
        z_minus = (index[:, 0], index[:, 1] - z_direction)
        # Factor
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2) + cp.zeros(size, CUPY_FLOAT)
        inv_rh = (
            CUPY_FLOAT(1 / self._grid.grid_width)
            * r_direction
            / (self._grid.coordinate.r[self_index])
        ).astype(CUPY_FLOAT)
        delta_u_r_h2 = scaled_u[r_plus] - scaled_u[self_index]
        delta_u_r_rh = (
            r_direction * delta_u_r_h2 * inv_rh * CUPY_FLOAT(self._grid.grid_width**2)
        )
        curv_u_r = (
            scaled_u[r_plus] - CUPY_FLOAT(2) * scaled_u[self_index] + scaled_u[r_minus]
        )
        delta_u_z_h2 = scaled_u[z_plus] - scaled_u[self_index]
        curv_u_z = (
            scaled_u[z_plus] - CUPY_FLOAT(2) * scaled_u[self_index] + scaled_u[z_minus]
        )
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        # r plus
        offset = r_direction * z_shape
        data.append(inv_h2 + inv_rh + delta_u_r_h2)
        col.append(row_index + offset)
        # r minus
        data.append(inv_h2)
        col.append(row_index - offset)
        # z plus
        offset = z_direction
        data.append(inv_h2 + delta_u_z_h2)
        col.append(row_index + offset)
        # z minus
        data.append(inv_h2)
        col.append(row_index - offset)
        # Self
        data.append(
            delta_u_r_rh
            + curv_u_r
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

    def _get_axial_symmetry_points(self, scaled_u, index):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        size = CUPY_INT(index.shape[0])
        self_index = (index[:, 0], index[:, 1])
        z_direction = (
            scaled_u[index[:, 0], index[:, 1] + 1] <= scaled_u[self_index]
        ).astype(CUPY_INT)
        z_direction[z_direction == 0] = CUPY_INT(-1)
        r_plus_1 = (index[:, 0] + 1, index[:, 1])
        r_plus_2 = (index[:, 0] + 2, index[:, 1])
        z_plus = (index[:, 0], index[:, 1] + z_direction)
        z_minus = (index[:, 0], index[:, 1] - z_direction)
        curv_u_r = (
            CUPY_FLOAT(8) * scaled_u[r_plus_1]
            - scaled_u[r_plus_2]
            - CUPY_FLOAT(7) * scaled_u[self_index]
        )
        curv_u_z = (
            scaled_u[z_plus] - CUPY_FLOAT(2) * scaled_u[self_index] + scaled_u[z_minus]
        )
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2) + cp.zeros(size, CUPY_FLOAT)
        delta_u_z_h2 = scaled_u[z_plus] - scaled_u[self_index]
        for i in range(5):
            row.append(row_index)
        # r+1
        data.append(inv_h2 * CUPY_FLOAT(8))
        col.append(row_index + z_shape)
        # r+2
        data.append(-inv_h2)
        col.append(row_index + z_shape + z_shape)
        # z+1
        data.append(inv_h2 + delta_u_z_h2)
        col.append(row_index + z_direction)
        # z-1
        data.append(inv_h2)
        col.append(row_index - z_direction)
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

    def _get_no_flux_inner_points(self, scaled_u, index, unit_vec):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        direction = unit_vec.copy()
        direction[direction >= 0] = 1
        direction[direction < 0] = -1
        direction = direction.astype(CUPY_INT)
        self_index = (index[:, 0], index[:, 1])
        factor = cp.zeros_like(direction, CUPY_INT)
        factor[:, 0] = (
            scaled_u[index[:, 0] + direction[:, 0], index[:, 1]] <= scaled_u[self_index]
        ).astype(CUPY_INT)
        factor[:, 1] = (
            scaled_u[index[:, 0], index[:, 1] + direction[:, 1]] <= scaled_u[self_index]
        ).astype(CUPY_INT)
        factor[direction == 0] = CUPY_INT(-1)
        direction *= factor
        factor = direction.astype(CUPY_FLOAT)
        factor *= unit_vec
        r_plus = (index[:, 0] + direction[:, 0], index[:, 1])
        z_plus = (index[:, 0], index[:, 1] + direction[:, 1])
        self_index = (index[:, 0], index[:, 1])
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2)
        inv_h2 += cp.zeros(index.shape[0], CUPY_FLOAT)
        for i in range(3):
            row.append(row_index)
        # r plus
        offset = z_shape * direction[:, 0]
        data.append(inv_h2 * factor[:, 0])
        col.append(row_index + offset)
        # z plus
        data.append(inv_h2 * factor[:, 1])
        col.append(row_index + direction[:, 1])
        # self
        self_factor = factor[:, 0] * (scaled_u[r_plus] - scaled_u[self_index] - inv_h2)
        self_factor += factor[:, 1] * (scaled_u[z_plus] - scaled_u[self_index] - inv_h2)
        data.append(self_factor)
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
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2)
        inv_h2 += cp.zeros(index.shape[0], CUPY_FLOAT)
        for i in range(3):
            row.append(row_index)
        # r+1
        offset = z_shape * direction
        data.append(inv_h2 * CUPY_FLOAT(4))
        col.append(row_index + offset)
        # r+1
        offset = z_shape * direction
        data.append(inv_h2 * CUPY_FLOAT(-1))
        col.append(row_index + offset)
        # self
        data.append(
            scaled_u[r_plus_1] * CUPY_FLOAT(4)
            - scaled_u[r_plus_2]
            - scaled_u[self_index] * CUPY_FLOAT(3)
            - inv_h2 * CUPY_FLOAT(3)
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
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2)
        inv_h2 += cp.zeros(index.shape[0], CUPY_FLOAT)
        for i in range(2):
            row.append(row_index)
        # z+1
        offset = direction
        data.append(inv_h2)
        col.append(row_index + offset)
        # self
        data.append(scaled_u[z_plus_1] - scaled_u[self_index] - inv_h2)
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

    def iterate(self, num_iterations, is_restart=False, solver_freq=100):
        self._grid.check_requirement()
        rho = getattr(self._grid.variable, "rho_%s" % self._ion_type)
        self._matrix, self._vector = self._get_equation(rho)
        if is_restart:
            x0 = rho.value.reshape(self._grid.num_points)
            res = spl.gmres(
                self._matrix,
                self._vector,
                x0=x0,
                maxiter=num_iterations,
                restart=solver_freq,
            )[0]
        else:
            res = spl.gmres(
                self._matrix, self._vector, maxiter=num_iterations, restart=solver_freq
            )[0]
        # res = spl.spsolve(self._matrix, self._vector)
        # res = spl.lsqr(self._matrix, self._vector)[0]
        # res[res < 0] = 0
        rho.value = res.reshape(self._grid.shape)

    def get_flux(self, dimension: int, direction: int):
        rho = getattr(self._grid.variable, "rho_%s" % self._ion_type).value
        d = getattr(self._grid.constant, "d_%s" % self._ion_type)
        scaled_u = (
            getattr(self._grid.field, "u_%s" % self._ion_type)
            * self._grid.constant.beta
        )
        inv_h = CUPY_FLOAT(1 / self._grid.grid_width)
        slice_plus = [slice(1, -1) for i in range(2)]
        slice_plus[dimension] = slice(2, None) if direction == 1 else slice(None, -2)
        slice_plus = tuple(slice_plus)
        flux = rho[slice_plus] - rho[1:-1, 1:-1]
        flux += rho[1:-1, 1:-1] * (scaled_u[slice_plus] - scaled_u[1:-1, 1:-1])
        flux *= inv_h * d * CUPY_FLOAT(direction)
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

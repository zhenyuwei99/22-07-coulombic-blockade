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
            - axial-symmetry: boundary point for r=0
                - `index` required
            - r-no-gradient: ∂phi/∂r = 0
                - `index`, `dimension`, `direction` required.
                - `direction`: the index difference between neighbor
            - z-no-gradient: ∂phi/∂z = 0
                - `index`, `dimension`, `direction` required.
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
            "axial-symmetry": self._get_axial_symmetry_points,
            "no-gradient-inner": self._get_no_gradient_inner_points,
            "r-no-gradient": self._get_r_no_gradient_points,
            "z-no-gradient": self._get_z_no_gradient_points,
        }
        # Attribute
        self._inv_epsilon0 = CUPY_FLOAT(0)
        self._epsilon_h2 = self._grid.zeros_field(CUPY_FLOAT)
        self._epsilon_2hr = self._grid.zeros_field(CUPY_FLOAT)
        self._delta_epsilon_h2_r = self._grid.zeros_field(CUPY_FLOAT)
        self._delta_epsilon_h2_z = self._grid.zeros_field(CUPY_FLOAT)
        self._upwind_direction_r = self._grid.zeros_field(CUPY_INT)
        self._upwind_direction_z = self._grid.zeros_field(CUPY_INT)

    def _update_factor(self):
        epsilon = self._grid.field.epsilon
        inv_h = CUPY_FLOAT(1 / self._grid.grid_width)
        inv_h2 = CUPY_FLOAT(inv_h**2)
        inv_2h2 = CUPY_FLOAT(inv_h2 / 2)
        # Epsilon factor
        self._inv_epsilon0 = NUMPY_FLOAT(-1 / self._grid.constant.epsilon0)

        self._epsilon_h2 = inv_h2 * self._grid.field.epsilon
        self._epsilon_2hr[1:, :] = (
            CUPY_FLOAT(0.5 * inv_h)
            * self._grid.field.epsilon[1:, :]
            / self._grid.coordinate.r[1:, :]
        )
        self._delta_epsilon_h2_r[1:-1, :] = inv_2h2 * (epsilon[2:, :] - epsilon[:-2, :])
        self._delta_epsilon_h2_z[:, 1:-1] = inv_2h2 * (epsilon[:, 2:] - epsilon[:, :-2])
        # Upwind factor
        self._upwind_direction_r[self._delta_epsilon_h2_r > 0] = CUPY_INT(1)
        self._upwind_direction_r[self._delta_epsilon_h2_r < 0] = CUPY_INT(-1)
        self._upwind_direction_z[self._delta_epsilon_h2_z > 0] = CUPY_INT(1)
        self._upwind_direction_z[self._delta_epsilon_h2_z < 0] = CUPY_INT(-1)

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
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        conv_term_r = (
            self._upwind_direction_r[self_index].astype(CUPY_FLOAT)
            * self._delta_epsilon_h2_r[self_index]
        )
        conv_term_z = (
            self._upwind_direction_z[self_index].astype(CUPY_FLOAT)
            * self._delta_epsilon_h2_z[self_index]
        )
        # r+1
        plus_index = self._upwind_direction_r[self_index] == 1
        factor = self._epsilon_h2[self_index] + self._epsilon_2hr[self_index]
        factor[plus_index] += conv_term_r[plus_index]
        data.append(factor)
        col.append(row_index + z_shape)
        # r-1
        factor = self._epsilon_h2[self_index] - self._epsilon_2hr[self_index]
        factor[~plus_index] += conv_term_r[~plus_index]
        data.append(factor)
        col.append(row_index - z_shape)
        # z+1
        plus_index = self._upwind_direction_z[self_index] == 1
        factor = self._epsilon_h2[self_index]
        factor[plus_index] += conv_term_z[plus_index]
        data.append(factor)
        col.append(row_index + 1)
        # z-1
        factor = self._epsilon_h2[self_index]
        factor[~plus_index] += conv_term_z[~plus_index]
        data.append(factor)
        col.append(row_index - 1)
        # Self
        data.append(
            -conv_term_r - conv_term_z - self._epsilon_h2[self_index] * CUPY_FLOAT(4)
        )
        col.append(row_index)
        # Vector
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        vector[row_index] = self._grid.field.rho[self_index] * self._inv_epsilon0
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

    def _get_axial_symmetry_points(self, index):
        # do not consider the ∇epsilon∇phi term
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        # r+1
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(8))
        col.append(row_index + z_shape)
        # r+2
        data.append(-self._epsilon_h2[self_index])
        col.append(row_index + CUPY_INT(2 * z_shape))
        # z+1
        data.append(self._epsilon_h2[self_index])
        col.append(row_index + 1)
        # z-1
        data.append(self._epsilon_h2[self_index])
        col.append(row_index - 1)
        # Self
        data.append(-self._epsilon_h2[self_index] * CUPY_FLOAT(9))
        col.append(row_index)
        # Vector
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        vector[row_index] = self._grid.field.rho[self_index] * self._inv_epsilon0
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_no_gradient_inner_points(self, index, unit_vector):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(3):
            row.append(row_index)
        factor_r = self._epsilon_h2[self_index] * self._upwind_direction_r[
            self_index
        ].astype(CUPY_FLOAT)
        factor_z = self._epsilon_h2[self_index] * self._upwind_direction_z[
            self_index
        ].astype(CUPY_FLOAT)
        # vector = cp.abs(unit_vector)
        # r plus
        offset = (self._upwind_direction_r[self_index] * z_shape).astype(CUPY_INT)
        data.append(factor_r * unit_vector[:, 0])
        col.append(row_index + offset)
        # z plus
        offset = self._upwind_direction_z[self_index].astype(CUPY_INT)
        data.append(factor_z * unit_vector[:, 1])
        col.append(row_index + offset)
        # Self
        data.append(-(factor_r * unit_vector[:, 0] + factor_z * unit_vector[:, 1]))
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
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(4):
            row.append(row_index)
        epsilon_h2 = self._epsilon_h2[self_index]
        # r+1
        data.append(epsilon_h2 * unit_vector[:, 0])
        col.append(row_index + z_shape)
        # r-1
        data.append(epsilon_h2 * -unit_vector[:, 0])
        col.append(row_index - z_shape)
        # z+1
        data.append(epsilon_h2 * unit_vector[:, 1])
        col.append(row_index + 1)
        # z-1
        data.append(epsilon_h2 * -unit_vector[:, 1])
        col.append(row_index - 1)
        # Vector
        vector = cp.zeros(self._grid.num_points, CUPY_FLOAT)
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_r_no_gradient_points(self, index, direction):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(3):
            row.append(row_index)
        # r+1
        offset = (z_shape * direction).astype(CUPY_INT)
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(4))
        col.append(row_index + offset)
        # r+2
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(-1))
        col.append(row_index + offset + offset)
        # self
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(-3))
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

    def _get_z_no_gradient_points(self, index, direction):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(3):
            row.append(row_index)
        # z+1
        offset = direction.astype(CUPY_INT)
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(4))
        col.append(row_index + offset)
        # z+2
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(-1))
        col.append(row_index + offset + offset)
        # self
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(-3))
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

    def iterate(self, num_iterations, is_restart=False, solver_freq=300):
        self._grid.check_requirement()
        self._update_factor()
        self._matrix, self._vector = self._get_equation(self._grid.variable.phi)
        if is_restart:
            x0 = self._grid.variable.phi.value.reshape(self._grid.num_points)
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
        # res = spl.lsqr(self._matrix, self._vector)[0]
        # res = spl.spsolve(self._matrix, self._vector)
        self._grid.variable.phi.value = res.reshape(self._grid.shape)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def residual(self):
        res = self._grid.variable.phi.value.reshape(self._grid.num_points)
        residual = (cp.abs(self._matrix.dot(res) - self._vector)).mean()
        return residual

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pe_center_cylinder.py
created time : 2023/02/23
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
from model.core import Grid, Variable
from model.utils import *


class PECenterCylinderSolver:
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
             - axial-symmetry-boundary: boundary point for r=0
                 - `index` required
             - r-no-gradient-boundary: ∂phi/∂r = 0
                 - `index`

        ### Field:
         - epsilon: Relative permittivity
         - depsilon_dr: ∂epsilon/∂r
         - depsilon_dz: ∂epsilon/∂z
         - rho_fixed: Fixed charge density

         ### Constant:
         - epsilon0 (added): Vacuum permittivity
        """
        # Read input
        self._grid = grid
        # Add requirement
        self._grid.add_requirement("variable", "phi")
        self._grid.add_requirement("field", "epsilon")
        self._grid.add_requirement("field", "depsilon_dr")
        self._grid.add_requirement("field", "depsilon_dz")
        # Add constant
        epsilon0 = check_quantity_value(
            EPSILON0,
            default_charge_unit**2 / (default_energy_unit * default_length_unit),
        )
        self._grid.add_constant("epsilon0", epsilon0)
        # Function map
        self._func_map = {
            "inner": self._get_inner,
            "dirichlet": self._get_dirichlet,
            "axial-symmetry-boundary": self._get_axial_symmetry_boundary,
            "r-no-gradient-boundary": self._get_r_no_gradient_boundary,
        }

    def _get_equation(self, phi: Variable):
        # Attribute
        inv_h2 = CUPY_FLOAT(1 / self._grid.grid_width**2)
        self._inv_epsilon0 = NUMPY_FLOAT(-1 / self._grid.constant.epsilon0)
        self._epsilon_h2 = inv_h2 * self._grid.field.epsilon
        # Add data
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

    def _get_inner(self, index):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        inv_2h = CUPY_FLOAT(0.5 / self._grid.grid_width)
        epsilon_h2 = self._epsilon_h2[self_index]
        depsilon_dr = self._grid.field.depsilon_dr[self_index] * inv_2h
        depsilon_dz = self._grid.field.depsilon_dz[self_index] * inv_2h
        epsilon_hr = inv_2h * (
            self._grid.field.epsilon[self_index] / self._grid.coordinate.r[self_index]
        )
        # r+1
        data.append(depsilon_dr + epsilon_h2 + epsilon_hr)
        col.append(row_index + z_shape)
        # r-1
        data.append(-depsilon_dr + epsilon_h2 - epsilon_hr)
        col.append(row_index - z_shape)
        # z+1
        data.append(depsilon_dz + epsilon_h2)
        col.append(row_index + 1)
        # z-1
        data.append(-depsilon_dr + epsilon_h2)
        col.append(row_index - 1)
        # Self
        data.append(epsilon_h2 * CUPY_FLOAT(-4))
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

    def _get_dirichlet(self, index, value):
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

    def _get_axial_symmetry_boundary(self, index):
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

    def _get_r_no_gradient_boundary(self, index):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(3):
            row.append(row_index)
        # r+1
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(4))
        col.append(row_index - z_shape)
        # r+2
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(-1))
        col.append(row_index - z_shape - z_shape)
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

    def iterate(self, num_iterations, is_restart=False, solver_freq=50):
        self._grid.check_requirement()
        self._matrix, self._vector = self._get_equation(self._grid.variable.phi)
        # if is_restart:
        #     x0 = self._grid.variable.phi.value.reshape(self._grid.num_points)
        #     res = spl.gmres(
        #         self._matrix,
        #         self._vector,
        #         x0=x0,
        #         maxiter=num_iterations,
        #         restart=solver_freq,
        #     )[0]
        # else:
        #     res = spl.gmres(
        #         self._matrix, self._vector, maxiter=num_iterations, restart=solver_freq
        #     )[0]
        # res = spl.lsqr(self._matrix, self._vector)[0]
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

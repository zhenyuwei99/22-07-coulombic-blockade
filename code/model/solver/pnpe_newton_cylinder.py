#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pnpe_newton_cylinder.py
created time : 2023/02/23
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as spl
from mdpy.utils import check_quantity, check_quantity_value
from mdpy.unit import *
from model import *
from model.core import Grid


class PNPENewtonCylinderSolver:
    def __init__(
        self,
        grid: Grid,
        ion_types: list[str],
        temperature: Quantity = Quantity(300, kelvin),
    ) -> None:
        """All grid and constant in default unit
        ### Variable:
        - phi: Electric potential
            - inner: Inner points
            - dirichlet: Dirichlet point
                - `index`, `value` required
            - axial-symmetry-boundary: ∂phi/∂r|r=0 = 0
                - `index` required.
            - r-no-gradient-boundary: ∂phi/∂r|r=rb = 0
                - `index` required.
        - rho_[ion]: Number density of [ion]
            - inner: Inner points
            - dirichlet: Dirichlet point
                - `index`, `value` required
            - axial-symmetry-boundary: ∂rho/∂r|r=0 = 0
                - `index` required.
            - r-no-flux-boundary: (∂rho/∂r + rho∂u/∂r)|r=rb = 0
                - `index` required.

        ### Field:
        - epsilon: Relative permittivity
        - depsilon_dr: ∂epsilon/∂r
        - depsilon_dz: ∂epsilon/∂z
        - rho_fixed: Fixed charge density

        ### Constant:
        - phi_direction: -1/1; 1: when phi(0) > phi(-1)
        - beta: (added) 1/kBT
        - epsilon0 (added): Vacuum permittivity
        - r_[ion] (added): Radius of [ion]
        - z_[ion] (added): Valence of [ion]
        - d_[ion] (added): Diffusion coefficient of [ion]
        """
        # Read input
        self._grid = grid
        self._ion_types = ion_types
        # Add requirement
        self._grid.add_requirement("variable", "phi")
        self._grid.add_requirement("field", "epsilon")
        self._grid.add_requirement("field", "depsilon_dr")
        self._grid.add_requirement("field", "depsilon_dz")
        self._grid.add_requirement("field", "rho_fixed")
        self._grid.add_requirement("constant", "phi_direction")
        self._grid.add_requirement("constant", "beta")
        for ion_type in self._ion_types:
            self._grid.add_requirement("variable", self._getattr_name("rho", ion_type))
            self._grid.add_requirement("constant", self._getattr_name("r", ion_type))
            self._grid.add_requirement("constant", self._getattr_name("z", ion_type))
            self._grid.add_requirement("constant", self._getattr_name("d", ion_type))
        # Add constant
        beta = check_quantity_value(
            check_quantity(temperature, kelvin) * KB, default_energy_unit
        )
        self._grid.add_constant("beta", CUPY_FLOAT(1 / beta))
        for ion_type in self._ion_types:
            self._grid.add_constant(
                self._getattr_name("r", ion_type),
                check_quantity_value(VDW_DICT[ion_type]["sigma"], default_length_unit),
            )
            self._grid.add_constant(
                self._getattr_name("z", ion_type),
                check_quantity_value(ION_DICT[ion_type]["val"], VAL_UNIT),
            )
            self._grid.add_constant(
                self._getattr_name("d", ion_type),
                check_quantity_value(ION_DICT[ion_type]["d"], DIFFUSION_UNIT),
            )
        # function map
        self._func_map = {
            "pe-inner": self._get_pe_inner,
            "pe-dirichlet": self._get_pe_dirichlet,
            "pe-axial-symmetry-boundary": self._get_pe_axial_symmetry_boundary,
            "pe-r-no-gradient-boundary": self._get_pe_r_no_gradient_boundary,
            "npe-inner": self._get_npe_inner,
            "npe-dirichlet": self._get_npe_dirichlet,
            "npe-axial-symmetry-boundary": self._get_npe_axial_symmetry_boundary,
            "npe-r-no-flux-boundary": self._get_npe_r_no_flux_boundary,
            "npe-z-no-gradient-boundary": self._get_npe_z_no_gradient_boundary,
        }
        # Attributes
        epsilon0 = check_quantity_value(
            EPSILON0,
            default_charge_unit**2 / (default_energy_unit * default_length_unit),
        )
        self._inv_epsilon0 = CUPY_FLOAT(1 / epsilon0)
        self._num_ions = len(self._ion_types)
        self._num_variables = (self._num_ions + 1) * self._grid.num_points
        self._h = CUPY_FLOAT(self._grid.grid_width)
        self._inv_h = CUPY_FLOAT(1 / self._h)
        self._inv_2h = CUPY_FLOAT(0.5 / self._h)
        self._inv_h2 = CUPY_FLOAT(1 / self._h**2)
        ## PE attribute
        self._pe_upwind_r = grid.zeros_field(CUPY_INT)
        self._pe_upwind_z = grid.zeros_field(CUPY_INT)
        self._inv_2hr = grid.zeros_field(CUPY_FLOAT)
        self._epsilon_h2 = grid.zeros_field(CUPY_FLOAT)
        self._epsilon_2hr = grid.zeros_field(CUPY_FLOAT)
        ## NPE attribute
        self._dphi_dr = grid.zeros_field(CUPY_FLOAT)
        self._dphi_dz = grid.zeros_field(CUPY_FLOAT)
        self._curv_phi = grid.zeros_field(CUPY_FLOAT)

    def _getattr_name(self, name: str, ion_type: str):
        return "%s_%s" % (name, ion_type)

    def _get_offset(self, ion_type: str):
        index = self._ion_types.index(ion_type)
        return CUPY_INT((index + 1) * self._grid.num_points)

    def _get_equation(self):
        self._update_variable_factor()
        data, row, col = [], [], []
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        for key, val in self._grid.variable.phi.points.items():
            res = self._func_map[key](**val)
            data.append(res[0])
            row.append(res[1])
            col.append(res[2])
            vector += res[3]
        for ion_type in self._ion_types:
            rho = getattr(self._grid.variable, "rho_" + ion_type)
            for key, val in rho.points.items():
                res = self._func_map[key](ion_type, **val)
                data.append(res[0])
                row.append(res[1])
                col.append(res[2])
                vector += res[3]
        # Matrix
        data = cp.hstack(data).astype(CUPY_FLOAT)
        row = cp.hstack(row).astype(CUPY_INT)
        col = cp.hstack(col).astype(CUPY_INT)
        matrix = sp.coo_matrix(
            (data, (row, col)),
            shape=(self._num_variables, self._num_variables),
            dtype=CUPY_FLOAT,
        )
        # Return
        return matrix.tocsr(), vector.astype(CUPY_FLOAT)

    def _update_constant_factor(self):
        self._inv_2hr[1:, :] = self._inv_2h / self._grid.coordinate.r[1:, :]
        self._epsilon_h2 = self._inv_h2 * self._grid.field.epsilon
        self._epsilon_2hr = self._inv_2hr * self._grid.field.epsilon
        self._pe_upwind_r[self._grid.field.depsilon_dr > 0] = CUPY_INT(1)
        self._pe_upwind_r[self._grid.field.depsilon_dr < 0] = CUPY_INT(-1)
        self._pe_upwind_z[self._grid.field.depsilon_dz > 0] = CUPY_INT(1)
        self._pe_upwind_z[self._grid.field.depsilon_dz < 0] = CUPY_INT(-1)

    def _get_pe_inner(self, index):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(5):
            row.append(row_index)
        conv_term_r = self._inv_h * (
            self._pe_upwind_r[self_index].astype(CUPY_FLOAT)
            * self._grid.field.depsilon_dr[self_index]
        )
        conv_term_z = self._inv_h * (
            self._pe_upwind_z[self_index].astype(CUPY_FLOAT)
            * self._grid.field.depsilon_dz[self_index]
        )
        # phi r+1
        plus_index_r = self._pe_upwind_r[self_index] == 1
        factor = self._epsilon_h2[self_index] + self._epsilon_2hr[self_index]
        factor[plus_index_r] += conv_term_r[plus_index_r]
        data.append(factor)
        col.append(row_index + z_shape)
        # phi r-1
        factor = self._epsilon_h2[self_index] - self._epsilon_2hr[self_index]
        factor[~plus_index_r] += conv_term_r[~plus_index_r]
        data.append(factor)
        col.append(row_index - z_shape)
        # phi z+1
        plus_index_z = self._pe_upwind_z[self_index] == 1
        factor = self._epsilon_h2[self_index]
        factor[plus_index_z] += conv_term_z[plus_index_z]
        data.append(factor)
        col.append(row_index + 1)
        # phi z-1
        factor = self._epsilon_h2[self_index]
        factor[~plus_index_z] += conv_term_z[~plus_index_z]
        data.append(factor)
        col.append(row_index - 1)
        # Self
        data.append(
            -conv_term_r - conv_term_z - self._epsilon_h2[self_index] * CUPY_FLOAT(4)
        )
        col.append(row_index)
        # c
        for ion_type in self._ion_types:
            row_offset = self._get_offset(ion_type)
            factor = getattr(self._grid.constant, "z_" + ion_type) * self._inv_epsilon0
            factor += cp.zeros(index.shape[0], CUPY_FLOAT)
            data.append(factor)
            col.append(row_index + row_offset)
            row.append(row_index)
        # Prediction
        phi = self._grid.variable.phi.value
        r_plus, r_minus = (index[:, 0] + 1, index[:, 1]), (index[:, 0] - 1, index[:, 1])
        z_plus, z_minus = (index[:, 0], index[:, 1] + 1), (index[:, 0], index[:, 1] - 1)
        pred = self._epsilon_h2[self_index] * (
            phi[r_plus]
            + phi[r_minus]
            + phi[z_plus]
            + phi[z_minus]
            - CUPY_FLOAT(4) * phi[self_index]
        )
        pred += self._epsilon_2hr[self_index] * (phi[r_plus] - phi[r_minus])
        pred += conv_term_r * (
            phi[index[:, 0] + self._pe_upwind_r[self_index], index[:, 1]]
            - phi[self_index]
        )
        pred += conv_term_z * (
            phi[index[:, 0], index[:, 1] + self._pe_upwind_z[self_index]]
            - phi[self_index]
        )
        for ion_type in self._ion_types:
            factor = getattr(self._grid.constant, "z_" + ion_type) * self._inv_epsilon0
            rho = getattr(self._grid.variable, "rho_" + ion_type).value[self_index]
            pred += factor * rho
        # Vector
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        vector[row_index] = -self._grid.field.rho_fixed[self_index] * self._inv_epsilon0
        vector[row_index] -= pred
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_pe_dirichlet(self, index, value):
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        self_index = (index[:, 0], index[:, 1])
        data = cp.ones(index.shape[0], CUPY_FLOAT)
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        vector[row_index] = value - self._grid.variable.phi.value[self_index]
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row_index).astype(CUPY_INT),
            cp.hstack(row_index).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_pe_axial_symmetry_boundary(self, index):
        # do not consider the ∇epsilon∇phi term
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(6):
            row.append(row_index)
        # r+1
        factor = self._epsilon_h2[self_index] * CUPY_FLOAT(-108 / 9)
        data.append(factor)
        col.append(row_index + z_shape)
        # r+2
        factor = self._epsilon_h2[self_index] * CUPY_FLOAT(27 / 9)
        data.append(factor)
        col.append(row_index + CUPY_INT(2 * z_shape))
        # r+3
        factor = self._epsilon_h2[self_index] * CUPY_FLOAT(4 / 9)
        data.append(factor)
        col.append(row_index + CUPY_INT(3 * z_shape))
        # z+1
        factor = self._epsilon_h2[self_index]
        data.append(factor)
        col.append(row_index + 1)
        # z-1
        factor = self._epsilon_h2[self_index]
        data.append(factor)
        col.append(row_index - 1)
        # Self
        factor = self._epsilon_h2[self_index] * CUPY_FLOAT(-2 + 77 / 9)
        data.append(factor)
        col.append(row_index)
        # c
        for ion_type in self._ion_types:
            row_offset = self._get_offset(ion_type)
            factor = getattr(self._grid.constant, "z_" + ion_type) * self._inv_epsilon0
            factor += cp.zeros(index.shape[0], CUPY_FLOAT)
            data.append(factor)
            col.append(row_index + row_offset)
            row.append(row_index)
        # Prediction
        phi = self._grid.variable.phi.value
        r_plus1, r_plus2, r_plus3 = (
            (index[:, 0] + 1, index[:, 1]),
            (index[:, 0] + 2, index[:, 1]),
            (index[:, 0] + 3, index[:, 1]),
        )
        z_plus, z_minus = (index[:, 0], index[:, 1] + 1), (index[:, 0], index[:, 1] - 1)
        pred = self._epsilon_h2[self_index] * (
            phi[r_plus1] * CUPY_FLOAT(-108 / 9)
            + phi[r_plus2] * CUPY_FLOAT(27 / 9)
            + phi[r_plus2] * CUPY_FLOAT(4 / 9)
            + phi[z_plus]
            + phi[z_minus]
            + CUPY_FLOAT(-2 + 77 / 9) * phi[self_index]
        )
        for ion_type in self._ion_types:
            factor = getattr(self._grid.constant, "z_" + ion_type) * self._inv_epsilon0
            rho = getattr(self._grid.variable, "rho_" + ion_type).value[self_index]
            pred += factor * rho
        # Vector
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        vector[row_index] = -self._grid.field.rho_fixed[self_index] * self._inv_epsilon0
        vector[row_index] -= pred
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_pe_r_no_gradient_boundary(self, index):
        data, row, col = [], [], []
        z_shape = CUPY_INT(self._grid.shape[1])
        self_index = (index[:, 0], index[:, 1])
        row_index = (index[:, 0] * z_shape + index[:, 1]).astype(CUPY_INT)
        for i in range(3):
            row.append(row_index)
        # r-1
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(4))
        col.append(row_index - z_shape)
        # r-2
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(-1))
        col.append(row_index - CUPY_INT(2 * z_shape))
        # self
        data.append(self._epsilon_h2[self_index] * CUPY_FLOAT(-3))
        col.append(row_index)
        # Prediction
        phi = self._grid.variable.phi.value
        r_minus = (index[:, 0] - 1, index[:, 1])
        r_minus2 = (index[:, 0] - 2, index[:, 1])
        pred = self._epsilon_h2[self_index] * (
            phi[r_minus] * CUPY_FLOAT(4)
            - phi[r_minus2]
            + phi[self_index] * CUPY_FLOAT(-3)
        )
        # Vector
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        vector[row_index] -= pred
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _update_variable_factor(self):
        phi = self._grid.variable.phi.value
        self._dphi_dr[1:-1, :] = (phi[2:, :] - phi[:-2, :]) * self._inv_2h
        self._dphi_dz[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) * self._inv_2h
        self._curv_phi[1:-1, 1:-1] = self._inv_h2 * (
            phi[2:, 1:-1]
            + phi[:-2, 1:-1]
            + phi[1:-1, 2:]
            + phi[1:-1, :-2]
            - CUPY_FLOAT(4) * phi[1:-1, 1:-1]
        ) + self._inv_2hr[1:-1, 1:-1] * (phi[2:, 1:-1] - phi[:-2, 1:-1])
        self._curv_phi[0, 1:-1] = self._inv_h2 * (
            phi[1, 1:-1] * CUPY_FLOAT(8)
            - phi[2, 1:-1]
            + phi[0, 2:]
            + phi[0, :-2]
            + phi[0, 1:-1] * CUPY_FLOAT(-9)
        )

    def _get_npe_inner(self, ion_type, index):
        data, row, col = [], [], []
        # Read data
        z = CUPY_FLOAT(getattr(self._grid.constant, "z_" + ion_type))
        rho = getattr(self._grid.variable, "rho_" + ion_type).value
        phi = self._grid.variable.phi.value
        # Constant
        alpha = CUPY_FLOAT(z * self._grid.constant.beta)
        alpha_h = CUPY_FLOAT(alpha * self._inv_h)
        alpha_h2 = CUPY_FLOAT(alpha * self._inv_h2)
        row_offset = self._get_offset(ion_type)
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1] + row_offset).astype(CUPY_INT)
        for i in range(10):
            row.append(row_index)
        # Upwind
        self_index = (index[:, 0], index[:, 1])
        upwind_r = CUPY_FLOAT(1)
        upwind_z = CUPY_FLOAT(z / np.abs(z))
        conv_term_r = upwind_r * self._dphi_dr[self_index] * alpha_h
        conv_term_z = upwind_z * self._dphi_dz[self_index] * alpha_h
        # Index
        r_upwind_plus = (index[:, 0] + CUPY_INT(upwind_r), index[:, 1])
        z_upwind_plus = (index[:, 0], index[:, 1] + CUPY_INT(upwind_z))
        r_plus, r_minus = (index[:, 0] + 1, index[:, 1]), (index[:, 0] - 1, index[:, 1])
        z_plus, z_minus = (index[:, 0], index[:, 1] + 1), (index[:, 0], index[:, 1] - 1)
        # c r+1
        factor = self._inv_h2 + self._inv_2hr[self_index] + conv_term_r
        data.append(factor)
        col.append(row_index + z_shape)
        # c r-1
        factor = self._inv_h2 - self._inv_2hr[self_index]
        data.append(factor)
        col.append(row_index - z_shape)
        # c z+1
        factor = self._inv_h2 + cp.zeros(index.shape[0], CUPY_FLOAT)
        if upwind_z == 1:
            factor += conv_term_z
        data.append(factor)
        col.append(row_index + 1)
        # c z-1
        factor = self._inv_h2 + cp.zeros(index.shape[0], CUPY_FLOAT)
        if upwind_z != 1:
            factor += conv_term_z
        data.append(factor)
        col.append(row_index - 1)
        # c
        factor = -conv_term_r - conv_term_z
        factor -= self._inv_h2 * CUPY_FLOAT(4)
        factor += alpha * self._curv_phi[self_index]
        data.append(factor)
        col.append(row_index)
        # phi
        phi_row_index = row_index - row_offset
        c_h2 = self._inv_h2 * rho[self_index]
        c_2hr = self._inv_2hr[self_index] * rho[self_index]
        dc_dr_h = self._inv_h2 * upwind_r * (rho[r_upwind_plus] - rho[self_index])
        dc_dz_h = self._inv_h2 * upwind_z * (rho[z_upwind_plus] - rho[self_index])
        # phi r+1
        factor = (c_h2 + c_2hr + dc_dr_h * CUPY_FLOAT(0.5)) * alpha
        data.append(factor)
        col.append(phi_row_index + z_shape)
        # phi r-1
        factor = (c_h2 - c_2hr - dc_dr_h * CUPY_FLOAT(0.5)) * alpha
        data.append(factor)
        col.append(phi_row_index - z_shape)
        # phi z+1
        factor = (c_h2 + dc_dz_h * CUPY_FLOAT(0.5)) * alpha
        data.append(factor)
        col.append(phi_row_index + 1)
        # phi z-1
        factor = (c_h2 - dc_dz_h * CUPY_FLOAT(0.5)) * alpha
        data.append(factor)
        col.append(phi_row_index - 1)
        # phi
        data.append(c_h2 * CUPY_FLOAT(-4 * alpha))
        col.append(phi_row_index)
        # Prediction
        pred = self._inv_h2 * (
            rho[r_plus]
            + rho[r_minus]
            + rho[z_plus]
            + rho[z_minus]
            + rho[self_index] * CUPY_FLOAT(-4)
        )
        pred += self._inv_2hr[self_index] * (rho[r_plus] - rho[r_minus])
        pred += dc_dr_h * self._dphi_dr[self_index] * (alpha * self._h)
        pred += dc_dz_h * self._dphi_dz[self_index] * (alpha * self._h)
        pred += alpha * rho[self_index] * self._curv_phi[self_index]

        # Vector
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        vector[row_index] -= pred
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_npe_dirichlet(self, ion_type, index, value):
        row_offset = self._get_offset(ion_type)
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1] + row_offset).astype(CUPY_INT)
        size = CUPY_INT(index.shape[0])
        data = cp.ones(size, CUPY_FLOAT)
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        rho = getattr(self._grid.variable, "rho_" + ion_type).value
        vector[row_index] = value - rho[index[:, 0], index[:, 1]]
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row_index).astype(CUPY_INT),
            cp.hstack(row_index).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_npe_axial_symmetry_boundary(self, ion_type, index):
        data, row, col = [], [], []
        # Read data
        z = CUPY_FLOAT(getattr(self._grid.constant, "z_" + ion_type))
        rho = getattr(self._grid.variable, "rho_" + ion_type).value
        # Constant
        alpha = CUPY_FLOAT(z * self._grid.constant.beta)
        alpha_h = CUPY_FLOAT(alpha * self._inv_h)
        row_offset = self._get_offset(ion_type)
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1] + row_offset).astype(CUPY_INT)
        for i in range(10):
            row.append(row_index)
        # Upwind
        self_index = (index[:, 0], index[:, 1])
        upwind_z = CUPY_FLOAT(z / np.abs(z))
        conv_term_z = upwind_z * self._dphi_dz[self_index] * alpha_h
        # Index
        z_upwind_plus = (index[:, 0], index[:, 1] + CUPY_INT(upwind_z))
        r_plus, r_plus2 = (index[:, 0] + 1, index[:, 1]), (index[:, 0] + 2, index[:, 1])
        z_plus, z_minus = (index[:, 0], index[:, 1] + 1), (index[:, 0], index[:, 1] - 1)
        # c r+1
        zeros = cp.zeros(index.shape[0], CUPY_FLOAT)
        factor = self._inv_h2 * CUPY_FLOAT(8) + zeros
        data.append(factor)
        col.append(row_index + z_shape)
        # c r+2
        factor = self._inv_h2 * CUPY_FLOAT(-1) + zeros
        data.append(factor)
        col.append(row_index + CUPY_INT(2 * z_shape))
        # c z+1
        factor = self._inv_h2 + zeros
        if upwind_z == 1:
            factor += conv_term_z
        data.append(factor)
        col.append(row_index + 1)
        # c z-1
        factor = self._inv_h2 + zeros
        if upwind_z != 1:
            factor += conv_term_z
        data.append(factor)
        col.append(row_index - 1)
        # c
        factor = -conv_term_z - self._inv_h2 * CUPY_FLOAT(9)
        factor += alpha * self._curv_phi[self_index]
        data.append(factor)
        col.append(row_index)
        # phi
        phi_row_index = row_index - row_offset
        c_h2 = self._inv_h2 * rho[self_index]
        dc_dz_h = self._inv_h2 * upwind_z * (rho[z_upwind_plus] - rho[self_index])
        # phi r+1
        factor = c_h2 * (alpha * CUPY_FLOAT(8))
        data.append(factor)
        col.append(phi_row_index + z_shape)
        # phi r+2
        factor = c_h2 * (alpha * CUPY_FLOAT(-1))
        data.append(factor)
        col.append(phi_row_index + CUPY_INT(2 * z_shape))
        # phi z+1
        factor = (c_h2 + dc_dz_h * CUPY_FLOAT(0.5)) * alpha
        data.append(factor)
        col.append(phi_row_index + 1)
        # phi z-1
        factor = (c_h2 - dc_dz_h * CUPY_FLOAT(0.5)) * alpha
        data.append(factor)
        col.append(phi_row_index - 1)
        # phi
        data.append(c_h2 * (alpha * CUPY_FLOAT(-9)))
        col.append(phi_row_index)
        # Prediction
        pred = self._inv_h2 * (
            rho[r_plus] * CUPY_FLOAT(8)
            - rho[r_plus2]
            + rho[z_plus]
            + rho[z_minus]
            + rho[self_index] * CUPY_FLOAT(-9)
        )
        pred += dc_dz_h * self._dphi_dz[self_index] * (alpha * self._h)
        pred += alpha * rho[self_index] * self._curv_phi[self_index]
        # Vector
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        vector[row_index] -= pred
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_npe_r_no_flux_boundary(self, ion_type, index):
        data, row, col = [], [], []
        row_offset = self._get_offset(ion_type)
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1] + row_offset).astype(CUPY_INT)
        self_index = (index[:, 0], index[:, 1])
        for i in range(3):
            row.append(row_index)
        # r+1
        inv_h2 = cp.ones(index.shape[0], CUPY_FLOAT) + self._inv_h2
        data.append(inv_h2 * CUPY_FLOAT(4))
        col.append(row_index - z_shape)
        # r+2
        data.append(inv_h2 * CUPY_FLOAT(-1))
        col.append(row_index - CUPY_INT(2 * z_shape))
        # self
        data.append(inv_h2 * CUPY_FLOAT(-3))
        col.append(row_index)
        # Prediction
        rho = getattr(self._grid.variable, "rho_" + ion_type).value
        r_minus = (index[:, 0] - 1, index[:, 1])
        r_minus2 = (index[:, 0] - 2, index[:, 1])
        pred = inv_h2 * (
            rho[r_minus] * CUPY_FLOAT(4)
            - rho[r_minus2]
            + rho[self_index] * CUPY_FLOAT(-3)
        )
        # Vector
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        vector[row_index] -= pred
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _get_npe_z_no_gradient_boundary(self, ion_type, index, direction):
        data, row, col = [], [], []
        row_offset = self._get_offset(ion_type)
        z_shape = CUPY_INT(self._grid.shape[1])
        row_index = (index[:, 0] * z_shape + index[:, 1] + row_offset).astype(CUPY_INT)
        self_index = (index[:, 0], index[:, 1])
        for i in range(3):
            row.append(row_index)
        # z+1
        inv_h2 = cp.ones(index.shape[0], CUPY_FLOAT) + self._inv_h2
        data.append(inv_h2 * CUPY_FLOAT(4))
        col.append(row_index + direction)
        # a+2
        data.append(inv_h2 * CUPY_FLOAT(-1))
        col.append(row_index + direction + direction)
        # self
        data.append(inv_h2 * CUPY_FLOAT(-3))
        col.append(row_index)
        # Prediction
        rho = getattr(self._grid.variable, "rho_" + ion_type).value
        z_minus = (index[:, 0], index[:, 1] + direction)
        z_minus2 = (index[:, 0], index[:, 1] + direction + direction)
        pred = inv_h2 * (
            rho[z_minus] * CUPY_FLOAT(4)
            - rho[z_minus2]
            + rho[self_index] * CUPY_FLOAT(-3)
        )
        # Vector
        vector = cp.zeros(self._num_variables, CUPY_FLOAT)
        vector[row_index] -= pred
        # Return
        return (
            cp.hstack(data).astype(CUPY_FLOAT),
            cp.hstack(row).astype(CUPY_INT),
            cp.hstack(col).astype(CUPY_INT),
            vector.astype(CUPY_FLOAT),
        )

    def _assign_res(self, res):
        self._grid.variable.phi.value += res[: self._grid.num_points].reshape(
            self._grid.shape
        )
        for ion_type in self._ion_types:
            offset = self._get_offset(ion_type)
            rho = getattr(self._grid.variable, "rho_" + ion_type)
            rho.value += res[offset : offset + self._grid.num_points].reshape(
                self._grid.shape
            )
            # rho.value[rho.value <= 0] = 0

    def iterate(self, num_iterations, num_jacobian_iterations=5000, solver_freq=500):
        self._grid.check_requirement()
        self._update_constant_factor()
        for iteration in range(num_iterations):
            self._matrix, self._vector = self._get_equation()
            # res = spl.gmres(
            #     self._matrix,
            #     self._vector,
            #     maxiter=num_jacobian_iterations,
            #     restart=solver_freq,
            # )[0]
            res = spl.spsolve(self._matrix, self._vector)
            residual = self._matrix.dot(res) - self._vector
            residual = cp.abs(residual).mean()
            self._assign_res(res)
            print(iteration, residual, cp.abs(res).mean())

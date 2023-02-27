#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pnpe_cylinder.py
created time : 2023/01/27
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import numpy as np
import cupy as cp
from mdpy.utils import check_quantity_value
from mdpy.unit import *

from model import *
from model.core import Grid
from model.solver.pe_cylinder import PECylinderSolver
from model.solver.npe_cylinder import NPECylinderSolver


class PNPECylinderSolver:
    def __init__(self, grid: Grid, ion_types: list[str]) -> None:
        """All grid and constant in default unit
        ### Variable:
        - phi: Electric potential
            - inner: Inner points
            - dirichlet: Dirichlet point
                - `index`, `value` required
            - no-gradient: dphi/dz = 0
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor
        - rho_[ion]: Number density of [ion]
            - inner: Inner points
            - dirichlet: Dirichlet point, constant density
                - `index`, `value` required
            - no-flux: no flux boundary
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor

        ### Field:
        - epsilon: Relative permittivity
        - rho: Fixed charge density
        - rho_fixed: Fixed charge density
        - u_[ion]: External potential of [ion]
        - u_s: Steric potential

        ### Constant:
        - beta: 1/kBT
        - epsilon0 (added): Vacuum permittivity
        - r_[ion] (added): Radius of [ion]
        - z_[ion] (added): Valence of [ion]
        - d_[ion] (added): Diffusion coefficient of [ion]
        """
        # Input
        self._grid = grid
        self._ion_types = ion_types
        self._grid.add_requirement("field", "u_s")
        self._grid.add_requirement("field", "rho_fixed")
        for ion_type in ion_types:
            self._grid.add_requirement("constant", "r_%s" % ion_type)
            self._grid.add_requirement("constant", "z_%s" % ion_type)
            # Radius
            self._grid.add_constant(
                "r_%s" % ion_type,
                check_quantity_value(VDW_DICT[ion_type]["sigma"], default_length_unit),
            )
            # Valence
            self._grid.add_constant(
                "z_%s" % ion_type,
                check_quantity_value(ION_DICT[ion_type]["val"], VAL_UNIT),
            )
        # Create solver
        self._pe_solver = PECylinderSolver(grid=self._grid)
        self._npe_solver_list = [
            NPECylinderSolver(self._grid, ion_type=i) for i in self._ion_types
        ]
        # Create res list
        self._pre_res = []

    def _update_rho(self):
        self._grid.field.rho = self._grid.zeros_field(CUPY_FLOAT)
        for ion_type in self._ion_types:
            z = getattr(self._grid.constant, "z_%s" % ion_type)  # valence
            self._grid.field.rho += (
                CUPY_FLOAT(z) * getattr(self.grid.variable, "rho_%s" % ion_type).value
            )
        self._grid.field.rho += self._grid.field.rho_fixed

    def _update_u_s(self):
        self._grid.field.u_s = self._grid.zeros_field(CUPY_FLOAT)
        for ion_type in self._ion_types:
            r = getattr(self._grid.constant, "r_%s" % ion_type)
            v = CUPY_FLOAT(4 / 3 * np.pi * r**3)
            self._grid.field.u_s -= (
                v * getattr(self._grid.variable, "rho_%s" % ion_type).value
            )
        self._grid.field.u_s[self._grid.field.u_s <= 1e-5] = 1e-5
        self._grid.field.u_s[self._grid.field.u_s >= 1] = 1
        self._grid.field.u_s = -cp.log(self._grid.field.u_s)
        self._grid.field.u_s *= CUPY_FLOAT(1 / self.grid.constant.beta)

    def _update_u_ion(self, ion_type: str):
        z = getattr(self._grid.constant, "z_%s" % ion_type)  # valence
        u = self._grid.zeros_field(CUPY_FLOAT)
        # Electric energy
        u += CUPY_FLOAT(z) * self._grid.variable.phi.value
        # Steric energy
        # u += self._grid.field.u_s
        setattr(self._grid.field, "u_%s" % ion_type, u.astype(CUPY_FLOAT))

    # def iterate(self, num_iterations, num_sub_iterations=100, is_restart=False):
    #     self._grid.check_requirement()
    #     for iterations in range(num_iterations):
    #         self._pre_res = []
    #         self._pre_res.append(self._grid.variable.phi.value)
    #         # target = (
    #         #     self._ion_types[::-1] if iterations % 2 == 0 else self._ion_types[:]
    #         # )
    #         target = self._ion_types  # if iterations % 10 <= 5 else self._ion_types[:]
    #         for index, ion_type in enumerate(target):
    #             self._update_rho()
    #             self._update_u_s()
    #             self._pe_solver.iterate(
    #                 num_iterations=num_sub_iterations, is_restart=is_restart
    #             )
    #             self._update_u_ion(ion_type=ion_type)
    #             self._pre_res.append(
    #                 getattr(self._grid.variable, "rho_%s" % ion_type).value
    #             )
    #             self._npe_solver_list[index].iterate(
    #                 num_iterations=num_sub_iterations, is_restart=is_restart
    #             )
    #         print(self.residual)
    #     for ion_type in self._ion_types:
    #         self._update_u_ion(ion_type=ion_type)

    def iterate(self, num_iterations, num_sub_iterations=100, is_restart=False):
        self._grid.check_requirement()
        # Initial diffusion
        for iterations in range(num_iterations):
            self._pre_res = []
            self._pre_res.append(self._grid.variable.phi.value)
            self._pe_solver.iterate(num_iterations=5000, is_restart=is_restart)
            target = (
                self._ion_types[::-1] if iterations % 2 == 5 else self._ion_types[:]
            )
            target = self._ion_types
            for index, ion_type in enumerate(target):
                # self._update_u_s()
                self._update_u_ion(ion_type=ion_type)
                self._pre_res.append(
                    getattr(self._grid.variable, "rho_%s" % ion_type).value
                )
                self._npe_solver_list[index].iterate(
                    num_iterations=num_sub_iterations, is_restart=is_restart
                )
            self._update_rho()
            self._update_u_s()
            print(self.residual)
        for ion_type in self._ion_types:
            self._update_u_ion(ion_type=ion_type)

    def _iterate_single_ion(self, ion_type: str, num_sub_iterations):
        index = self._ion_types.index(ion_type)
        solver = self._npe_solver_list[index]
        print(index)
        for i in range(5):
            self._update_rho()
            self._pe_solver.iterate(num_iterations=num_sub_iterations, is_restart=True)
            self._update_u_ion(ion_type=ion_type)
            solver.iterate(num_iterations=num_sub_iterations, is_restart=True)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def ion_types(self) -> list[str]:
        return self._ion_types

    @property
    def pe_solver(self) -> PECylinderSolver:
        return self._pe_solver

    @property
    def npe_solver_list(self) -> list[NPECylinderSolver]:
        return self._npe_solver_list

    @property
    def residual(self):
        residual = 0
        self._cur_res = [self._grid.variable.phi.value] + [
            getattr(self._grid.variable, "rho_%s" % ion_type).value
            for ion_type in self._ion_types
        ]
        for i, j in zip(self._pre_res, self._cur_res):
            residual += cp.abs(i - j).mean()
        return residual

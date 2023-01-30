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

import os
import sys
import numpy as np
import cupy as cp
import numba.cuda as cuda
import mdpy as md
from mdpy.core import Grid
from mdpy.utils import check_quantity_value, check_quantity
from mdpy.environment import *
from mdpy.unit import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import visualize_concentration
from solver import *
from hydration import *
from pe_cylinder import PECylinderSolver
from npe_cylinder import NPECylinderSolver


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
        u += self._grid.field.u_s
        setattr(self._grid.field, "u_%s" % ion_type, u.astype(CUPY_FLOAT))

    def iterate(self, num_iterations, num_sub_iterations=100, is_restart=False):
        self._grid.check_requirement()
        for iterations in range(num_iterations):
            self._pre_res = []
            self._pre_res.append(self._grid.variable.phi.value)
            for index, ion_type in enumerate(self._ion_types):
                self._update_rho()
                self._update_u_s()
                self._pe_solver.iterate(
                    num_iterations=num_sub_iterations, is_restart=is_restart
                )
                self._update_u_ion(ion_type=ion_type)
                self._pre_res.append(
                    getattr(self._grid.variable, "rho_%s" % ion_type).value
                )
                self._npe_solver_list[index].iterate(
                    num_iterations=num_sub_iterations, is_restart=is_restart
                )
            print(self.residual)

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


def get_phi(grid: Grid, voltage):
    phi = grid.empty_variable()
    voltage = check_quantity_value(voltage, volt)
    voltage = (
        Quantity(voltage, volt * elementary_charge)
        .convert_to(default_energy_unit)
        .value
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


def get_rho(grid: Grid, density, r0, z0):
    density = check_quantity(density, mol / decimeter**3) * NA
    density = check_quantity_value(density, 1 / default_length_unit**3)
    rho = grid.empty_variable()
    field = grid.zeros_field().astype(CUPY_INT) - 1
    value = grid.zeros_field().astype(CUPY_FLOAT)
    dimension = grid.zeros_field().astype(CUPY_INT)
    direction = grid.zeros_field().astype(CUPY_INT)
    # 0: inner; 1: dirichlet; 2: no-flux
    r = grid.coordinate.r
    z = grid.coordinate.z
    index = cp.argwhere((r > r0) & (z < z0) & (z > -z0))
    r_min_index = index[:, 0].min()
    z_min_index = index[:, 1].min()
    z_max_index = index[:, 1].max()
    # Inner
    field[1 : r_min_index - 1, 1:-1] = 0
    field[r_min_index - 1 : -1, 1 : z_min_index - 1] = 0
    field[r_min_index - 1 : -1, z_max_index + 2 : -1] = 0
    # dirichlet
    field[index[:, 0], index[:, 1]] = 1
    value[index[:, 0], index[:, 1]] = 0
    field[:, [0, -1]] = 1
    value[:, [0, -1]] = density
    # no-flux
    # # z
    # field[r_min_index:-1, [z_min_index - 1, z_max_index + 1]] = 2
    # dimension[r_min_index:-1, [z_min_index - 1, z_max_index + 1]] = 1
    # direction[r_min_index:-1, z_min_index - 1] = -1
    # direction[r_min_index:-1, z_max_index + 1] = 1
    # # r
    # field[0, 1:-1] = 2
    # dimension[0, 1:-1] = 0
    # direction[0, 1:-1] = 1
    # field[r_min_index - 1, z_min_index - 1 : z_max_index + 2] = 2
    # dimension[r_min_index - 1, z_min_index - 1 : z_max_index + 2] = 0
    # direction[r_min_index - 1, z_min_index - 1 : z_max_index + 2] = -1
    # field[-1, 1:z_min_index] = 2
    # dimension[-1, 1:z_min_index] = 0
    # direction[-1, 1:z_min_index] = -1
    # field[-1, z_max_index + 1 :] = 2
    # dimension[-1, z_max_index + 1 : -1] = 0
    # direction[-1, z_max_index + 1 : -1] = -1

    field[-1, 1:-1] = 2
    dimension[-1, 1:-1] = 0
    direction[-1, 1:-1] = -1
    field[field == -1] = 0

    # axial-symmetry
    field[0, 1:-1] = 3
    dimension[0, 1:-1] = 0
    direction[0, 1:-1] = 1

    print(r_min_index, z_min_index, z_max_index, cp.argwhere(field == -2), field.shape)

    # print(cp.count_nonzero(field == -1))
    # import matplotlib.pyplot as plt

    # c = plt.imshow(field.get().T)
    # plt.colorbar(c)
    # plt.show()

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
        type="r-no-flux",
        index=index,
        # dimension=dimension[index[:, 0], index[:, 1]],
        direction=direction[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    rho.register_points(type="axial-symmetry", index=index)
    return rho


def get_epsilon(grid: Grid, r0, z0):
    epsilon = grid.ones_field() * CUPY_FLOAT(78)
    epsilon[(grid.coordinate.r >= r0) & (cp.abs(grid.coordinate.z) <= z0)] = 2
    return epsilon.astype(CUPY_FLOAT)


def visualize_concentration(grid: Grid, ion_types, iteration=None):
    import matplotlib
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(cur_dir, "out/image")
    if iteration is None:
        img_file_path = os.path.join(img_dir, "concentration.png")
    else:
        img_file_path = os.path.join(
            img_dir, "concentration-%s.png" % str(iteration).zfill(3)
        )
    print("Result save to", img_file_path)

    num_ions = len(ion_types)
    cmap = "RdBu"
    phi = (
        Quantity(
            (grid.variable.phi.value[1:-1, 1:-1]).get(),
            default_energy_unit / default_charge_unit,
        )
        .convert_to(volt)
        .value
    )
    fig, ax = plt.subplots(1, 1 + num_ions, figsize=[8 * (1 + num_ions), 8])
    big_font = 20
    mid_font = 15
    num_levels = 100
    r = grid.coordinate.r[1:-1, 1:-1].get()
    z = grid.coordinate.z[1:-1, 1:-1].get()
    c1 = ax[0].contourf(r, z, phi, num_levels, cmap=cmap)
    ax[0].set_title("Electric Potential", fontsize=big_font)
    ax[0].set_xlabel(r"x ($\AA$)", fontsize=big_font)
    ax[0].set_ylabel(r"z ($\AA$)", fontsize=big_font)
    ax[0].tick_params(labelsize=mid_font)

    rho_list = []
    for index, ion_type in enumerate(ion_types):
        rho = getattr(grid.variable, "rho_%s" % ion_type).value[1:-1, 1:-1].get()
        rho = (
            (Quantity(rho, 1 / default_length_unit**3) / NA)
            .convert_to(mol / decimeter**3)
            .value
        )
        rho_list.append(rho)
    rho = np.stack(rho_list)
    norm = matplotlib.colors.Normalize(vmin=rho.min(), vmax=rho.max())
    for index, ion_type in enumerate(ion_types):
        fig_index = index + 1
        c = ax[fig_index].contourf(r, z, rho[index], num_levels, norm=norm, cmap=cmap)
        ax[fig_index].set_title("%s density" % ion_type, fontsize=big_font)
        ax[fig_index].set_xlabel(r"x ($\AA$)", fontsize=big_font)
        ax[fig_index].tick_params(labelsize=mid_font)
    # Set info
    fig.subplots_adjust(left=0.12, right=0.9)
    position = fig.add_axes([0.05, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
    cb1 = fig.colorbar(c1, cax=position)
    cb1.ax.set_title(r"$\phi$ (V)", fontsize=big_font)
    cb1.ax.tick_params(labelsize=mid_font, labelleft=True, labelright=False)

    position = fig.add_axes([0.93, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
    cb2 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=position)
    cb2.ax.set_title("Density (mol/L)", fontsize=big_font)
    cb2.ax.tick_params(labelsize=mid_font)
    plt.savefig(img_file_path)
    plt.close()


def visualize_flux(
    grid: Grid,
    pnpe_solver: PNPECylinderSolver,
    ion_types: list[str],
    iteration=None,
    img_file_path=None,
):
    import matplotlib
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(cur_dir, "out/image")
    if img_file_path is None:
        if iteration is None:
            img_file_path = os.path.join(img_dir, "z-flux.png")
        else:
            img_file_path = os.path.join(
                img_dir, "z-flux-%s.png" % str(iteration).zfill(3)
            )
    print("Result save to", img_file_path)

    # Analysis
    num_ions = len(ion_types)
    flux = []
    convert = (
        Quantity(1, elementary_charge / default_time_unit / default_length_unit**2)
        .convert_to(ampere / default_length_unit**2)
        .value
    )
    for solver in pnpe_solver.npe_solver_list:
        flux.append(solver.get_flux(1) * convert)

    fig, ax = plt.subplots(1, num_ions, figsize=[8 * num_ions, 8])
    cmap = "RdBu"
    big_font = 20
    mid_font = 15
    num_levels = 200
    r = grid.coordinate.r[1:-1, 1:-1].get()
    z = grid.coordinate.z[1:-1, 1:-1].get()
    flux = cp.stack(flux).get()
    norm = matplotlib.colors.Normalize(vmin=flux.min(), vmax=flux.max())
    index = flux.shape[1] // 2
    for i, j in enumerate(ion_types):
        current = (
            CUPY_FLOAT(np.pi * 2 * grid.grid_width**2)
            * r[:, index]
            * flux[i][:, index]
        )
        current = current.sum()
        c = ax[i].contourf(r, z, flux[i], num_levels, norm=norm, cmap=cmap)
        ax[i].set_title(
            "%s z-current, current %.3e A" % (j, current), fontsize=big_font
        )
        ax[i].set_xlabel(r"x ($\AA$)", fontsize=big_font)
        ax[i].tick_params(labelsize=mid_font)
    fig.subplots_adjust(left=0.05, right=0.90)
    position = fig.add_axes([0.93, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=position)
    cb.ax.set_title(r"ampere/$A^2$", fontsize=big_font, rotation=90, x=-0.8, y=0.4)
    cb.ax.yaxis.get_offset_text().set(size=mid_font)
    cb.ax.tick_params(labelsize=mid_font)
    plt.savefig(img_file_path)
    plt.close()


if __name__ == "__main__":
    import time

    r0, z0 = 10, 25
    voltage = Quantity(10.0, volt)
    density = Quantity(0.15, mol / decimeter**3)
    beta = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    beta = 1 / beta
    ion_types = ["cl", "k"]
    grid = Grid(grid_width=0.5, r=[0, 50], z=[-100, 100])
    solver = PNPECylinderSolver(grid=grid, ion_types=ion_types)
    grid.add_variable("phi", get_phi(grid, voltage=voltage))
    grid.add_field("epsilon", get_epsilon(grid, r0, z0))
    grid.add_field("rho", grid.zeros_field(CUPY_FLOAT))
    grid.add_field("u_s", grid.zeros_field(CUPY_FLOAT))
    for ion_type in ion_types:
        grid.add_variable("rho_%s" % ion_type, get_rho(grid, density, r0, z0))
        grid.add_field("u_%s" % ion_type, grid.zeros_field(CUPY_FLOAT))
    grid.add_constant("beta", beta)

    for i in range(100):
        print("Iteration", i)
        solver.iterate(5, 10000, is_restart=True)
        visualize_concentration(grid, ion_types=ion_types, iteration=i)
        visualize_flux(grid, pnpe_solver=solver, ion_types=ion_types, iteration=i)

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : mpnpe.py
created time : 2022/11/10
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import numpy as np
import cupy as cp

# import cupyx.scipy.signal as signal
import scipy.signal as signal
from mdpy.core import Grid
from mdpy.utils import check_quantity_value
from mdpy.environment import *
from mdpy.unit import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydration import *
from pnpe import *
from utils import *


def visualize_result(grid: Grid, iteration):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(cur_dir, "out/image")
    img_file_path = os.path.join(img_dir, "%s.png" % str(iteration).zfill(3))

    half_index = grid.coordinate.x.shape[1] // 2
    target_slice = (
        slice(1, -1),
        half_index,
        slice(1, -1),
    )
    phi = (
        Quantity(
            (grid.field.phi[target_slice] + grid.field.phi_s[target_slice]).get(),
            default_energy_unit / default_charge_unit,
        )
        .convert_to(volt)
        .value
    )
    rho_k = (
        (Quantity(grid.field.rho_k[target_slice], 1 / default_length_unit**3) / NA)
        .convert_to(mol / decimeter**3)
        .value
    )
    rho_cl = (
        (Quantity(grid.field.rho_cl[target_slice], 1 / default_length_unit**3) / NA)
        .convert_to(mol / decimeter**3)
        .value
    )
    phi = rho_k + rho_cl

    fig, ax = plt.subplots(1, 3, figsize=[25, 8])
    big_font = 20
    mid_font = 15
    num_levels = 100
    x = grid.coordinate.x[target_slice].get()
    z = grid.coordinate.z[target_slice].get()
    c1 = ax[0].contourf(x, z, phi, num_levels)
    ax[0].set_title("Electric Potential", fontsize=big_font)
    ax[0].set_xlabel(r"x ($\AA$)", fontsize=big_font)
    ax[0].set_ylabel(r"z ($\AA$)", fontsize=big_font)
    ax[0].tick_params(labelsize=mid_font)

    rho = np.stack([rho_k, rho_cl])
    norm = matplotlib.colors.Normalize(vmin=rho.min(), vmax=rho.max())
    c2 = ax[1].contourf(x, z, rho_k, num_levels, norm=norm)
    ax[1].set_title("POT density", fontsize=big_font)
    ax[1].set_xlabel(r"x ($\AA$)", fontsize=big_font)
    ax[1].tick_params(labelsize=mid_font)
    c3 = ax[2].contourf(x, z, rho_cl, num_levels, norm=norm)
    ax[2].set_title("CLA density", fontsize=big_font)
    ax[2].set_xlabel(r"x ($\AA$)", fontsize=big_font)
    ax[2].tick_params(labelsize=mid_font)
    fig.subplots_adjust(left=0.12, right=0.9)
    position = fig.add_axes([0.05, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
    cb1 = fig.colorbar(c1, cax=position)
    cb1.ax.set_title(r"$\phi$ (V)", fontsize=big_font)
    cb1.ax.tick_params(labelsize=mid_font, labelleft=True, labelright=False)

    position = fig.add_axes([0.93, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
    cb2 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm), cax=position)
    cb2.ax.set_title("Density (mol/L)", fontsize=big_font)
    cb2.ax.tick_params(labelsize=mid_font)
    plt.savefig(img_file_path)
    plt.close()


class MPNPESolver:
    def __init__(self, grid: Grid, ion_types: list[str]) -> None:
        """All grid and constant in default unit

        Field:
        - phi: Electric potential
        - phi_s: Steric potential
        - epsilon: Relative permittivity
        - mask: Channel shape, 0 while grid in pore area, 1 while grid in solvent area
        - rho_fix: Number density of fixed particle
        - rho_[ion]: Number density of [ion]
        - vdw_[ion]: VDW potential between pore and [ion]
        - hyd_[ion]: Hydration potential between pore and [ion]

        Constant:
        - beta: 1/kBT
        - epsilon0 (added): Vacuum permittivity
        - r_[ion] (added): Radius of [ion]
        - val_[ion] (added): Valence of [ion]
        - d_[ion] (added): Diffusion coefficient of [ion]
        """
        self._grid = grid
        self._ion_types = ion_types
        field_name_list = ["phi", "phi_s", "epsilon", "mask", "rho_fix"]
        constant_name_list = ["epsilon0", "beta"]
        # Add epsilon0
        epsilon0 = EPSILON0.convert_to(
            default_charge_unit**2 / (default_energy_unit * default_length_unit)
        ).value
        self._grid.add_constant("epsilon0", epsilon0)
        # Add ion
        for ion_type in self._ion_types:
            if not ion_type in ION_DICT.keys():
                raise KeyError("Ion %s is not supported" % ion_type)
            field_name_list.append("rho_%s" % ion_type)
            field_name_list.append("vdw_%s" % ion_type)
            # Radius
            r_name = "r_%s" % ion_type
            constant_name_list.append(r_name)
            self._grid.add_constant(
                r_name,
                check_quantity_value(VDW_DICT[ion_type]["sigma"], default_length_unit),
            )
            # Diffusion coefficient
            d_name = "d_%s" % ion_type
            constant_name_list.append(d_name)
            self._grid.add_constant(
                d_name, check_quantity_value(ION_DICT[ion_type]["d"], DIFFUSION_UNIT)
            )
            # Valence
            val_name = "val_%s" % ion_type
            constant_name_list.append(val_name)
            self._grid.add_constant(
                val_name, check_quantity_value(ION_DICT[ion_type]["val"], VAL_UNIT)
            )
        self._grid.set_requirement(
            field_name_list=field_name_list, constant_name_list=constant_name_list
        )

    def _generate_pe_coefficient(self):
        factor = NUMPY_FLOAT(0.5 / self._grid.grid_width**2)
        pre_factor = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        inv_denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        target_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                # 0 for plus and 1 for minus
                target_slice[i] = slice(2, None) if j == 0 else slice(0, -2)
                pre_factor[i, j] = factor * (
                    self._grid.field.epsilon[tuple(target_slice)]
                    + self._grid.field.epsilon[1:-1, 1:-1, 1:-1]
                )
                inv_denominator += pre_factor[i, j]
            target_slice[i] = slice(1, -1)
        inv_denominator = CUPY_FLOAT(1) / inv_denominator
        return pre_factor, inv_denominator

    def _solve_pe(
        self,
        pre_factor,
        inv_denominator,
        soa_factor: NUMPY_FLOAT,
        num_iterations: int = 200,
    ):
        soa_factor_a = NUMPY_FLOAT(soa_factor)
        soa_factor_b = NUMPY_FLOAT(1 - soa_factor)
        scaled_rho = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        scaled_rho += self._grid.field.rho_fix[1:-1, 1:-1, 1:-1]
        for ion_type in self._ion_types:
            scaled_rho += (
                getattr(self._grid.constant, "val_%s" % ion_type)
                * getattr(self._grid.field, "rho_%s" % ion_type)[1:-1, 1:-1, 1:-1]
            )
        scaled_rho *= NUMPY_FLOAT(1 / self._grid.constant.epsilon0)
        for iteration in range(num_iterations):
            nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            # X Neumann condition
            nominator[:-1, :, :] += (
                pre_factor[0, 0, :-1, :, :] * self._grid.field.phi[2:-1, 1:-1, 1:-1]
            )
            nominator[-1, :, :] += pre_factor[0, 0, -1, :, :] * (
                self._grid.field.phi[-2, 1:-1, 1:-1]
                + self._grid.field.phi[-1, 1:-1, 1:-1] * self._grid.grid_width
            )
            nominator[1:, :, :] += (
                pre_factor[0, 1, 1:, :, :] * self._grid.field.phi[1:-2, 1:-1, 1:-1]
            )
            nominator[0, :, :] += pre_factor[0, 1, 0, :, :] * (
                self._grid.field.phi[1, 1:-1, 1:-1]
                - self._grid.field.phi[0, 1:-1, 1:-1] * self._grid.grid_width
            )
            # Y Neumann
            nominator[:, :-1, :] += (
                pre_factor[1, 0, :, :-1, :] * self._grid.field.phi[1:-1, 2:-1, 1:-1]
            )
            nominator[:, -1, :] += pre_factor[1, 0, :, -1, :] * (
                self._grid.field.phi[1:-1, -2, 1:-1]
                + self._grid.field.phi[1:-1, -1, 1:-1] * self._grid.grid_width
            )
            nominator[:, 1:, :] += (
                pre_factor[1, 1, :, 1:, :] * self._grid.field.phi[1:-1, 1:-2, 1:-1]
            )
            nominator[:, 0, :] += pre_factor[1, 1, :, 0, :] * (
                self._grid.field.phi[1:-1, 1, 1:-1]
                - self._grid.field.phi[1:-1, 0, 1:-1] * self._grid.grid_width
            )
            # Z Dirichlet
            nominator += pre_factor[2, 0] * self._grid.field.phi[1:-1, 1:-1, 2:]
            nominator += pre_factor[2, 1] * self._grid.field.phi[1:-1, 1:-1, :-2]
            # Source term
            nominator += scaled_rho
            self._grid.field.phi[1:-1, 1:-1, 1:-1] = (
                soa_factor_a * self._grid.field.phi[1:-1, 1:-1, 1:-1]
                + soa_factor_b * nominator * inv_denominator
            )

    def _generate_npe_coefficient(self, ion_type: str):
        d_ion = getattr(self._grid.constant, "d_%s" % ion_type)
        factor = NUMPY_FLOAT(0.5 / self._grid.grid_width**2) * d_ion
        pre_factor = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        target_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        for i in range(self._grid.num_dimensions):
            for j in [0, 1]:
                # 0 for plus and 1 for minus
                target_slice[i] = slice(2, None) if j == 0 else slice(0, -2)
                pre_factor[i, j] = (
                    (
                        self._grid.field.mask[tuple(target_slice)]
                        == self._grid.field.mask[1:-1, 1:-1, 1:-1]
                    )
                    & (self._grid.field.mask[1:-1, 1:-1, 1:-1] == 1)
                ) * factor
            target_slice[i] = slice(1, -1)
        return pre_factor

    def _update_steric_potential(self):
        phi_s = cp.ones(self._grid.inner_shape, CUPY_FLOAT)
        for ion_type in self._ion_types:
            r = getattr(self._grid.constant, "r_%s" % ion_type)
            v = CUPY_FLOAT(4 / 3 * np.pi * r**3)
            # v = CUPY_FLOAT(r**3)
            phi_s -= (
                v * getattr(self._grid.field, "rho_%s" % ion_type)[1:-1, 1:-1, 1:-1]
            )
        phi_s[phi_s <= 1e-10] = 1e-10
        phi_s[phi_s >= 1] = 1
        phi_s = -cp.log(phi_s) / self.grid.constant.beta
        self._grid.field.phi_s[1:-1, 1:-1, 1:-1] = phi_s.astype(CUPY_FLOAT)
        self_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        target_slice = [slice(None, None) for i in range(self._grid.num_dimensions)]
        for i in range(self._grid.num_dimensions):
            target_slice[i] = 0
            self_slice[i] = 0
            self._grid.field.phi_s[tuple(self_slice)] = phi_s[tuple(target_slice)]
            target_slice[i] = -1
            self_slice[i] = -1
            self._grid.field.phi_s[tuple(self_slice)] = phi_s[tuple(target_slice)]
            target_slice[i] = slice(None, None)
            self_slice[i] = slice(1, -1)

    def _get_potential_grad(self, ion_type: str):
        val_ion = getattr(self._grid.constant, "val_%s" % ion_type)
        vdw_ion = getattr(self._grid.field, "vdw_%s" % ion_type)
        hyd_ion = getattr(self._grid.field, "hyd_%s" % ion_type)
        potential_grad = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        potential = self._grid.field.phi * val_ion
        ## X Neumann condition
        potential_grad[0, 0, :-1, :, :] = potential[2:-1, 1:-1, 1:-1]
        potential_grad[0, 0, -1, :, :] = (
            potential[-2, 1:-1, 1:-1]
            + potential[-1, 1:-1, 1:-1] * self._grid.grid_width
        )
        potential_grad[0, 1, 1:, :, :] = potential[1:-2, 1:-1, 1:-1]
        potential_grad[0, 1, 0, :, :] = (
            potential[1, 1:-1, 1:-1] - potential[0, 1:-1, 1:-1] * self._grid.grid_width
        )
        ## Y Neumann
        potential_grad[1, 0, :, :-1, :] = potential[1:-1, 2:-1, 1:-1]
        potential_grad[1, 0, :, -1, :] = (
            potential[1:-1, -2, 1:-1]
            + potential[1:-1, -1, 1:-1] * self._grid.grid_width
        )
        potential_grad[1, 1, :, 1:, :] = potential[1:-1, 1:-2, 1:-1]
        potential_grad[1, 1, :, 0, :] = (
            potential[1:-1, 1, 1:-1] - potential[1:-1, 0, 1:-1] * self._grid.grid_width
        )
        ## Z Dirichlet
        potential_grad[2, 0] = potential[1:-1, 1:-1, 2:]
        potential_grad[2, 1] = potential[1:-1, 1:-1, :-2]
        potential_grad -= potential[1:-1, 1:-1, 1:-1]
        # External field
        ext_potential = hyd_ion + self._grid.field.phi_s + vdw_ion
        potential_grad[0, 0] += ext_potential[2:, 1:-1, 1:-1]
        potential_grad[0, 1] += ext_potential[:-2, 1:-1, 1:-1]
        potential_grad[1, 0] += ext_potential[1:-1, 2:, 1:-1]
        potential_grad[1, 1] += ext_potential[1:-1, :-2, 1:-1]
        potential_grad[2, 0] += ext_potential[1:-1, 1:-1, 2:]
        potential_grad[2, 1] += ext_potential[1:-1, 1:-1, :-2]
        potential_grad -= ext_potential[1:-1, 1:-1, 1:-1]
        potential_grad *= self._grid.constant.beta * 0.5
        return potential_grad

    def _solve_npe(
        self,
        ion_type: str,
        pre_factor,
        soa_factor: NUMPY_FLOAT,
        num_iterations: int = 200,
    ):
        potential_grad = self._get_potential_grad(ion_type)
        rho_ion = getattr(self._grid.field, "rho_%s" % ion_type)
        soa_factor_a = NUMPY_FLOAT(soa_factor)
        soa_factor_b = NUMPY_FLOAT(1 - soa_factor)
        # Denominator
        potential_grad = CUPY_FLOAT(1) - potential_grad  # 1 - dV
        inv_denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                inv_denominator += pre_factor[i, j] * potential_grad[i, j]
        # For non-zero denominator, Add a small value for non-pore area
        threshold = -1e-8
        inv_denominator += (self._grid.field.mask[1:-1, 1:-1, 1:-1] - 1) * threshold
        inv_denominator = CUPY_FLOAT(1) / inv_denominator
        # Iteration solve
        # dV = (1 + dV) * prefactor
        potential_grad = CUPY_FLOAT(2) - potential_grad
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                potential_grad[i, j] *= pre_factor[i, j]
        for iteration in range(num_iterations):
            nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            # X Neumann condition
            nominator[:-1, :, :] += (
                potential_grad[0, 0, :-1, :, :] * rho_ion[2:-1, 1:-1, 1:-1]
            )
            nominator[-1, :, :] += potential_grad[0, 0, -1, :, :] * (
                rho_ion[-2, 1:-1, 1:-1]
                + rho_ion[-1, 1:-1, 1:-1] * self._grid.grid_width
            )
            nominator[1:, :, :] += (
                potential_grad[0, 1, 1:, :, :] * rho_ion[1:-2, 1:-1, 1:-1]
            )
            nominator[0, :, :] += potential_grad[0, 1, 0, :, :] * (
                rho_ion[1, 1:-1, 1:-1] - rho_ion[0, 1:-1, 1:-1] * self._grid.grid_width
            )
            # Y Neumann
            nominator[:, :-1, :] += (
                potential_grad[1, 0, :, :-1, :] * rho_ion[1:-1, 2:-1, 1:-1]
            )
            nominator[:, -1, :] += potential_grad[1, 0, :, -1, :] * (
                rho_ion[1:-1, -2, 1:-1]
                + rho_ion[1:-1, -1, 1:-1] * self._grid.grid_width
            )
            nominator[:, 1:, :] += (
                potential_grad[1, 1, :, 1:, :] * rho_ion[1:-1, 1:-2, 1:-1]
            )
            nominator[:, 0, :] += potential_grad[1, 1, :, 0, :] * (
                rho_ion[1:-1, 1, 1:-1] - rho_ion[1:-1, 0, 1:-1] * self._grid.grid_width
            )
            # Z Dirichlet
            nominator += potential_grad[2, 0] * rho_ion[1:-1, 1:-1, 2:]
            nominator += potential_grad[2, 1] * rho_ion[1:-1, 1:-1, :-2]
            # if iteration % 100 == 0:
            #     print(nominator)
            new = (
                soa_factor_a * rho_ion[1:-1, 1:-1, 1:-1]
                + soa_factor_b * nominator * inv_denominator
            )
            new[new >= NP_DENSITY_UPPER_THRESHOLD] = NP_DENSITY_UPPER_THRESHOLD
            new[new <= NP_DENSITY_LOWER_THRESHOLD] = NP_DENSITY_LOWER_THRESHOLD
            rho_ion[1:-1, 1:-1, 1:-1] = new

    def solve(self, soa_factor=NUMPY_FLOAT(0.01)):
        pe_pre_factor, pe_inv_denominator = self._generate_pe_coefficient()
        npe_pre_factor_list = [
            self._generate_npe_coefficient(i) for i in self._ion_types
        ]
        num_iterations = 100
        for i in range(10000):
            self._solve_pe(
                pre_factor=pe_pre_factor,
                inv_denominator=pe_inv_denominator,
                soa_factor=soa_factor,
                num_iterations=num_iterations,
            )
            self._update_steric_potential()
            for index, ion_type in enumerate(self._ion_types):
                self._solve_npe(
                    ion_type=ion_type,
                    pre_factor=npe_pre_factor_list[index],
                    soa_factor=soa_factor,
                    num_iterations=num_iterations,
                )
            if (i * num_iterations) % 2500 == 0:
                print(i)
                visualize_result(self._grid, i)

    @property
    def grid(self) -> Grid:
        return self._grid


def get_hyd(grid: Grid, json_dir: str, ion_type: str, r0, z0, thickness):
    pak = np
    potential = grid.zeros_field().get()
    n_bulk = (
        (Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton))
        .convert_to(1 / default_length_unit**3)
        .value
    )
    targets = ["oxygen", "hydrogen"]
    for target in targets:
        n0 = n_bulk if target == "oxygen" else n_bulk * 2
        pore_file_path = os.path.join(json_dir, "%s-pore.json" % target)
        ion_file_path = os.path.join(json_dir, "%s-%s.json" % (target, ion_type))
        g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
        g_ion = HydrationDistributionFunction(json_file_path=ion_file_path)
        r_cut = get_sigmoid_length(g_ion.bulk_alpha) + g_ion.bulk_rb + 2
        x_ion, y_ion, z_ion = pak.meshgrid(
            pak.arange(-r_cut, r_cut + grid.grid_width, grid.grid_width),
            pak.arange(-r_cut, r_cut + grid.grid_width, grid.grid_width),
            pak.arange(-r_cut, r_cut + grid.grid_width, grid.grid_width),
            indexing="ij",
        )
        coordinate_range = grid.coordinate_range.copy()
        coordinate_range[:, 0] -= r_cut
        coordinate_range[:, 1] += r_cut
        x_extend, y_extend, z_extend = pak.meshgrid(
            pak.arange(
                coordinate_range[0, 0],
                coordinate_range[0, 1] + grid.grid_width,
                grid.grid_width,
            ),
            pak.arange(
                coordinate_range[1, 0],
                coordinate_range[1, 1] + grid.grid_width,
                grid.grid_width,
            ),
            pak.arange(
                coordinate_range[2, 0],
                coordinate_range[2, 1] + grid.grid_width,
                grid.grid_width,
            ),
            indexing="ij",
        )
        # Convolve
        pore_distance = get_pore_distance(
            x_extend, y_extend, z_extend, r0=r0, z0=z0, thickness=thickness
        )
        ion_distance = get_distance(x_ion, y_ion, z_ion)
        f = g_pore(pore_distance)
        g = g_ion(ion_distance)
        g = g * pak.log(g)
        g = -(Quantity(300 * g, kelvin) * KB).convert_to(default_energy_unit).value
        energy_factor = grid.grid_width**3 * n0
        potential += (g.sum() - signal.fftconvolve(f, g, "valid")) * energy_factor
    return cp.array(potential, CUPY_FLOAT)


def get_vdw(grid: Grid, type1: str, type2: str, distance):
    threshold = Quantity(5, kilocalorie_permol).convert_to(default_energy_unit).value
    sigma1 = check_quantity_value(VDW_DICT[type1]["sigma"], default_length_unit)
    epsilon1 = check_quantity_value(VDW_DICT[type1]["epsilon"], default_energy_unit)
    sigma2 = check_quantity_value(VDW_DICT[type2]["sigma"], default_length_unit)
    epsilon2 = check_quantity_value(VDW_DICT[type2]["epsilon"], default_energy_unit)
    sigma = 0.5 * (sigma1 + sigma2)
    epsilon = np.sqrt(epsilon1 * epsilon2)
    scaled_distance = (sigma / (distance + 0.01)) ** 6
    vdw = grid.zeros_field()
    vdw[:, :, :] = 4 * epsilon * (scaled_distance**2 - scaled_distance)
    vdw[vdw > threshold] = threshold
    print(Quantity(vdw.min(), default_energy_unit).convert_to(kilojoule_permol).value)
    return vdw


def get_epsilon(grid: Grid, pore_distance: cp.ndarray):
    epsilon = grid.ones_field() * 78
    return epsilon.astype(CUPY_FLOAT)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(cur_dir, "../out")
    # Prefactor
    temperature = 300  # kelvin
    r0 = 10
    z0 = 10
    voltage = 1  # V
    thickness = 2.0
    rho_bulk = Quantity(0.15, mol / decimeter**3)
    grid_width = 0.25
    grid_range = np.array([[-20, 20], [-20, 20], [-30, 30]])
    # Create grid
    grid = Grid(
        grid_width=grid_width,
        x=grid_range[0],
        y=grid_range[1],
        z=grid_range[2],
    )
    # Create solver
    ion_types = ["k", "cl"]
    solver = MPNPESolver(grid, ion_types=ion_types)
    # Add beta
    beta = (Quantity(temperature, kelvin) * KB).convert_to(default_energy_unit)
    beta = CUPY_FLOAT(1 / beta.value)
    solver.grid.add_constant("beta", beta)
    # Add field
    rho_bulk = (rho_bulk * NA).convert_to(1 / default_length_unit**3).value
    pore_distance = get_pore_distance(
        x=solver.grid.coordinate.x,
        y=solver.grid.coordinate.y,
        z=solver.grid.coordinate.z,
        r0=r0,
        z0=z0,
        thickness=thickness,
    )
    solver.grid.add_field("phi", get_phi(grid, voltage))
    solver.grid.add_field("phi_s", grid.zeros_field())
    solver.grid.add_field("epsilon", get_epsilon(grid, pore_distance))
    solver.grid.add_field("mask", get_mask(grid, pore_distance))
    solver.grid.add_field("rho_fix", get_rho_fix(grid))
    solver.grid.add_field("rho_k", get_rho_ion(grid, rho_bulk))
    solver.grid.add_field("rho_cl", get_rho_ion(grid, rho_bulk))
    solver.grid.add_field("vdw_k", get_vdw(grid, "c", "k", pore_distance))
    solver.grid.add_field("vdw_cl", get_vdw(grid, "c", "cl", pore_distance))
    solver.grid.add_field("hyd_k", get_hyd(grid, json_dir, "pot", r0, z0, thickness))
    solver.grid.add_field("hyd_cl", get_hyd(grid, json_dir, "cla", r0, z0, thickness))

    solver.solve()

    fig, ax = plt.subplots(1, 1)
    half_index = grid.coordinate.x.shape[1] // 2
    if True:
        target_slice = (
            half_index,
            slice(1, -1),
            slice(1, -1),
        )
        print(pore_distance[half_index, 0, half_index])
        c = ax.contourf(
            grid.coordinate.y[target_slice].get(),
            grid.coordinate.z[target_slice].get(),
            # pore_distance[target_slice].get(),
            grid.field.hyd_k[target_slice],
            200,
        )
        fig.colorbar(c)
    else:
        target_slice = (
            half_index,
            slice(1, -1),
            half_index,
        )
        ax.plot(
            grid.coordinate.y[target_slice].get(),
            Quantity(grid.field.hyd_k[target_slice].get(), default_energy_unit)
            .convert_to(kilocalorie_permol)
            .value,
            200,
        )
        ax.set_ylim([-1, 4])
    plt.show()

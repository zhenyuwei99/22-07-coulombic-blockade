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
import numba.cuda as cuda
import scipy.signal as signal
import mdpy as md
from mdpy.core import Grid
from mdpy.utils import check_quantity_value
from mdpy.environment import *
from mdpy.unit import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy.hdf import *
from solver import *
from analysis import *
from utils import *
from pnpe import get_mask, get_phi, get_rho_fix, get_rho_ion


class MPNPESolver:
    def __init__(self, grid: Grid, ion_types: list[str], is_pnp=False) -> None:
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
        self._is_pnp = is_pnp
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

    def _get_mask(self):
        mask = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        target_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        for i in range(self._grid.num_dimensions):
            for j in [0, 1]:
                # 0 for plus and 1 for minus
                target_slice[i] = slice(2, None) if j == 0 else slice(0, -2)
                mask[i, j] = (
                    self._grid.field.mask[tuple(target_slice)]
                    == self._grid.field.mask[1:-1, 1:-1, 1:-1]
                ) & (self._grid.field.mask[1:-1, 1:-1, 1:-1] == 1)
            target_slice[i] = slice(1, -1)
        # mask = cp.ones(
        #     [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        # )
        return mask

    def _update_phi_s(self):
        phi_s = cp.ones(self._grid.inner_shape, CUPY_FLOAT)
        for ion_type in self._ion_types:
            r = getattr(self._grid.constant, "r_%s" % ion_type)
            v = CUPY_FLOAT(4 / 3 * np.pi * r**3)
            phi_s -= (
                v * getattr(self._grid.field, "rho_%s" % ion_type)[1:-1, 1:-1, 1:-1]
            )
        phi_s[phi_s <= 1e-5] = 1e-5
        phi_s[phi_s >= 1] = 1
        phi_s = -cp.log(phi_s) / self.grid.constant.beta
        self._grid.field.phi_s[1:-1, 1:-1, 1:-1] = phi_s.astype(CUPY_FLOAT)
        self_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        target_slice = [slice(None, None) for i in range(self._grid.num_dimensions)]
        # Set boundary value
        for i in range(self._grid.num_dimensions):
            target_slice[i] = 0
            self_slice[i] = 0
            self._grid.field.phi_s[tuple(self_slice)] = phi_s[tuple(target_slice)]
            target_slice[i] = -1
            self_slice[i] = -1
            self._grid.field.phi_s[tuple(self_slice)] = phi_s[tuple(target_slice)]
            target_slice[i] = slice(None, None)
            self_slice[i] = slice(1, -1)

    def _get_phi_grad(self):
        phi_grad = cp.zeros(
            [
                self._grid.num_dimensions,
                self._grid.inner_shape[0] + 1,
                self._grid.inner_shape[1] + 1,
                self._grid.inner_shape[2] + 1,
            ],
            CUPY_FLOAT,
        )
        # X Neumann condition
        phi_grad[0, 1:-1, :-1, :-1] = (
            self._grid.field.phi[2:-1, 1:-1, 1:-1]
            - self._grid.field.phi[1:-2, 1:-1, 1:-1]
        )
        phi_grad[0, 0, :-1, :-1] = self._grid.field.phi[0, 1:-1, 1:-1]
        phi_grad[0, -1, :-1, :-1] = self._grid.field.phi[-1, 1:-1, 1:-1]
        # Y Neumann
        phi_grad[1, :-1, 1:-1, :-1] = (
            self._grid.field.phi[1:-1, 2:-1, 1:-1]
            - self._grid.field.phi[1:-1, 1:-2, 1:-1]
        )
        phi_grad[1, :-1, 0, :-1] = self._grid.field.phi[1:-1, 0, 1:-1]
        phi_grad[1, :-1, -1, :-1] = self._grid.field.phi[1:-1, -1, 1:-1]
        # Z Dirichlet
        phi_grad[2, :-1, :-1, :] = (
            self._grid.field.phi[1:-1, 1:-1, 1:] - self._grid.field.phi[1:-1, 1:-1, :-1]
        )
        phi_grad *= CUPY_FLOAT(self._grid.constant.beta)
        return phi_grad.astype(CUPY_FLOAT)

    def _get_potential_grad(self, ion_type: str, phi_grad: cp.ndarray):
        potential_grad = cp.zeros(
            [
                self._grid.num_dimensions,
                self._grid.inner_shape[0] + 1,
                self._grid.inner_shape[1] + 1,
                self._grid.inner_shape[2] + 1,
            ],
            CUPY_FLOAT,
        )
        potential = self._grid.zeros_field()
        val_ion = getattr(self._grid.constant, "val_%s" % ion_type)
        potential += self._grid.field.phi_s
        if not self._is_pnp:
            # vdw_ion = getattr(self._grid.field, "vdw_%s" % ion_type)
            hyd_ion = getattr(self._grid.field, "hyd_%s" % ion_type)
            # External potential
            potential += hyd_ion  # + vdw_ion
        potential_grad[0, :, :-1, :-1] = (
            potential[1:, 1:-1, 1:-1] - potential[:-1, 1:-1, 1:-1]
        )
        potential_grad[1, :-1, :, :-1] = (
            potential[1:-1, 1:, 1:-1] - potential[1:-1, :-1, 1:-1]
        )
        potential_grad[2, :-1, :-1, :] = (
            potential[1:-1, 1:-1, 1:] - potential[1:-1, 1:-1, :-1]
        )
        potential_grad *= CUPY_FLOAT(self._grid.constant.beta)
        # Electrostatic potential
        potential_grad += phi_grad * CUPY_FLOAT(val_ion)
        potential_grad[(potential_grad < 1e-5) & (potential_grad > 0)] = 1e-5
        potential_grad[(potential_grad > -1e-5) & (potential_grad <= 0)] = -1e-5
        return potential_grad.astype(CUPY_FLOAT)

    def _solve_npe(
        self,
        ion_type: str,
        mask: cp.ndarray,
        phi_grad: cp.ndarray,
        soa_factor: NUMPY_FLOAT,
        num_iterations: int = 200,
    ):
        potential_grad = self._get_potential_grad(ion_type, phi_grad=phi_grad)
        rho_ion = getattr(self._grid.field, "rho_%s" % ion_type)
        soa_factor_a = NUMPY_FLOAT(soa_factor)
        soa_factor_b = NUMPY_FLOAT(1 - soa_factor)
        # Pre factor and inv_denominator
        pre_factor = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        exp_res = potential_grad / (cp.exp(potential_grad) - CUPY_FLOAT(1))
        # self._test_field(exp_res[1, :-1, :-1, :-1])
        pre_factor[0, 0] = exp_res[0, 1:, :-1, :-1]
        pre_factor[0, 1] = exp_res[0, :-1, :-1, :-1]
        pre_factor[1, 0] = exp_res[1, :-1, 1:, :-1]
        pre_factor[1, 1] = exp_res[1, :-1, :-1, :-1]
        pre_factor[2, 0] = exp_res[2, :-1, :-1, 1:]
        pre_factor[2, 1] = exp_res[2, :-1, :-1, :-1]

        inv_denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        # f(-u_i) = u_i - f(u_i)
        # Prefactor
        potential_grad_slice = [0, slice(None, -1), slice(None, -1), slice(None, -1)]
        for i in range(self._grid.num_dimensions):
            potential_grad_slice[0] = i
            potential_grad_slice[i + 1] = slice(1, None)
            inv_denominator += pre_factor[i, 0]  # f(u_i)
            pre_factor[i, 0] = mask[i, 0] * (
                potential_grad[tuple(potential_grad_slice)] + pre_factor[i, 0]
            )  # f(-u_i)
            potential_grad_slice[i + 1] = slice(None, -1)
            inv_denominator += (
                potential_grad[tuple(potential_grad_slice)] + pre_factor[i, 1]
            )  # f(-u_{i-1})
            pre_factor[i, 1] = mask[i, 1] * pre_factor[i, 1]  # f(u_{i-1})
        inv_denominator = CUPY_FLOAT(1) / inv_denominator
        # Iteration solve
        for iteration in range(num_iterations):
            nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            # X Neumann condition
            nominator[:-1, :, :] += (
                pre_factor[0, 0, :-1, :, :] * rho_ion[2:-1, 1:-1, 1:-1]
            )
            nominator[-1, :, :] += pre_factor[0, 0, -1, :, :] * (
                rho_ion[-2, 1:-1, 1:-1]
                + rho_ion[-1, 1:-1, 1:-1] * self._grid.grid_width
            )
            nominator[1:, :, :] += (
                pre_factor[0, 1, 1:, :, :] * rho_ion[1:-2, 1:-1, 1:-1]
            )
            nominator[0, :, :] += pre_factor[0, 1, 0, :, :] * (
                rho_ion[1, 1:-1, 1:-1] - rho_ion[0, 1:-1, 1:-1] * self._grid.grid_width
            )
            # Y Neumann
            nominator[:, :-1, :] += (
                pre_factor[1, 0, :, :-1, :] * rho_ion[1:-1, 2:-1, 1:-1]
            )
            nominator[:, -1, :] += pre_factor[1, 0, :, -1, :] * (
                rho_ion[1:-1, -2, 1:-1]
                + rho_ion[1:-1, -1, 1:-1] * self._grid.grid_width
            )
            nominator[:, 1:, :] += (
                pre_factor[1, 1, :, 1:, :] * rho_ion[1:-1, 1:-2, 1:-1]
            )
            nominator[:, 0, :] += pre_factor[1, 1, :, 0, :] * (
                rho_ion[1:-1, 1, 1:-1] - rho_ion[1:-1, 0, 1:-1] * self._grid.grid_width
            )
            # Z Dirichlet for 0 and Neumann for -1
            # nominator[:, :, :-1] += (
            #     pre_factor[2, 0, :, :, :-1] * rho_ion[1:-1, 1:-1, 2:-1]
            # )
            # nominator[:, :, -1] += pre_factor[2, 0, :, :, -1] * (
            #     rho_ion[1:-1, 1:-1, -2]
            #     + rho_ion[1:-1, 1:-1, -1] * self._grid.grid_width
            # )
            nominator += pre_factor[2, 0] * rho_ion[1:-1, 1:-1, 2:]
            # nominator[:, :, 1:] += (
            #     pre_factor[2, 1, :, :, 1:] * rho_ion[1:-1, 1:-1, 1:-2]
            # )
            # nominator[:, :, 0] += pre_factor[2, 1, :, :, 0] * (
            #     rho_ion[1:-1, 1:-1, 1] - rho_ion[1:-1, 1:-1, 0] * self._grid.grid_width
            # )
            nominator += pre_factor[2, 1] * rho_ion[1:-1, 1:-1, :-2]
            # if iteration % 100 == 0:
            #     print(nominator)
            new = (
                soa_factor_a * rho_ion[1:-1, 1:-1, 1:-1]
                + soa_factor_b * nominator * inv_denominator
            )
            new[new >= NP_DENSITY_UPPER_THRESHOLD] = NP_DENSITY_UPPER_THRESHOLD
            new[new <= NP_DENSITY_LOWER_THRESHOLD] = NP_DENSITY_LOWER_THRESHOLD
            rho_ion[1:-1, 1:-1, 1:-1] = new

    def _test_field(self, field):
        half_index = self._grid.coordinate.x.shape[1] // 2
        fig, ax = plt.subplots(1, 1, figsize=[8, 4])
        num_levels = 100
        x = grid.coordinate.x[1:-1, half_index, 1:-1].get()
        z = grid.coordinate.z[1:-1, half_index, 1:-1].get()
        if field.shape[0] == x.shape[0]:
            target_field = field[:, half_index - 1, :].get()
        else:
            target_field = field[1:-1, half_index, 1:-1].get()
        c = ax.contourf(x, z, target_field, num_levels)
        fig.colorbar(c)
        plt.show()

    def iterate(
        self,
        num_epochs: int,
        num_iterations_per_equation: int = 50,
        soa_factor: float = NUMPY_FLOAT(0.01),
    ):
        self._grid.field.mask = self._grid.field.mask.astype(cp.bool8)
        pe_pre_factor, pe_inv_denominator = self._generate_pe_coefficient()
        mask = self._get_mask()
        for i in range(num_epochs):
            self._solve_pe(
                pre_factor=pe_pre_factor,
                inv_denominator=pe_inv_denominator,
                soa_factor=soa_factor,
                num_iterations=num_iterations_per_equation,
            )
            self._update_phi_s()
            phi_grad = self._get_phi_grad()
            for index, ion_type in enumerate(self._ion_types):
                self._solve_npe(
                    ion_type=ion_type,
                    mask=mask,
                    phi_grad=phi_grad,
                    soa_factor=soa_factor,
                    num_iterations=num_iterations_per_equation,
                )

    def _get_energy(self):
        energy = CUPY_FLOAT(0)
        for ion_type in self._ion_types:
            # Entropy
            rho_ion = getattr(self._grid.field, "rho_%s" % ion_type)
            energy += (rho_ion * cp.log(rho_ion)).sum()
            # External potential
            val_ion = getattr(self._grid.constant, "val_%s" % ion_type)
            potential = val_ion * self._grid.field.phi
            potential += self._grid.field.phi_s
            if not self._is_pnp:
                vdw_ion = getattr(self._grid.field, "vdw_%s" % ion_type)
                hyd_ion = getattr(self._grid.field, "hyd_%s" % ion_type)
                potential += hyd_ion
            energy += (potential * rho_ion).sum()

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
    scaled_distance = (sigma / (distance + 0.0001)) ** 6
    vdw = grid.zeros_field()
    vdw[:, :, :] = 4 * epsilon * (scaled_distance**2 - scaled_distance)
    vdw[vdw > threshold] = threshold
    # print(Quantity(vdw.min(), default_energy_unit).convert_to(kilojoule_permol).value)
    return vdw


def get_epsilon(grid: Grid, pore_distance: cp.ndarray):
    epsilon = grid.ones_field()
    pore_index = pore_distance > 0
    epsilon[pore_index] = 78
    epsilon[~pore_index] = 2
    # epsilon = grid.ones_field() * 2
    return epsilon.astype(CUPY_FLOAT)


def get_phi(grid: Grid, voltage=1):
    phi = grid.zeros_field()
    phi[:, :, -1] = (
        Quantity(voltage / 2, volt)
        .convert_to(default_energy_unit / default_charge_unit)
        .value
    )
    phi[:, :, -1] = (
        Quantity(-voltage / 2, volt)
        .convert_to(default_energy_unit / default_charge_unit)
        .value
    )
    return phi


def get_rho_ion(grid: Grid, rho_bulk: float):
    rho_ion = grid.zeros_field()
    rho_ion[:, :, 0] = rho_bulk
    rho_ion[:, :, -1] = rho_bulk
    return rho_ion


if __name__ == "__main__":
    from itertools import product

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(cur_dir, "../out")
    out_dir = os.path.join(cur_dir, "out")
    is_skip = not True
    # Prefactor
    temperature = 300  # kelvin
    if not True:
        r0_list = [i * CC_BOND_LENGTH * 3 / (2 * np.pi) for i in [15]]
        # voltage_list = [-15, -10, -7.5, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 7.5, 10, 15]
        voltage_list = [0.5, 1, 2, 5, 7.5, 10, 15, 20]
        is_pnp_list = [False, True]
    else:
        r0_list = [i * CC_BOND_LENGTH * 3 / (2 * np.pi) for i in [15]]
        voltage_list = [-7.5]
        is_pnp_list = [False]
    z0 = 25
    zs = 30
    w0 = 100
    thickness = 2.0
    rho_bulk = Quantity(0.15, mol / decimeter**3)
    grid_width = 0.5
    grid_range = np.array([[-w0, w0], [-w0, w0], [-z0 - zs, z0 + zs]])
    ion_types = ["k", "cl"]
    num_cations = 0
    for ion_type in ion_types:
        num_cations += 0 if ion_type == "cl" else 1
    job_list = list(product(r0_list, voltage_list, is_pnp_list))
    num_jobs = len(job_list)
    print("r0 list", r0_list)
    print("%s jobs in total" % num_jobs)
    for i, (r0, voltage, is_pnp) in enumerate(job_list):
        name = generate_name(
            ion_types=ion_types,
            r0=r0,
            w0=w0,
            zs=zs,
            z0=z0,
            rho=rho_bulk,
            voltage=voltage,
            grid_width=grid_width,
            is_pnp=is_pnp,
        )
        file_path = os.path.join(out_dir, name + ".grid")
        if os.path.exists(file_path) and is_skip:
            print("Skip %s as result existed" % name)
            continue
        # Create grid
        grid = Grid(
            grid_width=grid_width,
            x=grid_range[0],
            y=grid_range[1],
            z=grid_range[2],
        )
        pore_distance = get_pore_distance(
            x=grid.coordinate.x,
            y=grid.coordinate.y,
            z=grid.coordinate.z,
            r0=r0,
            z0=z0,
            thickness=thickness,
        )
        hyd = get_hyd(grid, json_dir, "pot", r0, z0, thickness)
        import matplotlib.pyplot as plt

        hyd = (
            Quantity(hyd.get(), default_energy_unit)
            .convert_to(kilocalorie_permol)
            .value
        )
        half_index = grid.shape[1] // 2
        # plt.contour(
        #     grid.coordinate.x[:, half_index, :].get(),
        #     grid.coordinate.z[:, half_index, :].get(),
        #     hyd[:, half_index, :],
        #     200,
        # )
        half_index_z = grid.shape[2] // 2
        plt.plot(
            grid.coordinate.x[:, half_index, half_index_z].get(),
            hyd[:, half_index, half_index_z],
        )
        plt.show()

        print()
        # Create solver
        solver = MPNPESolver(grid, ion_types=ion_types, is_pnp=is_pnp)
        # Add beta
        beta = (Quantity(temperature, kelvin) * KB).convert_to(default_energy_unit)
        beta = CUPY_FLOAT(1 / beta.value)
        solver.grid.add_constant("beta", beta)
        # Add field
        rho_bulk_val = (rho_bulk * NA).convert_to(1 / default_length_unit**3).value
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
        for ion_type in ion_types:
            if ion_type != "cl":
                solver.grid.add_field(
                    "rho_%s" % ion_type, get_rho_ion(grid, rho_bulk_val)
                )
            else:
                solver.grid.add_field(
                    "rho_%s" % ion_type, get_rho_ion(grid, rho_bulk_val * num_cations)
                )
            solver.grid.add_field(
                "vdw_%s" % ion_type, get_vdw(grid, "c", ion_type, pore_distance)
            )
            solver.grid.add_field(
                "hyd_%s" % ion_type,
                get_hyd(grid, json_dir, "pot", r0, z0, thickness),
            )
        if not True:
            solver._grid = md.io.GridParser(file_path).grid
        total_iterations = 300000
        num_iter_per_eq = 50
        num_epochs = total_iterations // num_iter_per_eq
        visualize_freq = 100
        for iteration in range(num_epochs // visualize_freq):
            visualize_flux(
                solver.grid, iteration=iteration * visualize_freq, ion_types=ion_types
            )
            visualize_concentration(
                solver.grid,
                iteration=iteration * visualize_freq,
                ion_types=ion_types,
            )
            solver.iterate(
                num_epochs=visualize_freq, num_iterations_per_equation=num_iter_per_eq
            )
        writer = md.io.GridWriter(file_path)
        writer.write(grid=solver.grid)
        print(
            "Finish %s (%d/%d, %.3f %%)"
            % (name, i + 1, num_jobs, (i + 1) / num_jobs * 100)
        )

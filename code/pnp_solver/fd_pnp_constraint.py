#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : fd_pnp_constraint.py
created time : 2022/07/17
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import os
import math
import numpy as np
import numba as nb
import numba.cuda as cuda
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import Ensemble, Grid
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *


BSPLINE_ORDER = 4
NUM_GRIDS_PER_BLOCK = 8
DATA_SHAPE = NUM_GRIDS_PER_BLOCK + 2
NP_DENSITY_THRESHOLD = (
    (Quantity(500, mol / decimeter ** 3) * NA)
    .convert_to(1 / default_length_unit ** 3)
    .value
)


def visualize_pnp_solution(grid, file_path: str):
    big_font = 20
    mid_font = 15
    index = grid.inner_shape[1] // 2
    fig, ax = plt.subplots(1, 3, figsize=[25, 8])
    c1 = ax[0].contourf(
        grid.coordinate.x[1:-1, index, 1:-1].get(),
        grid.coordinate.z[1:-1, index, 1:-1].get(),
        Quantity(
            grid.field.electric_potential[1:-1, index, 1:-1].get(),
            default_energy_unit / default_charge_unit,
        )
        .convert_to(volt)
        .value,
        200,
    )
    ax[0].set_title("Electric Potential", fontsize=big_font)
    ax[0].set_xlabel(r"x ($\AA$)", fontsize=big_font)
    ax[0].set_ylabel(r"z ($\AA$)", fontsize=big_font)
    ax[0].tick_params(labelsize=mid_font)
    pot_density = (
        (
            Quantity(
                grid.field.pot_density[1:-1, index, 1:-1].get(),
                1 / default_length_unit ** 3,
            )
            / NA
        )
        .convert_to(mol / decimeter ** 3)
        .value
    )
    cla_density = (
        (
            Quantity(
                grid.field.cla_density[1:-1, index, 1:-1].get(),
                1 / default_length_unit ** 3,
            )
            / NA
        )
        .convert_to(mol / decimeter ** 3)
        .value
    )
    max1 = float(pot_density.max())
    max2 = float(cla_density.max())
    max = max1 if max1 > max2 else max2

    min1 = float(pot_density.min())
    min2 = float(cla_density.min())
    min = min1 if min1 < min2 else min2
    norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
    c2 = ax[1].contourf(
        grid.coordinate.x[1:-1, index, 1:-1].get(),
        grid.coordinate.z[1:-1, index, 1:-1].get(),
        pot_density,
        200,
        norm=norm,
    )
    ax[1].set_title("POT density", fontsize=big_font)
    ax[1].set_xlabel(r"x ($\AA$)", fontsize=big_font)
    ax[1].tick_params(labelsize=mid_font)
    c3 = ax[2].contourf(
        grid.coordinate.x[1:-1, index, 1:-1].get(),
        grid.coordinate.z[1:-1, index, 1:-1].get(),
        cla_density,
        200,
        norm=norm,
    )
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
    # fig.tight_layout()
    plt.savefig(file_path)
    plt.close()


class FDPoissonNernstPlanckConstraint(Constraint):
    def __init__(self, temperature: Quantity, grid: Grid, **ion_type) -> None:
        super().__init__()
        # Temperature
        self._temperature = check_quantity(temperature, kelvin)
        self._beta = 1 / (self._temperature * KB).convert_to(default_energy_unit).value
        self._k0 = 1 / (4 * np.pi * EPSILON0.value)
        self._temperature = self._temperature.value
        # Set grid
        self._grid = grid
        self._grid.set_requirement(
            {
                "electric_potential": {
                    "require_gradient": False,
                    "require_curvature": False,
                },
                "relative_permittivity": {
                    "require_gradient": False,
                    "require_curvature": False,
                },
                "charge_density": {
                    "require_gradient": False,
                    "require_curvature": False,
                },
                "channel_shape": {
                    "require_gradient": False,
                    "require_curvature": False,
                },
            }
        )
        # Set ion type
        self._ion_type_list = list(ion_type.keys())
        self._ion_valence_list = list(ion_type.values())
        self._num_ion_types = len(self._ion_type_list)
        for key in ion_type.keys():
            self._grid.requirement[key + "_density"] = {
                "require_gradient": False,
                "require_curvature": False,
            }
            self._grid.requirement[key + "_diffusion_coefficient"] = {
                "require_gradient": False,
                "require_curvature": False,
            }
        self._bspline_interpretation = cuda.jit(
            nb.void(
                NUMBA_FLOAT[:, ::1],  # charges
                NUMBA_FLOAT[:, ::1],  # positions
                NUMBA_FLOAT[:, ::1],  # pbc_matrix
                NUMBA_INT[::1],  # grid_shape
                NUMBA_FLOAT[:, :, ::1],  # charge_density
            )
        )(self._bspline_interpretation_kernel)
        # Flag
        self._is_nernst_plank_equation_within_tolerance = False
        self._is_verbose = False
        self._log_file = None
        self._is_img = False
        self._img_dir = None

    def set_log_file(self, file_path: str, mode: str = "w"):
        self._is_verbose = True
        self._log_file = file_path
        open(self._log_file, mode).close()

    def set_img_dir(self, dir_path: str):
        self._is_img = True
        self._img_dir = dir_path
        if not os.path.exists(self._img_dir):
            os.mkdir(self._img_dir)

    def _dump_log(self, text: str):
        with open(self._log_file, "a") as f:
            print(text, file=f)

    def __repr__(self) -> str:
        return "<mdpy.constraint.FDPoissonNernstPlanckConstraint object>"

    def __str__(self) -> str:
        return "FD Poisson-Nernst-Planck constraint"

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._forces = cp.zeros(
            [self._parent_ensemble.topology.num_particles, SPATIAL_DIM], CUPY_FLOAT
        )
        self._potential_energy = cp.array([0], CUPY_FLOAT)

    @staticmethod
    def _bspline_interpretation_kernel(
        charges, positions, pbc_matrix, grid_shape, charge_density
    ):
        particle_id = cuda.grid(1)
        thread_x = cuda.threadIdx.x
        if particle_id >= positions.shape[0]:
            return None
        # Shared array
        shared_pbc_matrix = cuda.shared.array((SPATIAL_DIM), NUMBA_FLOAT)
        shared_grid_shape = cuda.shared.array((SPATIAL_DIM), NUMBA_INT)
        if thread_x == 0:
            for i in range(SPATIAL_DIM):
                shared_pbc_matrix[i] = pbc_matrix[i, i]
        elif thread_x == 1:
            for i in range(SPATIAL_DIM):
                shared_grid_shape[i] = grid_shape[i]
        cuda.syncthreads()
        # Grid information
        grid_index = cuda.local.array((SPATIAL_DIM), NUMBA_INT)
        grid_fraction = cuda.local.array((SPATIAL_DIM), NUMBA_FLOAT)
        local_target_grid = cuda.local.array((SPATIAL_DIM, BSPLINE_ORDER), NUMBA_INT)
        for i in range(SPATIAL_DIM):
            index = (
                positions[particle_id, i] * shared_grid_shape[i] / shared_pbc_matrix[i]
            )
            grid_index[i] = math.floor(index)
            grid_fraction[i] = index - grid_index[i]
            for j in range(BSPLINE_ORDER):
                index = grid_index[i] + j - 1
                if index >= shared_grid_shape[i]:
                    index -= shared_grid_shape[i]
                elif index < 0:
                    index += shared_grid_shape[i]
                local_target_grid[i, j] = index
        # Spline coefficient
        local_spline_coefficient = cuda.local.array(
            (SPATIAL_DIM, BSPLINE_ORDER), NUMBA_FLOAT
        )
        # 3 order B-spline
        for i in range(SPATIAL_DIM):
            local_spline_coefficient[i, 2] = NUMBA_FLOAT(0.5) * grid_fraction[i] ** 2
            local_spline_coefficient[i, 0] = (
                NUMBA_FLOAT(0.5) * (1 - grid_fraction[i]) ** 2
            )
            local_spline_coefficient[i, 1] = (
                NUMBA_FLOAT(1)
                - local_spline_coefficient[i, 0]
                - local_spline_coefficient[i, 2]
            )
        # 4 order spline coefficient
        for i in range(SPATIAL_DIM):
            local_spline_coefficient[i, 3] = (
                grid_fraction[i] * local_spline_coefficient[i, 2] / NUMBA_FLOAT(3)
            )
            local_spline_coefficient[i, 2] = (
                (NUMBA_FLOAT(1) + grid_fraction[i]) * local_spline_coefficient[i, 1]
                + (NUMBA_FLOAT(3) - grid_fraction[i]) * local_spline_coefficient[i, 2]
            ) / 3
            local_spline_coefficient[i, 0] = (
                (NUMBA_FLOAT(1) - grid_fraction[i])
                * local_spline_coefficient[i, 0]
                / NUMBA_FLOAT(3)
            )
            local_spline_coefficient[i, 1] = NUMBA_FLOAT(1) - (
                local_spline_coefficient[i, 0]
                + local_spline_coefficient[i, 2]
                + local_spline_coefficient[i, 3]
            )
        # Assign charge
        charge = charges[particle_id, 0]
        for i in range(BSPLINE_ORDER):
            grid_x = local_target_grid[0, i]
            charge_x = charge * local_spline_coefficient[0, i]
            for j in range(BSPLINE_ORDER):
                grid_y = local_target_grid[1, j]
                charge_xy = charge_x * local_spline_coefficient[1, j]
                for k in range(BSPLINE_ORDER):
                    grid_z = local_target_grid[2, k]
                    charge_xyz = charge_xy * local_spline_coefficient[2, k]
                    cuda.atomic.add(
                        charge_density, (grid_x, grid_y, grid_z), charge_xyz
                    )

    def _generate_poisson_equation_coefficient(self):
        inv_square = (1 / self._grid.grid_width) ** 2
        pre_factor = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        inv_denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        target_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                # 0 for plus and 1 for minus
                target_slice[i] = slice(2, None) if j == 0 else slice(0, -2)
                pre_factor[i, j] = (0.5 * inv_square[i]) * (
                    self._grid.field.relative_permittivity[tuple(target_slice)]
                    + self._grid.field.relative_permittivity[1:-1, 1:-1, 1:-1]
                )
                inv_denominator += pre_factor[i, j]
        inv_denominator = CUPY_FLOAT(1) / inv_denominator
        return pre_factor, inv_denominator

    def _solve_poisson_equation(
        self, pre_factor, inv_denominator, max_iterations=200, soa_factor=0.05
    ):
        # Charge density
        charge_density = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        for i in range(self._num_ion_types):
            attribute_name = self._ion_type_list[i] + "_density"
            charge_density += (
                self._ion_valence_list[i]
                * getattr(self._grid.field, attribute_name)[1:-1, 1:-1, 1:-1]
            )
        charge_density += self._grid.field.charge_density[1:-1, 1:-1, 1:-1]
        charge_density *= self._k0
        # Iteration
        for iteration in range(max_iterations):
            nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            # X Neumann condition
            nominator[:-1, :, :] += (
                pre_factor[0, 0, :-1, :, :]
                * self._grid.field.electric_potential[2:-1, 1:-1, 1:-1]
            )
            nominator[-1, :, :] += pre_factor[0, 0, -1, :, :] * (
                self._grid.field.electric_potential[-2, 1:-1, 1:-1]
                + self._grid.field.electric_potential[-1, 1:-1, 1:-1]
                * self._grid.grid_width[0]
            )
            nominator[1:, :, :] += (
                pre_factor[0, 1, 1:, :, :]
                * self._grid.field.electric_potential[1:-2, 1:-1, 1:-1]
            )
            nominator[0, :, :] += pre_factor[0, 1, 1, :, :] * (
                self._grid.field.electric_potential[1, 1:-1, 1:-1]
                - self._grid.field.electric_potential[0, 1:-1, 1:-1]
                * self._grid.grid_width[0]
            )
            # Y Neumann
            nominator[:, :-1, :] += (
                pre_factor[1, 0, :, :-1, :]
                * self._grid.field.electric_potential[1:-1, 2:-1, 1:-1]
            )
            nominator[:, -1, :] += pre_factor[1, 0, :, -1, :] * (
                self._grid.field.electric_potential[1:-1, -2, 1:-1]
                + self._grid.field.electric_potential[1:-1, -1, 1:-1]
                * self._grid.grid_width[1]
            )
            nominator[:, 1:, :] += (
                pre_factor[1, 1, :, 1:, :]
                * self._grid.field.electric_potential[1:-1, 1:-2, 1:-1]
            )
            nominator[:, 0, :] += pre_factor[1, 1, :, -1, :] * (
                self._grid.field.electric_potential[1:-1, 1, 1:-1]
                - self._grid.field.electric_potential[1:-1, 0, 1:-1]
                * self._grid.grid_width[1]
            )
            # Z Dirichlet
            nominator += (
                pre_factor[2, 0] * self._grid.field.electric_potential[1:-1, 1:-1, 2:]
            )
            nominator += (
                pre_factor[2, 1] * self._grid.field.electric_potential[1:-1, 1:-1, :-2]
            )
            nominator += charge_density
            self._grid.field.electric_potential[1:-1, 1:-1, 1:-1] = (
                soa_factor * self._grid.field.electric_potential[1:-1, 1:-1, 1:-1]
                + (1 - soa_factor) * nominator * inv_denominator
            )

    def _generate_nernst_plank_equation_coefficient(self, ion_type: str):
        inv_square = (1 / self._grid.grid_width) ** 2
        diffusion_coefficient = getattr(
            self._grid.field, ion_type + "_diffusion_coefficient"
        )
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
                        (
                            self._grid.field.channel_shape[tuple(target_slice)]
                            == self._grid.field.channel_shape[1:-1, 1:-1, 1:-1]
                        )
                        & (self._grid.field.channel_shape[1:-1, 1:-1, 1:-1] == 0)
                    )
                    * (0.5 * inv_square[i])
                    * (
                        diffusion_coefficient[tuple(target_slice)]
                        + diffusion_coefficient[1:-1, 1:-1, 1:-1]
                    )
                )
                target_slice[i] = slice(1, -1)
        return pre_factor

    def _check_nernst_plank_equation_judgement(self):
        # Only judge once
        if not self._is_nernst_plank_equation_within_tolerance:
            is_within_tolerance = True
            for ion_type in self._ion_type_list:
                ion_density = getattr(self._grid.field, ion_type + "_density")
                if cp.count_nonzero(ion_density >= NP_DENSITY_THRESHOLD):
                    is_within_tolerance = False
            self._is_nernst_plank_equation_within_tolerance = is_within_tolerance

    def _solve_nernst_plank_equation(
        self, ion_type: str, pre_factor, max_iterations=200, soa_factor=0.05,
    ):
        # Read input
        ion_valence = self._ion_valence_list[self._ion_type_list.index(ion_type)]
        ion_density = getattr(self._grid.field, ion_type + "_density")
        energy = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        ## X Neumann condition
        energy[0, 0, :-1, :, :] = self._grid.field.electric_potential[2:-1, 1:-1, 1:-1]
        energy[0, 0, -1, :, :] = (
            self._grid.field.electric_potential[-2, 1:-1, 1:-1]
            + self._grid.field.electric_potential[-1, 1:-1, 1:-1]
            * self._grid.grid_width[0]
        )
        energy[0, 1, 1:, :, :] = self._grid.field.electric_potential[1:-2, 1:-1, 1:-1]
        energy[0, 1, 0, :, :] = (
            self._grid.field.electric_potential[1, 1:-1, 1:-1]
            - self._grid.field.electric_potential[0, 1:-1, 1:-1]
            * self._grid.grid_width[0]
        )
        ## Y Neumann
        energy[1, 0, :, :-1, :] = self._grid.field.electric_potential[1:-1, 2:-1, 1:-1]
        energy[1, 0, :, -1, :] = (
            self._grid.field.electric_potential[1:-1, -2, 1:-1]
            + self._grid.field.electric_potential[1:-1, -1, 1:-1]
            * self._grid.grid_width[1]
        )
        energy[1, 1, :, 1:, :] = self._grid.field.electric_potential[1:-1, 1:-2, 1:-1]
        energy[1, 1, :, 0, :] = (
            self._grid.field.electric_potential[1:-1, 1, 1:-1]
            - self._grid.field.electric_potential[1:-1, 0, 1:-1]
            * self._grid.grid_width[1]
        )
        ## Z Dirichlet
        energy[2, 0] = self._grid.field.electric_potential[1:-1, 1:-1, 2:]
        energy[2, 1] = self._grid.field.electric_potential[1:-1, 1:-1, :-2]
        energy -= self._grid.field.electric_potential[1:-1, 1:-1, 1:-1]
        energy *= self._beta * 0.5 * ion_valence
        # Denominator
        energy = CUPY_FLOAT(1) - energy  # 1 - V
        inv_denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                inv_denominator += pre_factor[i, j] * energy[i, j]
        # For non-zero denominator, Add a small value for non-pore area
        threshold = 1e-8
        inv_denominator += self._grid.field.channel_shape[1:-1, 1:-1, 1:-1] * threshold
        inv_denominator = CUPY_FLOAT(1) / inv_denominator
        energy = CUPY_FLOAT(2) - energy  # 1 + V
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                energy[i, j] *= pre_factor[i, j]
        for iteration in range(max_iterations):
            nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            # X Neumann condition
            nominator[:-1, :, :] += (
                energy[0, 0, :-1, :, :] * ion_density[2:-1, 1:-1, 1:-1]
            )
            nominator[-1, :, :] += energy[0, 0, -1, :, :] * (
                ion_density[-2, 1:-1, 1:-1]
                + ion_density[-1, 1:-1, 1:-1] * self._grid.grid_width[0]
            )
            nominator[1:, :, :] += (
                energy[0, 1, 1:, :, :] * ion_density[1:-2, 1:-1, 1:-1]
            )
            nominator[0, :, :] += energy[0, 1, 1, :, :] * (
                ion_density[1, 1:-1, 1:-1]
                - ion_density[0, 1:-1, 1:-1] * self._grid.grid_width[0]
            )
            # Y Neumann
            nominator[:, :-1, :] += (
                energy[1, 0, :, :-1, :] * ion_density[1:-1, 2:-1, 1:-1]
            )
            nominator[:, -1, :] += energy[1, 0, :, -1, :] * (
                ion_density[1:-1, -2, 1:-1]
                + ion_density[1:-1, -1, 1:-1] * self._grid.grid_width[1]
            )
            nominator[:, 1:, :] += (
                energy[1, 1, :, 1:, :] * ion_density[1:-1, 1:-2, 1:-1]
            )
            nominator[:, 0, :] += energy[1, 1, :, -1, :] * (
                ion_density[1:-1, 1, 1:-1]
                - ion_density[1:-1, 0, 1:-1] * self._grid.grid_width[1]
            )
            # Z Dirichlet
            nominator += energy[2, 0] * ion_density[1:-1, 1:-1, 2:]
            nominator += energy[2, 1] * ion_density[1:-1, 1:-1, :-2]
            new = (
                soa_factor * ion_density[1:-1, 1:-1, 1:-1]
                + (1 - soa_factor) * nominator * inv_denominator
            )
            # For converge, avoiding extreme large result in the beginning of iteration
            if not self._is_nernst_plank_equation_within_tolerance:
                new[new >= NP_DENSITY_THRESHOLD] = NP_DENSITY_THRESHOLD
                new[new <= -NP_DENSITY_THRESHOLD] = -NP_DENSITY_THRESHOLD
            ion_density[1:-1, 1:-1, 1:-1] = new

    def _update_charge_density(self):
        # Update bspline interpretation
        thread_per_block = 128
        block_per_thread = int(
            np.ceil(self._parent_ensemble.topology.num_particles / thread_per_block)
        )
        charge_density = cp.zeros(self._grid.shape, CUPY_FLOAT)
        self._bspline_interpretation[block_per_thread, thread_per_block](
            self._parent_ensemble.topology.device_charges,
            self._parent_ensemble.state.positions
            + self._parent_ensemble.state.device_half_pbc_diag,
            self._parent_ensemble.state.device_pbc_matrix,
            self._grid.device_shape,
            charge_density,
        )
        self._grid.add_field(
            "charge_density", charge_density / cp.prod(self._grid.device_grid_width)
        )

    def _get_current_field(self):
        cur_field = [
            getattr(self._grid.field, "%s_density" % i).copy()
            for i in self._ion_type_list
        ]
        return cur_field

    @staticmethod
    def _calculate_error(array1: cp.ndarray, array2: cp.ndarray):
        diff = array1 - array2
        denominator = 0.5 * (array1 + array2)
        denominator[denominator == 0] = 1e-9
        return float((cp.abs(diff / denominator)[20:-20, 20:-20, 20:-20]).max())

    def update(
        self, max_iterations=2500, error_tolerance=1e-2, check_freq=50, image_dir=False,
    ):
        self._check_bound_state()
        self._update_charge_density()
        self._grid.check_requirement()
        start_time = datetime.now().replace(microsecond=0)
        self._dump_log("Start at %s" % start_time)
        pre_factor, inv_denominator = self._generate_poisson_equation_coefficient()
        pre_factor_list = [
            self._generate_nernst_plank_equation_coefficient(i)
            for i in self._ion_type_list
        ]
        for iteration in range(max_iterations):
            self._solve_poisson_equation(pre_factor, inv_denominator)
            for i in range(self._num_ion_types):
                self._solve_nernst_plank_equation(
                    self._ion_type_list[i], pre_factor_list[i],
                )
            if self._is_img and iteration % 100 == 0:
                visualize_pnp_solution(
                    self._grid, os.path.join(image_dir, "iteration-%d.png" % iteration)
                )
            self._check_nernst_plank_equation_judgement()
            if iteration % check_freq == 0:
                if iteration == 0:
                    pre_list = self._get_current_field()
                    continue
                cur_list = self._get_current_field()
                error = [
                    self._calculate_error(i, j) for i, j in zip(cur_list, pre_list)
                ]
                if self._is_verbose:
                    log = "Iteration: %d; " % iteration
                    for i in range(self._num_ion_types):
                        log += "NPE %s: %.3e; " % (self._ion_type_list[i], error[i],)
                    log += (
                        "NPE with tolerance: %s"
                        % self._is_nernst_plank_equation_within_tolerance
                    )
                    self._dump_log(log)
                if np.mean(np.array(error)) <= error_tolerance:
                    break
                pre_list = cur_list
        end_time = datetime.now().replace(microsecond=0)
        if self._is_verbose:
            self._dump_log(
                "Finish at %d steps; Total time: %s"
                % (iteration, end_time - start_time)
            )

    @property
    def grid(self) -> Grid:
        return self._grid

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : fd_pnp_constraint.py
created time : 2022/07/17
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import numba as nb
import numba.cuda as cuda
import cupy as cp
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import Ensemble
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *
from grid import Grid
from cupy.cuda.nvtx import RangePush, RangePop


BSPLINE_ORDER = 4
NUM_GRIDS_PER_BLOCK = 8
DATA_SHAPE = NUM_GRIDS_PER_BLOCK + 2


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
                    "require_gradient": True,
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
                "require_gradient": True,
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

    def _solve_poisson_equation(
        self, max_iterations=500, error_tolerance=5e-6, soa_factor=0.01
    ):
        inv_square = (1 / self._grid.grid_width) ** 2
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
        # Relative permittivity
        relative_permittivity = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        target_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                # 0 for plus and 1 for minus
                target_slice[i] = slice(2, None) if j == 0 else slice(0, -2)
                relative_permittivity[i, j] = (0.5 * inv_square[i]) * (
                    self._grid.field.relative_permittivity[tuple(target_slice)]
                    + self._grid.field.relative_permittivity[1:-1, 1:-1, 1:-1]
                )
                denominator += relative_permittivity[i, j]
        denominator = 1 / denominator
        # Iteration
        phi = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        for iteration in range(max_iterations):
            # X Neumann condition
            phi[0, 0, :-1, :, :] = self._grid.field.electric_potential[2:-1, 1:-1, 1:-1]
            phi[0, 0, -1, :, :] = (
                self._grid.field.electric_potential[-2, 1:-1, 1:-1]
                + self._grid.field.electric_potential[-1, 1:-1, 1:-1]
                * self._grid.grid_width[0]
            )
            phi[0, 1, 1:, :, :] = self._grid.field.electric_potential[1:-2, 1:-1, 1:-1]
            phi[0, 1, 0, :, :] = (
                self._grid.field.electric_potential[1, 1:-1, 1:-1]
                - self._grid.field.electric_potential[0, 1:-1, 1:-1]
                * self._grid.grid_width[0]
            )
            # Y Neumann
            phi[1, 0, :, :-1, :] = self._grid.field.electric_potential[1:-1, 2:-1, 1:-1]
            phi[1, 0, :, -1, :] = (
                self._grid.field.electric_potential[1:-1, -2, 1:-1]
                + self._grid.field.electric_potential[1:-1, -1, 1:-1]
                * self._grid.grid_width[1]
            )
            phi[1, 1, :, 1:, :] = self._grid.field.electric_potential[1:-1, 1:-2, 1:-1]
            phi[1, 1, :, 0, :] = (
                self._grid.field.electric_potential[1:-1, 1, 1:-1]
                - self._grid.field.electric_potential[1:-1, 0, 1:-1]
                * self._grid.grid_width[1]
            )
            # Z Dirichlet
            phi[2, 0] = self._grid.field.electric_potential[1:-1, 1:-1, 2:]
            phi[2, 1] = self._grid.field.electric_potential[1:-1, 1:-1, :-2]
            nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            for i in range(self._grid.num_dimensions):
                for j in range(2):
                    nominator += relative_permittivity[i, j] * phi[i, j]
            nominator += charge_density
            new = (
                soa_factor * self._grid.field.electric_potential[1:-1, 1:-1, 1:-1]
                + (1 - soa_factor) * nominator * denominator
            )
            if iteration % 25 == 0:
                diff = cp.abs(
                    self._grid.field.electric_potential[1:-1, 1:-1, 1:-1] - new
                )
                index = np.unravel_index(cp.argmax(diff), self._grid.inner_shape)
                error = abs(
                    diff[index]
                    / self._grid.field.electric_potential[1:-1, 1:-1, 1:-1][index]
                )
                if error <= error_tolerance:
                    self._grid.field.electric_potential[1:-1, 1:-1, 1:-1] = new
                    break
            self._grid.field.electric_potential[1:-1, 1:-1, 1:-1] = new
        return (iteration, error)

    def _solve_nernst_plank_equation(
        self, ion_type: str, max_iterations=500, error_tolerance=5e-5, soa_factor=0.05
    ):
        # Read input
        ion_valence = self._ion_valence_list[self._ion_type_list.index(ion_type)]
        ion_density = getattr(self._grid.field, ion_type + "_density")
        diffusion_coefficient = getattr(
            self._grid.field, ion_type + "_diffusion_coefficient"
        )
        inv_square = (1 / self._grid.grid_width) ** 2
        channel_mask = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )  # True for no-boundary
        energy = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        diffusion = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        density = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        target_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        for i in range(self._grid.num_dimensions):
            for j in [0, 1]:
                # 0 for plus and 1 for minus
                target_slice[i] = slice(2, None) if j == 0 else slice(0, -2)
                channel_mask[i, j] = (
                    self._grid.field.channel_shape[tuple(target_slice)]
                    == self._grid.field.channel_shape[1:-1, 1:-1, 1:-1]
                )
                diffusion[i, j] = (0.5 * inv_square[i]) * (
                    diffusion_coefficient[tuple(target_slice)]
                    + diffusion_coefficient[1:-1, 1:-1, 1:-1]
                )
                target_slice[i] = slice(1, -1)
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
        energy = 1 - energy  # 1 - V
        denominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                denominator += channel_mask[i, j] * diffusion[i, j] * energy[i, j]
        threshold = 5e-4
        denominator[(denominator <= threshold) & (denominator >= 0)] = threshold
        denominator[(denominator >= -threshold) & (denominator < 0)] = -threshold
        denominator = 1 / denominator
        energy = 2 - energy  # 1 + V

        for i in range(self._grid.num_dimensions):
            for j in range(2):
                energy[i, j] *= diffusion[i, j]

        for iteration in range(max_iterations):
            # X Neumann condition
            density[0, 0, :-1, :, :] = ion_density[2:-1, 1:-1, 1:-1]
            density[0, 0, -1, :, :] = (
                ion_density[-2, 1:-1, 1:-1]
                + ion_density[-1, 1:-1, 1:-1] * self._grid.grid_width[0]
            )
            density[0, 1, 1:, :, :] = ion_density[1:-2, 1:-1, 1:-1]
            density[0, 1, 0, :, :] = (
                ion_density[1, 1:-1, 1:-1]
                - ion_density[0, 1:-1, 1:-1] * self._grid.grid_width[0]
            )
            # Y Neumann
            density[1, 0, :, :-1, :] = ion_density[1:-1, 2:-1, 1:-1]
            density[1, 0, :, -1, :] = (
                ion_density[1:-1, -2, 1:-1]
                + ion_density[1:-1, -1, 1:-1] * self._grid.grid_width[1]
            )
            density[1, 1, :, 1:, :] = ion_density[1:-1, 1:-2, 1:-1]
            density[1, 1, :, 0, :] = (
                ion_density[1:-1, 1, 1:-1]
                - ion_density[1:-1, 0, 1:-1] * self._grid.grid_width[1]
            )
            # Z Neumann
            density[2, 0, :, :, :-1] = ion_density[1:-1, 1:-1, 2:-1]
            density[2, 0, :, :, -1] = (
                ion_density[1:-1, 1:-1, -2]
                + ion_density[1:-1, 1:-1, -1] * self._grid.grid_width[2]
            )
            density[2, 1, :, :, 1:] = ion_density[1:-1, 1:-1, 1:-2]
            density[2, 1, :, :, 0] = (
                ion_density[1:-1, 1:-1, 1]
                - ion_density[1:-1, 1:-1, 0] * self._grid.grid_width[2]
            )
            nominator = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            for i in range(self._grid.num_dimensions):
                for j in range(2):
                    nominator += channel_mask[i, j] * energy[i, j] * density[i, j]
            new = (
                soa_factor * ion_density[1:-1, 1:-1, 1:-1]
                + (1 - soa_factor) * nominator * denominator
            )
            if iteration % 25 == 0:
                diff = cp.abs(ion_density[1:-1, 1:-1, 1:-1] - new)
                index = np.unravel_index(cp.argmax(diff), self._grid.inner_shape)
                error = abs(diff[index] / ion_density[1:-1, 1:-1, 1:-1][index])
                if error <= error_tolerance:
                    ion_density[1:-1, 1:-1, 1:-1] = new
                    break
            ion_density[1:-1, 1:-1, 1:-1] = new
        return (iteration, error)

    def update(self, max_iterations=500, is_verbose=True):
        self._check_bound_state()
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
        self._grid.check_requirement()
        for iteration in range(max_iterations):
            iteration_vec = np.zeros(self._num_ion_types + 1)
            error_vec = np.zeros(self._num_ion_types + 1)
            iteration_vec[0], error_vec[0] = self._solve_poisson_equation()
            for i in range(self._num_ion_types):
                (
                    iteration_vec[i + 1],
                    error_vec[i + 1],
                ) = self._solve_nernst_plank_equation(self._ion_type_list[i])
            if is_verbose:
                log = "Iteration: %d; " % iteration
                log += "PE: %d, %.3e; " % (iteration_vec[0], error_vec[0])
                for i in range(self._num_ion_types):
                    log += "NPE %s: %d, %.3e; " % (
                        self._ion_type_list[i],
                        iteration_vec[i + 1],
                        error_vec[i + 1],
                    )
                print(log)
            if np.all(iteration_vec == 0):
                if is_verbose:
                    print("Finish at %d steps" % iteration)
                break
            if iteration % 5 == 0:
                grid = 64
                fig, ax = plt.subplots(3, 1, figsize=[16, 27])
                fig.tight_layout()
                c = ax[0].contourf(
                    self.grid.x[1:-1, grid, 1:-1].get(),
                    self.grid.z[1:-1, grid, 1:-1].get(),
                    self.grid.field.electric_potential[1:-1, grid, 1:-1].get(),
                    100,
                )
                fig.colorbar(c, ax=ax[0])
                c = ax[1].contourf(
                    self.grid.x[1:-1, grid, 1:-1].get(),
                    self.grid.z[1:-1, grid, 1:-1].get(),
                    self.grid.field.sod_density[1:-1, grid, 1:-1].get(),
                    100,
                )
                fig.colorbar(c, ax=ax[1])
                c = ax[2].contourf(
                    self.grid.x[1:-1, grid, 1:-1].get(),
                    self.grid.z[1:-1, grid, 1:-1].get(),
                    self.grid.field.cla_density[1:-1, grid, 1:-1].get(),
                    100,
                )
                fig.colorbar(c, ax=ax[2])
                plt.savefig("pnp-%d.png" % iteration)
                plt.close()

    @property
    def grid(self) -> Grid:
        return self._grid

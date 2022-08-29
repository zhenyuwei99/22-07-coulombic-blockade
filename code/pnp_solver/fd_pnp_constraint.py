#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : fd_pnp_constraint.py
created time : 2022/07/17
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import math
import numpy as np
import numba as nb
import numba.cuda as cuda
import cupy as cp
import cupyx.scipy.sparse as sparse
import cupyx.scipy.sparse.linalg as sp_linalg
import matplotlib
import matplotlib.pyplot as plt
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import Ensemble, Grid
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *


BSPLINE_ORDER = 4
NP_DENSITY_THRESHOLD = (
    (Quantity(500, mol / decimeter ** 3) * NA)
    .convert_to(1 / default_length_unit ** 3)
    .value
)


class FDPoissonNernstPlanckConstraint(Constraint):
    def __init__(self, temperature: Quantity, grid: Grid, **ion_information) -> None:
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
        self._ion_type_list = list(ion_information.keys())
        self._ion_valence_list = list(ion_information.values())
        self._num_ion_types = len(self._ion_type_list)
        for key in ion_information.keys():
            self._grid.requirement[key + "_density"] = {
                "require_gradient": False,
                "require_curvature": False,
            }
            self._grid.requirement[key + "_diffusion_coefficient"] = {
                "require_gradient": False,
                "require_curvature": False,
            }
        # Kernel
        self._bspline_interpretation = cuda.jit(
            nb.void(
                NUMBA_FLOAT[:, ::1],  # charges
                NUMBA_FLOAT[:, ::1],  # positions
                NUMBA_FLOAT[:, ::1],  # pbc_matrix
                NUMBA_INT[::1],  # grid_shape
                NUMBA_FLOAT[:, :, ::1],  # charge_density
            )
        )(self._bspline_interpretation_kernel)
        # Attributes
        self._num_inner_grids = int(np.prod(self._grid.inner_shape))
        self._coefficient_shape = (self._num_inner_grids, self._num_inner_grids)
        self._grid_index = (
            cp.arange(self._num_inner_grids)
            .reshape(self._grid.inner_shape)
            .astype(CUPY_INT)
        )

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

    def _join_coefficient(self, data, row, col):
        # join
        data = cp.hstack(data)
        row = cp.hstack(row)
        col = cp.hstack(col)
        return sparse.coo_matrix(
            (data, (row, col)), shape=self._coefficient_shape, dtype=CUPY_FLOAT
        ).tocsr()

    def _generate_poisson_equation_coefficient(self):
        # Factor
        inv_square = (1 / self._grid.grid_width) ** 2
        neighbor_factor = cp.zeros(
            [self._grid.num_dimensions, 2] + self._grid.inner_shape, CUPY_FLOAT
        )
        self_factor = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
        target_slice = [slice(1, -1) for i in range(self._grid.num_dimensions)]
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                # 0 for plus and 1 for minus
                target_slice[i] = slice(2, None) if j == 0 else slice(0, -2)
                neighbor_factor[i, j] = (0.5 * inv_square[i]) * (
                    self._grid.field.relative_permittivity[tuple(target_slice)]
                    + self._grid.field.relative_permittivity[1:-1, 1:-1, 1:-1]
                )
                self_factor -= neighbor_factor[i, j]
        # Equation coefficient
        source_vector = -(
            self._grid.field.charge_density[1:-1, 1:-1, 1:-1].flatten() * self._k0
        )
        # Self (boundary point with Neumann boundary will include the neighbor)
        data = self_factor.copy()
        # X Neumann (change coefficient and source)
        # right
        data[-1, :, :] += neighbor_factor[0, 0, -1, :, :]
        source_vector[self._grid_index[-1, :, :].flatten()] += -(
            self._grid.field.electric_potential[-1, 1:-1, 1:-1]
            * self._grid.grid_width[0]
        ).flatten()
        # left
        data[0, :, :] += neighbor_factor[0, 1, 0, :, :]
        source_vector[self._grid_index[0, :, :].flatten()] += (
            self._grid.field.electric_potential[0, 1:-1, 1:-1]
            * self._grid.grid_width[0]
        ).flatten()
        # Y Neumann (change coefficient and source)
        # front
        data[:, -1, :] += neighbor_factor[1, 0, :, -1, :]
        source_vector[self._grid_index[:, -1, :].flatten()] += -(
            self._grid.field.electric_potential[1:-1, -1, 1:-1]
            * self._grid.grid_width[1]
        ).flatten()
        # back
        data[:, 0, :] = neighbor_factor[1, 1, :, 0, :]
        source_vector[self._grid_index[:, 0, :].flatten()] += (
            self._grid.field.electric_potential[1:-1, 0, 1:-1]
            * self._grid.grid_width[1]
        ).flatten()
        # Z Dirichlet (change source)
        # top
        source_vector[self._grid_index[:, :, -1].flatten()] += -(
            self._grid.field.electric_potential[1:-1, 1:-1, -1]
            * neighbor_factor[2, 0, :, :, -1]
        ).flatten()
        # bottom
        source_vector[self._grid_index[:, :, 0].flatten()] += -(
            self._grid.field.electric_potential[1:-1, 1:-1, 0]
            * neighbor_factor[2, 1, :, :, 0]
        ).flatten()
        coefficient = self._join_coefficient(
            [data.flatten()], [self._grid_index.flatten()], [self._grid_index.flatten()]
        )
        # Neighbor (neighbor of boundary points have been handled)
        target_slice = [slice(None, None) for i in range(self._grid.num_dimensions)]
        col_slice = [slice(None, None) for i in range(self._grid.num_dimensions)]
        data, row, col = [], [], []
        for i in range(self._grid.num_dimensions):
            for j in range(2):
                # 0 for plus and 1 for minus
                # Target slice stand for the self point and the index in the neighbor factor
                # when plus, only (0, -1) points are no boundary
                # when minus, only (1, 0) points are no boundary
                target_slice[i] = slice(0, -1) if j == 0 else slice(1, None)
                data.append(neighbor_factor[tuple([i, j] + target_slice)].flatten())
                row.append(self._grid_index[tuple(target_slice)].flatten())
                # col slice stand for the real index for the neighbor point
                col_slice[i] = slice(1, None) if j == 0 else slice(0, -1)
                col.append(self._grid_index[tuple(col_slice)].flatten())
                # Refresh
                target_slice[i] = slice(None, None)
                col_slice[i] = slice(None, None)
        coefficient += self._join_coefficient(data, row, col)
        return coefficient, source_vector

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

    def _generate_equation_array(self):
        coefficient, source_vector = self._generate_poisson_equation_coefficient()
        return coefficient, source_vector

    def update(self):
        self._check_bound_state()
        self._update_charge_density()
        self._grid.check_requirement()
        coefficient, source_vector = self._generate_equation_array()
        x, info = sp_linalg.minres(coefficient, source_vector, tol=1e-5)

        x = x.reshape(self._grid.inner_shape)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=[16, 9])
        convert = (
            Quantity(1, default_energy_unit / default_charge_unit)
            .convert_to(volt)
            .value
        )
        c = ax.contour(
            self._grid.coordinate.x[1:-1, 33, 1:-1].get(),
            self._grid.coordinate.z[1:-1, 33, 1:-1].get(),
            x[:, 64, :].get() * convert,
            200,
        )
        fig.colorbar(c)
        plt.savefig("test_pe.png")

    @property
    def grid(self) -> Grid:
        return self._grid


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

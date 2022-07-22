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
    def __init__(self, grid: Grid) -> None:
        super().__init__()
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
            }
        )
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

    def _solve_poisson_equation(self, max_iteration=500, error_tolerance=5e-9):
        inv_2x, inv_2y, inv_2z = 0.5 / self._grid.grid_width
        inv_x2, inv_y2, inv_z2 = (1 / self._grid.grid_width) ** 2
        inv_k0 = 1 / (4 * np.pi * EPSILON0.value)
        denominator = 2 * (inv_x2 + inv_y2 + inv_z2)
        for i in range(max_iteration):
            # X Neumann condition
            x_plus = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            x_plus[:-1, :, :] = self._grid.field.electric_potential[2:-1, 1:-1, 1:-1]
            x_plus[-1, :, :] = (
                self._grid.field.electric_potential[-2, 1:-1, 1:-1]
                + self._grid.field.electric_potential[-1, 1:-1, 1:-1]
                * self._grid.grid_width[0]
            )
            x_minus = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            x_minus[1:, :, :] = self._grid.field.electric_potential[1:-2, 1:-1, 1:-1]
            x_minus[0, :, :] = (
                self._grid.field.electric_potential[1, 1:-1, 1:-1]
                - self._grid.field.electric_potential[0, 1:-1, 1:-1]
                * self._grid.grid_width[0]
            )
            # Y Neumann
            y_plus = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            y_plus[:, :-1, :] += self._grid.field.electric_potential[1:-1, 2:-1, 1:-1]
            y_plus[:, -1, :] += (
                self._grid.field.electric_potential[1:-1, -2, 1:-1]
                + self._grid.field.electric_potential[1:-1, -1, 1:-1]
                * self._grid.grid_width[1]
            )
            y_minus = cp.zeros(self._grid.inner_shape, CUPY_FLOAT)
            y_minus[:, 1:, :] = self._grid.field.electric_potential[1:-1, 1:-2, 1:-1]
            y_minus[:, 0, :] = (
                self._grid.field.electric_potential[1:-1, 1, 1:-1]
                - self._grid.field.electric_potential[1:-1, 0, 1:-1]
                * self._grid.grid_width[1]
            )
            # Z Dirichlet
            z_plus = self._grid.field.electric_potential[1:-1, 1:-1, 2:]
            z_minus = self._grid.field.electric_potential[1:-1, 1:-1, :-2]
            # Iteration rule
            new = (
                (x_plus + x_minus) * inv_x2
                + (y_plus + y_minus) * inv_y2
                + (z_plus + z_minus) * inv_z2
            )
            new += (
                (x_plus - x_minus)
                * self._grid.gradient.relative_permittivity[0, :, :, :]
                * inv_2x
                + (y_plus - y_minus)
                * self._grid.gradient.relative_permittivity[1, :, :, :]
                * inv_2y
                + (z_plus - z_minus)
                * self._grid.gradient.relative_permittivity[2, :, :, :]
                * inv_2z
            ) / self._grid.field.relative_permittivity[1:-1, 1:-1, 1:-1]
            new += (
                self._grid.field.charge_density[1:-1, 1:-1, 1:-1]
                / self._grid.field.relative_permittivity[1:-1, 1:-1, 1:-1]
                * inv_k0
            )
            new *= 1 / denominator
            print(
                cp.max(
                    cp.abs(self._grid.field.electric_potential[1:-1, 1:-1, 1:-1] - new)
                )
            )
            self._grid.field.electric_potential[1:-1, 1:-1, 1:-1] = (
                0.01 * self._grid.field.electric_potential[1:-1, 1:-1, 1:-1]
                + 0.99 * new
            )

    @staticmethod
    def _solve_nernst_plank_equation_kernel(
        grid_width,
        diffusion_coefficient,
        exclusion_potential_energy,
        electric_potential_energy,
        concentration_boundary,
        concentration,
    ):
        pass

    def update(self, max_iterations=1000, error_tolerance=5e-9):
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
        self._solve_poisson_equation()

    @property
    def grid(self) -> Grid:
        return self._grid


if __name__ == "__main__":
    import os
    import mdpy as md
    import numpy as np
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    str_dir = os.path.join(cur_dir, "str")
    pdb = md.io.PDBParser(os.path.join(str_dir, "sio2_pore.pdb"))
    psf = md.io.PSFParser(os.path.join(str_dir, "sio2_pore.psf"))
    topology = psf.topology
    # for particle in topology.particles:
    #     particle._charge = 0
    # topology.join()
    pbc = pdb.pbc_matrix
    pbc[2, 2] *= 4
    ensemble = md.core.Ensemble(topology, pbc, is_use_tile_list=False)
    ensemble.state.set_positions(pdb.positions)
    # Solver
    grid = Grid(
        x=[-pbc[0, 0] / 2, pbc[0, 0] / 2, 128],
        y=[-pbc[1, 1] / 2, pbc[1, 1] / 2, 128],
        z=[-pbc[2, 2] / 2, pbc[2, 2] / 2, 256],
    )
    r = cp.sqrt(grid.x**2 + grid.y**2)
    alpha = 2
    relative_permittivity = (
        78
        / (
            (1 + cp.exp(-alpha * (r - 10)))
            * (1 + cp.exp(alpha * (cp.abs(grid.z) - 20)))
        )
        + 2
    )
    # print(relative_permittivity[(r > 10) & (grid.z < 20) & (grid.z > -20)])
    # relative_permittivity[(r > 9) & (grid.z < 18) & (grid.z > -18)] = 2
    grid.add_field("relative_permittivity", relative_permittivity)
    electric_potential = grid.zeros_field()
    electric_potential[:, :, 0] = 0.0002
    electric_potential[:, :, -1] = 0
    grid.add_field("electric_potential", electric_potential)
    constraint = FDPoissonNernstPlanckConstraint(grid)

    ensemble.add_constraints(constraint)
    ensemble.update_constraints()

    grid = 64
    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    # c = ax.contourf(
    #     constraint.grid.x[1:-1, grid, 1:-1].get(),
    #     constraint.grid.z[1:-1, grid, 1:-1].get(),
    #     constraint.grid.gradient.relative_permittivity[2, :, grid, :].get(),
    #     100,
    # )
    c = ax.contourf(
        constraint.grid.x[1:-1, grid, 1:-1].get(),
        constraint.grid.z[1:-1, grid, 1:-1].get(),
        constraint.grid.field.electric_potential[1:-1, grid, 1:-1].get(),
        100,
    )
    plt.colorbar(c)
    plt.show()

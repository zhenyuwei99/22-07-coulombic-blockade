#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_vdw_fft.py
created time : 2022/11/20
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import math
import mdpy as md
import cupy as cp
import numpy as np
import numba.cuda as cuda
import scipy.signal as signal
from mdpy import SPATIAL_DIM
from mdpy.core import Topology, Grid
from mdpy.utils import select, check_topological_selection_condition
from mdpy.environment import *
from test_vdw import VDW_DICT

BSPLINE_ORDER = 4


class ParticleDensityAnalyser:
    def __init__(
        self, topology: Topology, positions: np.ndarray, pbc_matrix: np.ndarray
    ) -> None:
        self._topology = topology
        self._positions = cp.array(positions, CUPY_FLOAT)
        self._pbc_matrix = cp.array(pbc_matrix, CUPY_FLOAT)
        # Kernel
        self._bspline = cuda.jit(
            nb.void(
                NUMBA_FLOAT[:, ::1],  # positions
                NUMBA_FLOAT[::1],  # grid_range
                NUMBA_INT[::1],  # grid_shape
                NUMBA_FLOAT[:, :, ::1],  # density
            )
        )(self._bspline_kernel)

    @staticmethod
    def _bspline_kernel(positions, grid_range, grid_shape, density):
        particle_id = cuda.grid(1)
        thread_x = cuda.threadIdx.x
        if particle_id >= positions.shape[0]:
            return None
        # Shared array
        shared_grid_range = cuda.shared.array((SPATIAL_DIM), NUMBA_FLOAT)
        shared_grid_shape = cuda.shared.array((SPATIAL_DIM), NUMBA_INT)
        if thread_x == 0:
            for i in range(SPATIAL_DIM):
                shared_grid_range[i] = grid_range[i]
        elif thread_x == 1:
            for i in range(SPATIAL_DIM):
                shared_grid_shape[i] = grid_shape[i]
        cuda.syncthreads()
        # Grid information
        for i in range(SPATIAL_DIM):
            if positions[particle_id, i] < 0:
                return None
            if positions[particle_id, i] > shared_grid_range[i]:
                return None
        grid_index = cuda.local.array((SPATIAL_DIM), NUMBA_INT)
        grid_fraction = cuda.local.array((SPATIAL_DIM), NUMBA_FLOAT)
        local_target_grid = cuda.local.array((SPATIAL_DIM, BSPLINE_ORDER), NUMBA_INT)
        for i in range(SPATIAL_DIM):
            index = (
                positions[particle_id, i] * shared_grid_shape[i] / shared_grid_range[i]
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
        for i in range(BSPLINE_ORDER):
            grid_x = local_target_grid[0, i]
            x = local_spline_coefficient[0, i]
            for j in range(BSPLINE_ORDER):
                grid_y = local_target_grid[1, j]
                xy = x * local_spline_coefficient[1, j]
                for k in range(BSPLINE_ORDER):
                    grid_z = local_target_grid[2, k]
                    xyz = xy * local_spline_coefficient[2, k]
                    cuda.atomic.add(density, (grid_x, grid_y, grid_z), xyz)

    def analysis(
        self, grid: Grid, selection_condition: list[dict], extend_radius: float
    ) -> cp.ndarray:
        check_topological_selection_condition(selection_condition)
        matrix_id = select(self._topology, selection_condition)
        coordinate_range = grid.coordinate_range.copy()
        coordinate_range[:, 0] -= extend_radius
        coordinate_range[:, 1] += extend_radius
        import time

        s = time.time()
        positions = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    positions.append(
                        self._positions[matrix_id, :]
                        + cp.dot(self._pbc_matrix, cp.array([i, j, k], CUPY_FLOAT))
                    )
        positions = cp.vstack(positions).astype(CUPY_FLOAT)
        positive_positions = positions - cp.array(coordinate_range[:, 0], CUPY_FLOAT)
        x, y, z = cp.meshgrid(
            cp.arange(
                coordinate_range[0, 0],
                coordinate_range[0, 1] + grid.grid_width,
                grid.grid_width,
                CUPY_FLOAT,
            ),
            cp.arange(
                coordinate_range[1, 0],
                coordinate_range[1, 1] + grid.grid_width,
                grid.grid_width,
                CUPY_FLOAT,
            ),
            cp.arange(
                coordinate_range[2, 0],
                coordinate_range[2, 1] + grid.grid_width,
                grid.grid_width,
                CUPY_FLOAT,
            ),
            indexing="ij",
        )
        density = cp.zeros_like(x, CUPY_FLOAT)
        grid_range = cp.array(
            coordinate_range[:, 1] - coordinate_range[:, 0], CUPY_FLOAT
        )
        grid_shape = cp.array(x.shape, CUPY_INT)
        thread_per_block = 64
        block_per_thread = int(np.ceil(positions.shape[0] / thread_per_block))
        self._bspline[block_per_thread, thread_per_block](
            positive_positions, grid_range, grid_shape, density
        )
        density.get()
        e = time.time()
        print("Run xxx for %s s" % (e - s))
        return x, y, z, density


def get_vdw_kernel(grid: Grid, r_cut, sigma, epsilon):
    grid_width = grid.grid_width
    x, y, z = np.meshgrid(
        np.arange(-r_cut, r_cut + grid_width, grid_width),
        np.arange(-r_cut, r_cut + grid_width, grid_width),
        np.arange(-r_cut, r_cut + grid_width, grid_width),
        indexing="ij",
    )
    r = np.sqrt(x**2 + y**2 + z**2)
    scaled_r = (sigma / (r + 0.01)) ** 6
    vdw = 4 * epsilon * (scaled_r**2 - scaled_r)
    vdw[vdw >= 5] = 5
    return vdw


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Path info
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, "data")
    pdb_file_path = os.path.join(data_dir, "str.pdb")
    psf_file_path = os.path.join(data_dir, "str.psf")
    # Read pdb
    pdb = md.io.PDBParser(pdb_file_path)
    positions = pdb.positions
    pbc_matrix = pdb.pbc_matrix.copy()
    pbc_diag = np.diagonal(pbc_matrix).copy()
    # Read psf
    topology = md.io.PSFParser(psf_file_path).topology
    # Grid
    r_cut = 8
    grid = Grid(
        grid_width=0.5,
        x=[-pbc_diag[0] / 2, pbc_diag[0] / 2],
        y=[-pbc_diag[1] / 2, pbc_diag[1] / 2],
        z=[-pbc_diag[2] / 2, pbc_diag[2] / 2],
    )
    analyser = ParticleDensityAnalyser(
        topology=topology, positions=positions, pbc_matrix=pbc_matrix
    )
    x, y, z, density = analyser.analysis(
        grid=grid,
        selection_condition=[{"particle type": [["CA"]]}],
        extend_radius=r_cut,
    )
    vdw = get_vdw_kernel(
        grid,
        r_cut,
        sigma=0.5 * (VDW_DICT["c"]["sigma"].value + VDW_DICT["cl"]["sigma"].value),
        epsilon=np.sqrt(
            VDW_DICT["c"]["epsilon"].value * VDW_DICT["cl"]["epsilon"].value
        ),
    )
    print(vdw.min())
    energy = signal.fftconvolve(density.get(), vdw, mode="valid")
    energy[energy >= 5] = 5

    print(vdw.min(), energy.min(), energy.min() / vdw.min())

    # Visualize
    fig, ax = plt.subplots(1, 1)
    x = grid.coordinate.x
    z = grid.coordinate.z
    if not True:
        target_slice = (
            slice(None, None),
            x.shape[1] // 2,
            slice(None, None),
        )
        c = ax.contourf(
            x[target_slice].get(), z[target_slice].get(), energy[target_slice], 200
        )
        fig.colorbar(c)
    else:
        target_slice = (
            slice(None, None),
            x.shape[1] // 2,
            x.shape[2] // 2,
        )
        ax.plot(x[target_slice].get(), energy[target_slice])
    plt.show()

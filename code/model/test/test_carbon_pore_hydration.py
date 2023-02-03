#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_carbon_pore_hydration.py
created time : 2022/11/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import mdpy as md
import numpy as np
import scipy.signal as signal
from mdpy.core import Grid
from mdpy.environment import *
from test_vdw import VDW_DICT
from test_vdw_fft import ParticleDensityAnalyser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydration import *
from utils import *


def get_hydration_kernel(grid: Grid, r_cut, file_path: str):
    grid_width = grid.grid_width
    x, y, z = np.meshgrid(
        np.arange(-r_cut, r_cut + grid_width, grid_width),
        np.arange(-r_cut, r_cut + grid_width, grid_width),
        np.arange(-r_cut, r_cut + grid_width, grid_width),
        indexing="ij",
    )
    r = np.sqrt(x**2 + y**2 + z**2)
    g = HydrationDistributionFunction(json_file_path=file_path)
    return g(r)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Path info
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, "data")
    json_dir = os.path.join(cur_dir, "../out")
    json_file_path = os.path.join(json_dir, "hydrogen-cla.json")
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
    r_cut = 14
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
    energy = get_hydration_kernel(grid, r_cut, file_path=json_file_path)
    energy = signal.fftconvolve(density.get(), energy, mode="valid")

    # Visualize
    fig, ax = plt.subplots(1, 1)
    x = grid.coordinate.x
    z = grid.coordinate.z
    if True:
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

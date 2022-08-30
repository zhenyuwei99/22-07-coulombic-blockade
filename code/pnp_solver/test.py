#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test.py
created time : 2022/08/29
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import shutil
import mdpy as md
from fd_pnp_constraint import FDPoissonNernstPlanckConstraint
from object.utils import *


def mkdir(dir_path):
    try:
        os.mkdir(dir_path)
    except:
        mkdir(os.path.dirname(dir_path))
        mkdir(dir_path)


def check_dir(dir_path: str, restart=False):
    if not os.path.exists(dir_path):
        mkdir(dir_path)
    if restart:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    return dir_path


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    str_dir = os.path.join(cur_dir, "str")
    out_dir = check_dir(os.path.join(cur_dir, "out/test/minus-1.00-V"), restart=True)
    # Read data
    pdb = md.io.PDBParser(os.path.join(str_dir, "sio2_pore.pdb"))
    psf = md.io.PSFParser(os.path.join(str_dir, "sio2_pore.psf"))
    grid_matrix = pdb.pbc_matrix
    grid_matrix[2, 2] += 40 * 2
    topology = psf.topology
    positions = pdb.positions
    # Grid
    r0 = Quantity(20, angstrom)
    l = Quantity(pdb.pbc_matrix[2, 2], angstrom)
    grid = md.core.Grid(
        x=[-grid_matrix[0, 0] / 2, grid_matrix[0, 0] / 2, 128],
        y=[-grid_matrix[1, 1] / 2, grid_matrix[1, 1] / 2, 128],
        z=[-grid_matrix[2, 2] / 2, grid_matrix[2, 2] / 2, 256],
    )
    grid.add_field(
        "channel_shape",
        generate_channel_shape(grid=grid, r0=r0, l=l, lb=Quantity(1, angstrom)),
    )
    grid.add_field(
        "relative_permittivity",
        generate_relative_permittivity_field(
            grid=grid, r0=r0, l=l, lb=Quantity(1, angstrom), ls=Quantity(1, angstrom),
        ),
    )
    grid.add_field(
        "electric_potential",
        generate_electric_potential_field(grid=grid, voltage=Quantity(-0.8, volt)),
    )
    # Pot
    pot_diffusion = Quantity(1.96 * 1e-9, meter ** 2 / second)
    grid.add_field(
        "pot_diffusion_coefficient",
        generate_diffusion_field(
            grid=grid, r0=r0, l=l, ls=30, diffusion=pot_diffusion, boundary_ratio=0.02,
        ),
    )
    grid.add_field(
        "pot_density",
        generate_density_field(
            grid=grid, density=Quantity(1.0, mol / decimeter ** 3) * NA
        ),
    )
    # Cla
    cla_diffusion = Quantity(2.032 * 1e-9, meter ** 2 / second)
    grid.add_field(
        "cla_diffusion_coefficient",
        generate_diffusion_field(
            grid=grid, r0=r0, l=l, ls=30, diffusion=cla_diffusion, boundary_ratio=0.02,
        ),
    )
    grid.add_field(
        "cla_density",
        generate_density_field(
            grid=grid, density=Quantity(1.0, mol / decimeter ** 3) * NA
        ),
    )
    # Constraint
    constraint = FDPoissonNernstPlanckConstraint(
        Quantity(300, kelvin), grid, pot=1, cla=-1
    )
    constraint.set_img_dir(out_dir)
    constraint.set_log_file(os.path.join(out_dir, "solver.log"))
    ensemble = md.core.Ensemble(topology, grid_matrix, is_use_tile_list=False)
    ensemble.state.set_positions(positions)
    ensemble.add_constraints(constraint)
    constraint.update()

    writer = md.io.GridWriter(os.path.join(out_dir, "res.grid"))
    writer.write(constraint.grid)

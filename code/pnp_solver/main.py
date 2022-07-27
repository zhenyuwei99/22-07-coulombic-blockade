#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : main.py
created time : 2022/07/27
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""


import os
import mdpy as md
import matplotlib.pyplot as plt
import cupy as cp
from mdpy.unit import *
from grid import Grid
from fd_pnp_constraint import FDPoissonNernstPlanckConstraint

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    str_dir = os.path.join(cur_dir, "str")
    os.system("rm -rf %s/*.png" % cur_dir)
    # Hyper parameter
    voltage = (
        Quantity(0.1, volt).convert_to(default_energy_unit / default_charge_unit).value
    )
    sod_diffusion_coefficient = (
        Quantity(1.334 * 1e-9, meter ** 2 / second)
        .convert_to(default_length_unit ** 2 / default_time_unit)
        .value
    )
    sod_density = (
        (Quantity(1.0, mol / decimeter ** 3) * NA)
        .convert_to(1 / default_length_unit ** 3)
        .value
    )
    cla_diffusion_coefficient = (
        Quantity(2.032 * 1e-9, meter ** 2 / second)
        .convert_to(default_length_unit ** 2 / default_time_unit)
        .value
    )
    cla_density = sod_density
    # Read structure
    pdb = md.io.PDBParser(os.path.join(str_dir, "sio2_pore.pdb"))
    psf = md.io.PSFParser(os.path.join(str_dir, "sio2_pore.psf"))
    topology = psf.topology
    # for particle in topology.particles:
    #   particle._charge = 0
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
    # Set constraint
    constraint = FDPoissonNernstPlanckConstraint(
        Quantity(300, kelvin), grid, sod=1, cla=-1
    )
    r = cp.sqrt(grid.x ** 2 + grid.y ** 2)
    alpha = 2
    nanopore_shape = 1 / (
        (1 + cp.exp(-alpha * (r - 9.5))) * (1 + cp.exp(alpha * (cp.abs(grid.z) - 20)))
    )  # 1 for pore 0 for solvation
    channel_shape = nanopore_shape >= 0.5
    constraint.grid.add_field("channel_shape", channel_shape)
    # relative_permittivity
    relative_permittivity = (1 - nanopore_shape) * 78 + 2
    constraint.grid.add_field("relative_permittivity", relative_permittivity)
    # electric_potential
    electric_potential = constraint.grid.zeros_field()
    electric_potential[:, :, 0] = voltage
    electric_potential[:, :, -1] = 0
    constraint.grid.add_field("electric_potential", electric_potential)
    # sod
    sod_diffusion_coefficient = (1 - nanopore_shape) * sod_diffusion_coefficient
    constraint.grid.add_field("sod_diffusion_coefficient", sod_diffusion_coefficient)
    sod_density = (1 - nanopore_shape) * sod_density
    sod_density[[0, -1], :, :] = 0
    sod_density[:, [0, -1], :] = 0
    sod_density[:, :, [0, -1]] = 0
    constraint.grid.add_field("sod_density", sod_density)
    # cla
    cla_diffusion_coefficient = (1 - nanopore_shape) * cla_diffusion_coefficient
    constraint.grid.add_field("cla_diffusion_coefficient", cla_diffusion_coefficient)
    cla_density = (1 - nanopore_shape) * cla_density
    cla_density[[0, -1], :, :] = 0
    cla_density[:, [0, -1], :] = 0
    cla_density[:, :, [0, -1]] = 0
    constraint.grid.add_field("cla_density", cla_density)

    ensemble.add_constraints(constraint)
    ensemble.update_constraints()

    grid = 64
    fig, ax = plt.subplots(1, 1, figsize=[16, 9])
    fig.tight_layout()
    # c = ax.contourf(
    #     constraint.grid.x[1:-1, grid, 1:-1].get(),
    #     constraint.grid.z[1:-1, grid, 1:-1].get(),
    #     constraint.grid.field.exclusion_potential_energy[1:-1, grid, 1:-1].get(),
    #     100,
    # )
    c = ax.contourf(
        constraint.grid.x[1:-1, grid, 1:-1].get(),
        constraint.grid.z[1:-1, grid, 1:-1].get(),
        constraint.grid.field.electric_potential[1:-1, grid, 1:-1].get(),
        100,
    )
    # c = ax.contourf(
    #     constraint.grid.x[1:-1, 1:-1, grid].get(),
    #     constraint.grid.y[1:-1, 1:-1, grid].get(),
    #     constraint.grid.field.electric_potential[1:-1, 1:-1, grid].get(),
    #     100,
    # )
    plt.colorbar(c)
    plt.savefig("res.png")

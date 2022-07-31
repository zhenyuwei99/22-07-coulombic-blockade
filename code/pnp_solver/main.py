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
from mdpy.core import Grid
from mdpy.io import GridWriter
from mdpy.unit import *
from mdpy.utils import *
from mdpy.environment import *
from fd_pnp_constraint import FDPoissonNernstPlanckConstraint

generate_job_name = (
    lambda voltage: "pnp-minus-%.3f-volt.grid" % abs(voltage)
    if voltage <= 0
    else "pnp-plus-%.3f-volt.grid" % abs(voltage)
)


def job(voltage, sod_density, cla_density, radius):
    voltage = check_quantity_value(voltage, default_energy_unit / default_charge_unit)
    sod_density = check_quantity_value(sod_density, 1 / default_length_unit ** 3)
    cla_density = check_quantity_value(cla_density, 1 / default_length_unit ** 3)
    sod_diffusion_coefficient = (
        Quantity(1.334 * 1e-9, meter ** 2 / second)
        .convert_to(default_length_unit ** 2 / default_time_unit)
        .value
    )
    cla_diffusion_coefficient = (
        Quantity(2.032 * 1e-9, meter ** 2 / second)
        .convert_to(default_length_unit ** 2 / default_time_unit)
        .value
    )
    extend_radius = radius * 1.1
    shrink_radius = radius * 0.9
    # job_name
    job_name = generate_job_name(
        Quantity(voltage, default_energy_unit / default_charge_unit)
        .convert_to(volt)
        .value
    )
    file_name = os.path.join(out_dir, job_name)
    if not os.path.exists(file_name):
        print("Start %s" % job_name)
        # Read structure
        pdb = md.io.PDBParser(os.path.join(str_dir, "sio2_pore.pdb"))
        psf = md.io.PSFParser(os.path.join(str_dir, "sio2_pore.psf"))
        topology = psf.topology
        positions = pdb.positions
        if False:
            topology.split()
            topology.add_particles(
                [
                    md.core.Particle(charge=1, molecule_type="FIX"),
                    md.core.Particle(charge=1, molecule_type="FIX"),
                ]
            )
            positions = np.zeros([topology.num_particles, 3], NUMPY_FLOAT)
            positions[: pdb.positions.shape[0], :] = pdb.positions
            positions[-2, :] = [-radius, 0, 0]
            positions[-1, :] = [radius, 0, 0]
        pbc = pdb.pbc_matrix
        pbc[2, 2] *= 8
        ensemble = md.core.Ensemble(topology, pbc, is_use_tile_list=False)
        ensemble.state.set_positions(positions)
        # Solver
        grid = Grid(
            x=[-pbc[0, 0] / 2, pbc[0, 0] / 2, 128],
            y=[-pbc[1, 1] / 2, pbc[1, 1] / 2, 128],
            z=[-pbc[2, 2] / 2, pbc[2, 2] / 2, 256],
        )
        constraint = FDPoissonNernstPlanckConstraint(
            Quantity(300, kelvin), grid, sod=1, cla=-1
        )
        r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
        alpha = 2
        nanopore_shape = 1 / (
            (1 + cp.exp(-alpha * (r - shrink_radius)))
            * (1 + cp.exp(alpha * (cp.abs(grid.coordinate.z) - 18)))
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
        constraint.grid.add_field(
            "sod_diffusion_coefficient", sod_diffusion_coefficient
        )
        tmp = sod_density
        sod_density = (1 - nanopore_shape) * tmp
        sod_density[[0, -1], :, :] = 0
        sod_density[:, [0, -1], :] = 0
        sod_density[:, :, [0, -1]] = tmp
        constraint.grid.add_field("sod_density", sod_density)
        # cla
        cla_diffusion_coefficient = (1 - nanopore_shape) * cla_diffusion_coefficient
        constraint.grid.add_field(
            "cla_diffusion_coefficient", cla_diffusion_coefficient
        )
        tmp = cla_density
        cla_density = (1 - nanopore_shape) * cla_density
        cla_density[[0, -1], :, :] = 0
        cla_density[:, [0, -1], :] = 0
        cla_density[:, :, [0, -1]] = tmp
        constraint.grid.add_field("cla_density", cla_density)

        ensemble.add_constraints(constraint)
        ensemble.update_constraints()

        writer = GridWriter(file_name)
        writer.write(constraint.grid)
    else:
        print("Job %s exists, skipping current job" % job_name)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    str_dir = os.path.join(cur_dir, "str")
    out_dir = os.path.join(cur_dir, "out/15nm-pore-no-fixed")
    os.system("rm -rf %s/*.png" % cur_dir)
    # Hyper parameter
    radius = 15
    for v in np.linspace(-5, 5, 40, endpoint=True):
        voltage = (
            Quantity(v, volt)
            .convert_to(default_energy_unit / default_charge_unit)
            .value
        )
        sod_density = Quantity(0.5, mol / decimeter ** 3) * NA
        cla_density = sod_density
        job(
            voltage=voltage,
            sod_density=sod_density,
            cla_density=cla_density,
            radius=radius,
        )

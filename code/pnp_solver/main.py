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
import sys
import shutil
import mdpy as md
import cupy as cp
import multiprocessing as mp
from time import sleep
from datetime import datetime
from mdpy.core import Grid
from mdpy.io import GridWriter
from mdpy.unit import *
from mdpy.utils import *
from mdpy.environment import *
from fd_pnp_constraint import FDPoissonNernstPlanckConstraint
from manager import *

generate_job_name = (
    lambda voltage: "pnp-minus-%.2f-volt" % abs(voltage)
    if voltage <= 0
    else "pnp-plus-%.2f-volt" % abs(voltage)
)


def check_dir(dir_path: str, restart=False):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if restart:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    return dir_path


def job(
    device_file_path: str,
    voltage: Quantity,
    sod_density: Quantity,
    cla_density: Quantity,
    radius: float,
    root_dir: str,
    str_dir: str,
):
    job_name = generate_job_name(check_quantity_value(voltage, volt))
    out_dir = check_dir(os.path.join(root_dir, job_name))
    log_file = os.path.join(out_dir, "log.txt")
    grid_file = os.path.join(out_dir, "res.grid")
    if not os.path.exists(grid_file):
        device, job = get_available_device(device_file_path)
        with md.device.Device(device):
            sys.stdout = open(log_file, "w")
            register_device(device_file_path, device, job)
            print(
                "Submit %s to device-%d-job-%d at %s"
                % (job_name, device, job, datetime.now().replace(microsecond=0))
            )
            voltage = check_quantity_value(
                voltage, default_energy_unit / default_charge_unit
            )
            sod_density = check_quantity_value(
                sod_density, 1 / default_length_unit ** 3
            )
            cla_density = check_quantity_value(
                cla_density, 1 / default_length_unit ** 3
            )
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
            sod_density = grid.zeros_field()
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
            cla_density = grid.zeros_field()
            cla_density[[0, -1], :, :] = 0
            cla_density[:, [0, -1], :] = 0
            cla_density[:, :, [0, -1]] = tmp
            constraint.grid.add_field("cla_density", cla_density)

            ensemble.add_constraints(constraint)
            ensemble.update_constraints()

            writer = GridWriter(grid_file)
            writer.write(constraint.grid)
        free_device(device_file_path, device, job)
        print(
            "Finish %s to device-%d-job-%d at %s"
            % (job_name, device, job, datetime.now().replace(microsecond=0))
        )
    else:
        print("Job %s exists, skipping current job" % job_name)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    str_dir = os.path.join(cur_dir, "str")
    root_dir = os.path.join(cur_dir, "out/15nm-pore-no-fixed")
    os.system("rm -rf %s/*.png" % cur_dir)
    # Hyper parameter
    radius = 15
    sod_density = Quantity(0.5, mol / decimeter ** 3) * NA
    cla_density = sod_density
    # Manager
    num_devices = 4
    num_jobs_per_device = 2
    num_total_jobs = num_devices * num_jobs_per_device
    device_file_path = init_device_file(
        file_path=os.path.join(cur_dir, "device.h5"),
        num_devices=num_devices,
        num_jobs_per_device=num_jobs_per_device,
    )
    target_voltage = np.linspace(-2, 2, 200, endpoint=True)
    interval = target_voltage.shape[0] // num_total_jobs
    pool = mp.Pool(num_total_jobs)
    for i in range(interval):
        for voltage in target_voltage[i::interval]:
            voltage = Quantity(voltage, volt)
            pool.apply_async(
                job,
                args=(
                    device_file_path,
                    voltage,
                    sod_density,
                    cla_density,
                    radius,
                    root_dir,
                    str_dir,
                ),
            )
            sleep(0.5)
    pool.close()
    pool.join()

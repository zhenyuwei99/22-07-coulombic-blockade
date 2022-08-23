#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : ls.py
created time : 2022/08/23
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import traceback
import multiprocessing as mp
from time import sleep
from main import check_dir
from utils import *
from fd_pnp_constraint import *
from manager import *


def generate_job_name(ls_pot: Quantity, ls_cla: Quantity, voltage: Quantity):
    job_name = "pot-%.2fA-cla-%.2fA" % (
        check_quantity_value(ls_pot, angstrom),
        check_quantity_value(ls_cla, angstrom),
    )
    voltage = check_quantity_value(voltage, volt)
    job_name += "-minus-" if voltage < 0 else "-plus-"
    job_name += "%.2fV" % abs(voltage)
    return job_name


def job(
    device_file_path: str,
    ls_pot: Quantity,
    ls_cla: Quantity,
    voltage: Quantity,
    str_dir: str,
    root_dir: str,
):
    try:
        job_name = generate_job_name(ls_pot=ls_pot, ls_cla=ls_cla, voltage=voltage)
        out_dir = check_dir(os.path.join(root_dir, job_name))
        log_file = os.path.join(out_dir, "log.txt")
        grid_file = os.path.join(out_dir, "res.grid")
        if not os.path.exists(grid_file):
            device, job = get_available_device(device_file_path)
            with open(log_file, "w") as f:
                print("Submit %s to device-%d-job-%d" % (job_name, device, job), file=f)
            with md.device.Device(device):
                register_device(device_file_path, device, job)
                # Create ensemble
                pdb = md.io.PDBParser(os.path.join(str_dir, "sio2_pore.pdb"))
                psf = md.io.PSFParser(os.path.join(str_dir, "sio2_pore.psf"))
                grid_matrix = pdb.pbc_matrix
                grid_matrix[2, 2] += 40 * 2
                topology = psf.topology
                positions = pdb.positions
                ensemble = md.core.Ensemble(
                    topology, grid_matrix, is_use_tile_list=False
                )
                ensemble.state.set_positions(positions)
                # Generate grid
                r0 = Quantity(20, angstrom)
                l = Quantity(pdb.pbc_matrix[2, 2], angstrom)
                grid = md.core.Grid(
                    x=[-grid_matrix[0, 0] / 2, grid_matrix[0, 0] / 2, 128],
                    y=[-grid_matrix[1, 1] / 2, grid_matrix[1, 1] / 2, 128],
                    z=[-grid_matrix[2, 2] / 2, grid_matrix[2, 2] / 2, 256],
                )
                grid.add_field(
                    "channel_shape",
                    generate_channel_shape(
                        grid=grid, r0=r0, l=l, lb=Quantity(1, angstrom)
                    ),
                )
                grid.add_field(
                    "relative_permittivity",
                    generate_relative_permittivity_field(
                        grid=grid,
                        r0=r0,
                        l=l,
                        lb=Quantity(1, angstrom),
                        ls=Quantity(1, angstrom),
                    ),
                )
                grid.add_field(
                    "electric_potential",
                    generate_electric_potential_field(grid=grid, voltage=voltage),
                )
                # Pot
                pot_diffusion = Quantity(1.96 * 1e-9, meter ** 2 / second)
                grid.add_field(
                    "pot_diffusion_coefficient",
                    generate_diffusion_field(
                        grid=grid,
                        r0=r0,
                        l=l,
                        ls=ls_pot,
                        diffusion=pot_diffusion,
                        boundary_ratio=0.1,
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
                        grid=grid,
                        r0=r0,
                        l=l,
                        ls=ls_cla,
                        diffusion=cla_diffusion,
                        boundary_ratio=0.1,
                    ),
                )
                grid.add_field(
                    "cla_density",
                    generate_density_field(
                        grid=grid, density=Quantity(1.0, mol / decimeter ** 3) * NA
                    ),
                )
                # Create constraint
                constraint = FDPoissonNernstPlanckConstraint(
                    Quantity(300, kelvin), grid, pot=1, cla=-1
                )
                constraint.set_log_file(log_file, "a")
                constraint.set_img_dir(out_dir)
                ensemble.add_constraints(constraint)
                constraint.update(
                    max_iterations=1500, error_tolerance=1e-3, image_dir=out_dir
                )

                writer = md.io.GridWriter(grid_file)
                writer.write(constraint.grid)
            free_device(device_file_path, device, job)
        else:
            print("Job %s exists, skipping current job" % job_name)
    except:
        error = traceback.format_exc()
        raise Exception(error)


if __name__ == "__main__":
    # Read input
    root_dir = sys.argv[1]
    ls_pot = float(sys.argv[2])
    ls_cla = float(sys.argv[3])

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    str_dir = os.path.join(cur_dir, "../str")

    voltage_list = np.linspace(-1, 1, 12, endpoint=True)  # Manager
    num_devices = 3
    num_jobs_per_device = 2
    num_total_jobs = num_devices * num_jobs_per_device
    device_file_path = init_device_file(
        file_path=os.path.join(cur_dir, "device.h5"),
        num_devices=num_devices,
        num_jobs_per_device=num_jobs_per_device,
    )
    pool = mp.Pool(num_total_jobs)
    for voltage in voltage_list:
        ls_pot = Quantity(ls_pot, angstrom)
        ls_cla = Quantity(ls_cla, angstrom)
        voltage = Quantity(voltage, volt)
        pool.apply_async(
            job,
            args=(device_file_path, ls_pot, ls_cla, voltage, str_dir, root_dir),
            error_callback=print,
        )
        sleep(0.5)
    pool.close()
    pool.join()
    error = calculate_error(root_dir)
    print("Error: ", error)


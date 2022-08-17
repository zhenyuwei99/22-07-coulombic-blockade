#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : parametrization.py
created time : 2022/08/15
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import mdpy as md
import cupy as cp
import multiprocessing as mp
from time import sleep
from itertools import product
from mdpy.utils import check_quantity_value
from mdpy.unit import *
from mdpy.environment import *
from main import check_dir
from analyzer import analysis
from fd_pnp_constraint import *
from sigmoid import *
from manager import *


def generate_channel_shape(grid: md.core.Grid, r0: Quantity, l: Quantity, lb: Quantity):
    r0 = check_quantity_value(r0, default_length_unit)
    l = check_quantity_value(l, default_length_unit)
    lb = check_quantity_value(lb, default_length_unit)
    r0 = r0 - lb
    l = l / 2 + lb
    r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
    channel_shape = CUPY_FLOAT(1) / (
        (1 + cp.exp(-(r - r0))) * (1 + cp.exp((cp.abs(grid.coordinate.z) - l)))
    )  # 1 for pore 0 for solvation
    channel_shape = channel_shape >= 0.5
    return channel_shape.astype(cp.bool8)


def generate_relative_permittivity_field(
    grid: md.core.Grid,
    r0: Quantity,
    l: Quantity,
    lb: Quantity,
    ls: Quantity,
    cavity_permittivity=2,
    solution_permittivity=80,
):
    r0 = check_quantity_value(r0, default_length_unit)
    l = check_quantity_value(l, default_length_unit)
    lb = check_quantity_value(lb, default_length_unit)
    ls = check_quantity_value(ls, default_length_unit)
    r0 = r0 - lb
    l = l / 2 + lb
    alpha = reasoning_alpha(ls)
    r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
    channel_shape = CUPY_FLOAT(1) / (
        (1 + cp.exp(-alpha * (r - r0)))
        * (1 + cp.exp(alpha * (cp.abs(grid.coordinate.z) - l)))
    )  # 1 for pore 0 for solvation
    relative_permittivity = (1 - channel_shape) * (
        solution_permittivity - cavity_permittivity
    ) + cavity_permittivity
    return relative_permittivity.astype(CUPY_FLOAT)


def generate_diffusion_field(
    grid: md.core.Grid, r0: Quantity, l: Quantity, ls: Quantity, diffusion: Quantity, A
):
    # Channel shape
    r0 = check_quantity_value(r0, default_length_unit)
    l = check_quantity_value(l, default_length_unit) / 2
    ls = check_quantity_value(ls, default_length_unit)
    alpha = reasoning_alpha(ls)
    r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
    channel_shape = CUPY_FLOAT(1) / (
        (1 + cp.exp(-alpha * (r - r0)))
        * (1 + cp.exp(alpha * (cp.abs(grid.coordinate.z) - l)))
    )  # 1 for pore 0 for solvation

    diffusion = check_quantity_value(
        diffusion, default_length_unit ** 2 / default_time_unit
    )
    factor = 0.5 + A
    return ((factor - channel_shape) * diffusion / factor).astype(CUPY_FLOAT)


def generate_density_field(grid, density: Quantity):
    density = check_quantity_value(density, 1 / default_length_unit ** 3)
    density_field = grid.zeros_field()
    density_field[:, :, [0, -1]] = density
    return density_field.astype(CUPY_FLOAT)


def generate_electric_potential_field(grid, voltage: Quantity):
    voltage = check_quantity_value(voltage, default_energy_unit / default_charge_unit)
    electric_potential_field = grid.zeros_field()
    electric_potential_field[:, :, 0] = voltage
    return electric_potential_field.astype(CUPY_FLOAT)


def visulize_field(grid: md.core.Grid, field: cp.ndarray):
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    field_shape = list(field.shape)
    fig, ax = plt.subplots(1, 1, figsize=[12, 8])
    index = field_shape[1] // 2
    c = ax.contour(
        grid.coordinate.x[:, index, :].get(),
        grid.coordinate.z[:, index, :].get(),
        field[:, index, :].get(),
        100,
    )
    plt.colorbar(c)
    plt.savefig(os.path.join(cur_dir, "field.png"))


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
            ensemble = md.core.Ensemble(topology, grid_matrix, is_use_tile_list=False)
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
                generate_channel_shape(grid=grid, r0=r0, l=l, lb=Quantity(1, angstrom)),
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
                    grid=grid, r0=r0, l=l, ls=ls_pot, diffusion=pot_diffusion, A=0.1,
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
                    grid=grid, r0=r0, l=l, ls=ls_cla, diffusion=cla_diffusion, A=0.1,
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


VOLTAGE_LIST = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
REFERENCE_CURRENT = np.array(
    [
        -5.08e-9,
        -4.14e-9,
        -2.97e-9,
        -2.29e-9,
        -1.28e-9,
        -4.39e-11,
        9.8e-10,
        1.7e-9,
        2.5e-9,
        2.9e-9,
        3.7e-9,
    ]
)


def calculate_error(root_dir, ls_pot, ls_cla):
    pot_res, cla_res = [], []
    for voltage in VOLTAGE_LIST:
        job_name = generate_job_name(ls_pot, ls_cla, voltage)
        index = 128
        target_file = os.path.join(root_dir, job_name, "res.grid")
        grid = md.io.GridParser(target_file).grid
        pot_res.append(analysis(grid, "pot", 1, index))
        cla_res.append(analysis(grid, "cla", -1, index))
    pot_res = (
        Quantity(pot_res, default_charge_unit / default_time_unit)
        .convert_to(ampere)
        .value
    )
    cla_res = (
        Quantity(cla_res, default_charge_unit / default_time_unit)
        .convert_to(ampere)
        .value
    )
    error = np.abs((pot_res + cla_res - REFERENCE_CURRENT) / REFERENCE_CURRENT).mean()
    print(job_name, error)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    str_dir = os.path.join(cur_dir, "str")
    root_dir = check_dir(os.path.join(cur_dir, "out/parametrization"))
    ls_pot_list = [15, 20, 25, 30]
    ls_cla_list = [15, 20, 25, 30]
    if True:
        grid_matrix = np.eye(3)
        grid_matrix[0] = 100
        grid_matrix[1] = 100
        grid_matrix[2] = 350
        grid = md.core.Grid(
            x=[-grid_matrix[0, 0] / 2, grid_matrix[0, 0] / 2, 128],
            y=[-grid_matrix[1, 1] / 2, grid_matrix[1, 1] / 2, 128],
            z=[-grid_matrix[2, 2] / 2, grid_matrix[2, 2] / 2, 256],
        )
        diffusion = generate_diffusion_field(
            grid=grid, r0=20, l=250, ls=5, diffusion=1, A=0.1
        )
        visulize_field(grid, diffusion)
    if not True:
        ls_pot_list = [15, 20]
        ls_cla_list = [15, 20, 25, 30]
        for ls_pot, ls_cla in product(ls_pot_list, ls_cla_list):
            calculate_error(root_dir, ls_pot, ls_cla)
    if not True:
        voltage_list = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        job_list = list(product(ls_pot_list, ls_cla_list, voltage_list))
        print("%d jobs in total" % len(job_list))
        # Manager
        num_devices = 3
        num_jobs_per_device = 2
        num_total_jobs = num_devices * num_jobs_per_device
        device_file_path = init_device_file(
            file_path=os.path.join(cur_dir, "device.h5"),
            num_devices=num_devices,
            num_jobs_per_device=num_jobs_per_device,
        )
        pool = mp.Pool(num_total_jobs)
        for ls_pot, ls_cla, voltage in job_list:
            ls_pot = Quantity(ls_pot, angstrom)
            ls_cla = Quantity(ls_cla, angstrom)
            voltage = Quantity(voltage, volt)
            # job(device_file_path, ls_pot, ls_cla, voltage, str_dir, root_dir)
            pool.apply_async(
                job,
                args=(device_file_path, ls_pot, ls_cla, voltage, str_dir, root_dir),
                error_callback=print,
            )
            sleep(0.5)
        pool.close()
        pool.join()

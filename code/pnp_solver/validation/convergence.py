#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : convergence.py
created time : 2022/08/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import mdpy as md
import cupy as cp
import multiprocessing as mp
from mdpy.core import Grid
from mdpy.io import GridWriter
from mdpy.unit import *
from mdpy.utils import *
from mdpy.environment import *
from fd_pnp_constraint import FDPoissonNernstPlanckConstraint

MODE = ["all_zero", "all_bulk", "boltzmann"]
NUM_MODES = len(MODE)


def check_dir(dir_path: str, restart=False):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if restart:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    return dir_path


def job(
    mode: str,
    voltage: Quantity,
    sod_density: Quantity,
    cla_density: Quantity,
    radius: float,
    out_dir: str,
    str_dir: str,
    device_index: int,
):
    with md.device.Device(device_index):
        # sys.stdout = open(os.path.join(out_dir, "log.txt"), "w")
        voltage = check_quantity_value(
            voltage, default_energy_unit / default_charge_unit
        )
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
        file_name = os.path.join(out_dir, mode + ".grid")
        # Read structure
        pdb = md.io.PDBParser(os.path.join(str_dir, "sio2_pore.pdb"))
        psf = md.io.PSFParser(os.path.join(str_dir, "sio2_pore.psf"))
        topology = psf.topology
        positions = pdb.positions
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
        ensemble.add_constraints(constraint)
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
        sod_density_array = grid.zeros_field()
        sod_density_array[[0, -1], :, :] = 0
        sod_density_array[:, [0, -1], :] = 0
        sod_density_array[:, :, [0, -1]] = sod_density
        constraint.grid.add_field("sod_density", sod_density_array)
        # cla
        cla_diffusion_coefficient = (1 - nanopore_shape) * cla_diffusion_coefficient
        constraint.grid.add_field(
            "cla_diffusion_coefficient", cla_diffusion_coefficient
        )
        cla_density_array = grid.zeros_field()
        cla_density_array[[0, -1], :, :] = 0
        cla_density_array[:, [0, -1], :] = 0
        cla_density_array[:, :, [0, -1]] = sod_density
        constraint.grid.add_field("cla_density", cla_density_array)
        # Solve poisson equation
        constraint._update_charge_density()
        constraint._solve_poisson_equation()
        # Set density
        if mode == "all_zero":
            sod_density_array = cp.zeros(grid.inner_shape, CUPY_FLOAT)
            cla_density_array = cp.zeros(grid.inner_shape, CUPY_FLOAT)
        elif mode == "all_bulk":
            sod_density_array = cp.zeros(grid.inner_shape, CUPY_FLOAT) + sod_density
            cla_density_array = cp.zeros(grid.inner_shape, CUPY_FLOAT) + cla_density
        elif mode == "boltzmann":
            beta = (
                1 / (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
            )
            sod_density_array = cp.exp(
                -beta * grid.field.electric_potential[1:-1, 1:-1, 1:-1]
            )
            sod_density_array /= sod_density_array.mean()
            sod_density_array *= sod_density

            cla_density_array = cp.exp(
                beta * grid.field.electric_potential[1:-1, 1:-1, 1:-1]
            )
            cla_density_array /= cla_density_array.mean()
            cla_density_array *= cla_density

        grid.field.sod_density[1:-1, 1:-1, 1:-1] = sod_density_array
        grid.field.cla_density[1:-1, 1:-1, 1:-1] = cla_density_array
        constraint.update(image_dir=out_dir)

        writer = GridWriter(file_name)
        writer.write(constraint.grid)


def calculate_error(array1: cp.ndarray, array2: cp.ndarray):
    diff = cp.abs(array1 - array2)
    denominator = 0.5 * (array1 + array2)
    denominator[denominator == 0] = 1e-9
    return (diff / cp.abs(denominator)).max()


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    str_dir = os.path.join(cur_dir, "../str")
    radius = 15
    voltage = (
        Quantity(1, volt).convert_to(default_energy_unit / default_charge_unit).value
    )
    sod_density = Quantity(0.5, mol / decimeter ** 3) * NA
    cla_density = sod_density
    if False:
        pool = mp.Pool(NUM_MODES)
        for index, mode in enumerate(MODE):
            out_dir = check_dir(os.path.join(cur_dir, "out", mode))
            pool.apply_async(
                job,
                args=(
                    mode,
                    voltage,
                    sod_density,
                    cla_density,
                    radius,
                    out_dir,
                    str_dir,
                    index,
                ),
            )
        pool.close()
        pool.join()
    electric_potential_list = []
    sod_density_list = []
    cla_density_list = []
    for mode in MODE:
        out_dir = check_dir(os.path.join(cur_dir, "out", mode))
        grid = md.io.GridParser(os.path.join(out_dir, mode + ".grid")).grid
        electric_potential_list.append(grid.field.electric_potential)
        sod_density_list.append(grid.field.sod_density)
        cla_density_list.append(grid.field.cla_density)
    for index1, mode1 in enumerate(MODE):
        for index2, mode2 in enumerate(MODE[index1 + 1 :]):
            index2 += index1 + 1
            log = "%s-%s:\n" % (mode1, mode2)
            log += "electric_potential %.3e;\t" % calculate_error(
                electric_potential_list[index1], electric_potential_list[index2]
            )
            log += "sod_density %.3e;\t" % calculate_error(
                sod_density_list[index1], sod_density_list[index2]
            )
            log += "cla_density %.3e;\t" % calculate_error(
                cla_density_list[index1], cla_density_list[index2]
            )
            print(log)


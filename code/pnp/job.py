#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : job.py
created time : 2022/08/31
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import json
import time
import traceback
import mdpy as md
import cupy as cp
import numpy as np
import multiprocessing as mp
from mdpy.unit import *
from mdpy.utils import *
from mdpy.environment import *
from fd_pnp_constraint import FDPoissonNernstPlanckConstraint
from manager import *
from utils import *

CUR_DIR = cur_dir = os.path.dirname(os.path.abspath(__file__))
STR_DIR = os.path.join(CUR_DIR, "str")
TCL_TEMPLATE_NAME = "template.tcl"
STR_NAME = "r0-%.4fA-l0-%.4fA-w0-%.4fA"
Z_PADDING_LENGTH = Quantity(80, angstrom).convert_to(default_length_unit).value
CAVITY_PERMITTIVITY = 2
SOLUTION_PERMITTIVITY = 80
SIO2_LATTICE = [4.978, 4.978, 6.848]
CRYST1 = "CRYST1" + "%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f\n"
ION_DICT = {
    "sod": {
        "valence": 1,
        "diffusion": Quantity(1.33 * 1e-9, meter ** 2 / second),
        "ls": Quantity(25, angstrom),
        "boundary_ratio": 0.1,
    },
    "pot": {
        "valence": 1,
        "diffusion": Quantity(1.96 * 1e-9, meter ** 2 / second),
        "ls": Quantity(25, angstrom),
        "boundary_ratio": 0.1,
    },
    "cal": {
        "valence": 2,
        "diffusion": Quantity(0.79 * 1e-9, meter ** 2 / second),
        "ls": Quantity(25, angstrom),
        "boundary_ratio": 0.1,
    },
    "lan": {
        "valence": 3,
        "diffusion": Quantity(0.62 * 1e-9, meter ** 2 / second),
        "ls": Quantity(25, angstrom),
        "boundary_ratio": 0.1,
    },
    "cla": {
        "valence": -1,
        "diffusion": Quantity(2.032 * 1e-9, meter ** 2 / second),
        "ls": Quantity(25, angstrom),
        "boundary_ratio": 0.1,
    },
}

"""
Argument for PNP solver job:
# Equation parameters
- voltage (compulsory)
- [ion_name]_density (compulsory)

# Structure parameters
- r0 (compulsory)
- l0 (compulsory)
- w0 (compulsory)
- str_dir (compulsory)

# Hyper Parameter
- lb
- relative_permittivity_ls
- [ion_name]_ls
- [ion_name]_boundary_ratio
"""


def job(device_file_path, json_file_path):
    # Execution
    try:
        # Parse path
        root_dir = os.path.dirname(json_file_path)
        log_file_path = os.path.join(root_dir, "log.txt")
        grid_file_path = os.path.join(root_dir, "res.grid")
        job_name = os.path.basename(root_dir)
        job_dict = read_json(json_file_path)
        if os.path.exists(grid_file_path):
            print("Job %s exists, skipping current job" % job_name)
            return None
        device, job = get_available_device(device_file_path)
        with open(log_file_path, "w") as f:
            print(
                "Submit %s to device-%d-job-%d" % (job_name, device, job), file=f,
            )
        with md.device.Device(device):
            register_device(device_file_path, device, job)
            # Read data
            str_dir = job_dict["str_dir"]
            str_name = job_dict["str_name"]
            pdb_file_path = job_dict["pdb_file_path"]
            psf_file_path = job_dict["psf_file_path"]
            voltage = job_dict["voltage"]
            r0 = job_dict["r0"]
            l0 = job_dict["l0"]
            w0 = job_dict["w0"]
            lb = job_dict["lb"]
            relative_permittivity_ls = job_dict["relative_permittivity_ls"]
            ion_types = job_dict["ion_types"]
            ion_density = []
            ion_ls = []
            ion_boundary_ratio = []
            for ion_type in ion_types:
                ion_density.append(job_dict["%s_density" % ion_type])
                ion_ls.append(job_dict["%s_ls" % ion_type])
                ion_boundary_ratio.append(job_dict["%s_boundary_ratio" % ion_type])
            num_ion_types = job_dict["num_ion_types"]
            # Create gird
            grid = md.core.Grid(x=job_dict["x"], y=job_dict["y"], z=job_dict["z"])
            grid.add_field("channel_shape", generate_channel_shape(grid, r0, l0, lb))
            grid.add_field(
                "relative_permittivity",
                generate_relative_permittivity_field(
                    grid, r0, l0, lb, relative_permittivity_ls
                ),
            )
            grid.add_field(
                "electric_potential", generate_electric_potential_field(grid, voltage),
            )
            for ion_type in ion_types:
                grid.add_field(
                    "%s_density" % ion_type,
                    generate_density_field(grid, ion_type, ion_types, ion_density),
                )
                grid.add_field(
                    "%s_diffusion_coefficient" % ion_type,
                    generate_diffusion_field(
                        grid, r0, l0, ion_type, ion_types, ion_ls, ion_boundary_ratio,
                    ),
                )
            # Create ensemble
            pdb = md.io.PDBParser(pdb_file_path)
            psf = md.io.PSFParser(psf_file_path)
            topology = psf.topology
            positions = pdb.positions
            ensemble = md.core.Ensemble(
                topology, pdb.pbc_matrix, is_use_tile_list=False
            )
            ensemble.state.set_positions(positions)
            # Create constraint
            ion_dict = {}
            for ion_type in ion_types:
                ion_dict[ion_type] = ION_DICT[ion_type]["valence"]
            constraint = FDPoissonNernstPlanckConstraint(
                Quantity(300, kelvin), grid, **ion_dict
            )
            constraint.set_log_file(log_file_path, "a")
            constraint.set_img_dir(root_dir)
            ensemble.add_constraints(constraint)
            constraint.update(max_iterations=5000, error_tolerance=1e-2)
            # Output
            writer = md.io.GridWriter(grid_file_path)
            writer.write(constraint.grid)
        free_device(device_file_path, device, job)
        post("Finish %s" % job_name)
    except:
        error = traceback.format_exc()
        return error


def generate_channel_shape(grid, r0, l0, lb):
    r0 = r0 - lb
    l0 = l0 / 2 + lb
    r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
    channel_shape = CUPY_FLOAT(1) / (
        (1 + cp.exp(-(r - r0))) * (1 + cp.exp((cp.abs(grid.coordinate.z) - l0)))
    )  # 1 for pore 0 for solvation
    channel_shape = channel_shape >= 0.5
    return channel_shape.astype(cp.bool8)


def generate_relative_permittivity_field(grid, r0, l0, lb, relative_permittivity_ls):
    r0 = r0 - lb
    l0 = l0 / 2 + lb
    alpha = reasoning_alpha(relative_permittivity_ls)
    r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
    channel_shape = CUPY_FLOAT(1) / (
        (1 + cp.exp(-alpha * (r - r0)))
        * (1 + cp.exp(alpha * (cp.abs(grid.coordinate.z) - l0)))
    )  # 1 for pore 0 for solvation
    relative_permittivity = (1 - channel_shape) * (
        SOLUTION_PERMITTIVITY - CAVITY_PERMITTIVITY
    ) + CAVITY_PERMITTIVITY
    return relative_permittivity.astype(CUPY_FLOAT)


def generate_diffusion_field(
    grid, r0, l0, ion_type, ion_types, ion_ls, ion_boundary_ratio
):
    ion_index = ion_types.index(ion_type)
    alpha = reasoning_alpha(ion_ls[ion_index])
    r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
    channel_shape = CUPY_FLOAT(1) / (
        (1 + cp.exp(-alpha * (r - r0)))
        * (1 + cp.exp(alpha * (cp.abs(grid.coordinate.z) - l0 / 2)))
    )  # 1 for pore 0 for solvation
    diffusion = (
        ION_DICT[ion_type]["diffusion"]
        .convert_to(default_length_unit ** 2 / default_time_unit)
        .value
    )
    factor = 0.5 + ion_boundary_ratio[ion_index]
    return ((factor - channel_shape) * diffusion / factor).astype(CUPY_FLOAT)


def generate_density_field(grid, ion_type, ion_types, ion_density):
    ion_index = ion_types.index(ion_type)
    density_field = grid.zeros_field()
    density_field[:, :, [0, -1]] = ion_density[ion_index]
    return density_field.astype(CUPY_FLOAT)


def generate_electric_potential_field(grid, voltage):
    electric_potential_field = grid.zeros_field()
    electric_potential_field[:, :, 0] = voltage
    return electric_potential_field.astype(CUPY_FLOAT)


def check_structure(str_dir, r0, l0, w0):
    # Calculate reasonable parameter
    r0 = check_quantity_value(r0, default_length_unit)
    l0 = check_quantity_value(l0, default_length_unit)
    w0 = check_quantity_value(w0, default_length_unit)
    grid_x = np.round(w0 / SIO2_LATTICE[0])
    grid_y = np.round(w0 / SIO2_LATTICE[1])
    grid_z = np.round(l0 / SIO2_LATTICE[2])
    # Update parameter
    w0 = grid_x * SIO2_LATTICE[0]
    l0 = grid_z * SIO2_LATTICE[2]
    z_length = l0 + Z_PADDING_LENGTH
    # Generate file
    str_name = STR_NAME % (r0, l0, w0)
    pdb_file_path = os.path.join(str_dir, str_name + ".pdb")
    psf_file_path = os.path.join(str_dir, str_name + ".psf")
    tcl_file_path = os.path.join(str_dir, str_name + ".tcl")
    if os.path.exists(pdb_file_path) and os.path.exists(psf_file_path):
        return r0, l0, w0, z_length
    with open(os.path.join(str_dir, TCL_TEMPLATE_NAME), "r") as f:
        tcl_command = f.read() % (str_name, grid_x, grid_y, grid_z, r0)
    with open(tcl_file_path, "w") as f:
        print(tcl_command, file=f)
    os.system("cd %s && vmd -dispdev text -e %s" % (str_dir, tcl_file_path))
    # Set pbc
    with open(pdb_file_path, "r") as f:
        pdb_data = f.readlines()
        pdb_data[0] = CRYST1 % (w0, w0, z_length, 90, 90, 90)
    with open(pdb_file_path, "w") as f:
        f.writelines(pdb_data)
    os.remove(tcl_file_path)
    return r0, l0, w0, z_length


def generate_json(
    json_file_path: str,
    voltage: Quantity,
    r0: Quantity,
    l0: Quantity,
    w0: Quantity,
    grid_width: Quantity,
    lb: Quantity = Quantity(1, angstrom),
    relative_permittivity_ls: Quantity = Quantity(1, angstrom),
    str_dir: str = STR_DIR,
    **ion_data,
) -> None:
    job_dict = {}
    # Data
    job_dict["str_dir"] = str_dir
    r0, l0, w0, z_length = check_structure(
        str_dir=job_dict["str_dir"], r0=r0, l0=l0, w0=w0,
    )
    job_dict["r0"] = float(r0)
    job_dict["l0"] = float(l0)
    job_dict["w0"] = float(w0)
    job_dict["z_length"] = float(z_length)
    job_dict["lb"] = float(check_quantity_value(lb, default_length_unit))
    job_dict["voltage"] = float(
        check_quantity_value(voltage, default_energy_unit / default_charge_unit)
    )
    job_dict["relative_permittivity_ls"] = float(
        check_quantity_value(relative_permittivity_ls, default_length_unit)
    )
    # Ion
    job_dict["ion_types"] = [i.split("_")[0] for i in ion_data.keys() if "density" in i]
    job_dict["num_ion_types"] = len(job_dict["ion_types"])
    for ion_type in job_dict["ion_types"]:
        if not ion_type in ION_DICT.keys():
            raise KeyError(
                "%s is not supported. Supported list: %s"
                % (ion_type, list(ION_DICT.keys()))
            )
        # Density (compulsory)
        density_key = "%s_density" % ion_type
        job_dict[density_key] = float(
            (
                check_quantity(
                    ion_data[density_key], default_mol_unit / default_length_unit ** 3
                )
                * NA
            ).value
        )
        # ls (optional)
        ls_key = "%s_ls" % ion_type
        ls = ion_data[ls_key] if ls_key in ion_data.keys() else ION_DICT[ion_type]["ls"]
        job_dict[ls_key] = float(check_quantity_value(ls, default_length_unit))
        # boundary_ratio (optional)
        boundary_ratio_key = "%s_boundary_ratio" % ion_type
        job_dict[boundary_ratio_key] = float(
            ion_data[boundary_ratio_key]
            if boundary_ratio_key in ion_data.keys()
            else ION_DICT[ion_type]["boundary_ratio"]
        )
    # Gird: +1 stands for including the endpoint
    grid_width = check_quantity_value(grid_width, default_length_unit)
    job_dict["x"] = [
        float(-job_dict["w0"] / 2),
        float(job_dict["w0"] / 2),
        int(np.ceil(job_dict["w0"] / grid_width) + 1),
    ]
    job_dict["y"] = [
        float(-job_dict["w0"] / 2),
        float(job_dict["w0"] / 2),
        int(np.ceil(job_dict["w0"] / grid_width) + 1),
    ]
    job_dict["z"] = [
        float(-job_dict["z_length"] / 2),
        float(job_dict["z_length"] / 2),
        int(np.ceil(job_dict["z_length"] / grid_width) + 1),
    ]
    # Structure attribute
    job_dict["str_name"] = STR_NAME % (job_dict["r0"], job_dict["l0"], job_dict["w0"])
    job_dict["pdb_file_path"] = os.path.join(str_dir, job_dict["str_name"] + ".pdb")
    job_dict["psf_file_path"] = os.path.join(str_dir, job_dict["str_name"] + ".psf")
    # Output
    with open(json_file_path, "w") as f:
        data = json.dumps(job_dict, sort_keys=True, indent=2)
        data = data.encode("utf-8").decode("unicode_escape")
        print(data, file=f)
    return json_file_path


def read_json(json_file_path: str) -> dict:
    with open(json_file_path, "r") as f:
        job_dict = json.load(f)
    return job_dict


if __name__ == "__main__":
    device_file_path = sys.argv[1]
    num_devices = int(sys.argv[2])
    num_jobs_per_device = int(sys.argv[3])
    pool = mp.Pool(num_devices * num_jobs_per_device)
    for json_file_path in sys.argv[4:]:
        print(json_file_path)
        pool.apply_async(job, args=(device_file_path, json_file_path), callback=print)
        time.sleep(0.5)
    pool.close()
    pool.join()

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : generator.py
created time : 2022/09/21
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import json
import datetime
import traceback
from mdpy.unit import *
from mdpy.utils import *
from str.generator import generate_structure
from simulator import Simulator, SIMULATION_NAME_LIST
from utils import *

CUR_DIR = cur_dir = os.path.dirname(os.path.abspath(__file__))
STR_DIR = os.path.join(CUR_DIR, "str")
FORCEFIELD_DIR = os.path.join(CUR_DIR, "forcefield")
PARAMETER_FILE_PATHS = [
    os.path.join(FORCEFIELD_DIR, "carbon.par"),
    os.path.join(FORCEFIELD_DIR, "water.par"),
]


def modify_simulation_unit(simulation_recipe: dict):
    keys = simulation_recipe.keys()
    if "temperature" in keys:
        simulation_recipe["temperature"] = float(
            check_quantity_value(simulation_recipe["temperature"], kelvin)
        )
    if "pressure" in keys:
        simulation_recipe["pressure"] = float(simulation_recipe["pressure"])
    if "step_size" in keys:
        simulation_recipe["step_size"] = float(
            check_quantity_value(simulation_recipe["step_size"], femtosecond)
        )
    if "electric_field" in keys:
        simulation_recipe["electric_field"] = float(
            check_quantity_value(simulation_recipe["electric_field"], volt / nanometer)
        )
    return simulation_recipe


def generate_json(
    json_file_path: str,
    r0: Quantity,
    l0: Quantity,
    w0: Quantity,
    ls: Quantity,
    ions: dict,
    wall_charges: list,
    simulation_recipes: list,
) -> str:
    out_dir = check_dir(os.path.dirname(json_file_path))
    # Check unit
    r0 = float(check_quantity_value(r0, angstrom))
    l0 = float(check_quantity_value(l0, angstrom))
    w0 = float(check_quantity_value(w0, angstrom))
    ls = float(check_quantity_value(ls, angstrom))
    for key, value in ions.items():
        ions[key] = float(check_quantity_value(value, mol / decimeter**3))
    charges = {"num_sites": int(len(wall_charges))}
    for i, j in enumerate(wall_charges):
        charges["site-%d" % i] = {
            "z0": float(check_quantity_value(j["z0"], angstrom)),
            "q": float(check_quantity_value(j["q"], elementary_charge)),
            "n": int(j["n"]),
        }
    new_simulation_recipes = {}
    for index, simulation_recipe in enumerate(simulation_recipes):
        simulation_recipe["order"] = index
        simulation_name = simulation_recipe["name"]
        if not simulation_name in SIMULATION_NAME_LIST:
            raise KeyError("Unsupported simulation %s" % simulation_name)
        simulation_recipe = modify_simulation_unit(simulation_recipe)
        new_simulation_recipes[str(index)] = simulation_recipe
    new_simulation_recipes["num_simulations"] = int(len(simulation_recipes))
    job_dict = {
        "unit": {
            "length": "angstrom",
            "time": "femtosecond",
            "temperature": "kelvin",
            "pressure": "atm",
            "charge": "e",
            "concentration": "mol/L",
            "electric_field_intensity": "volt/nm",
        },
        "str": {
            "r0": r0,
            "l0": l0,
            "w0": w0,
            "ls": ls,
            "ions": ions,
            "wall_charges": charges,
        },
        "simulation": new_simulation_recipes,
    }
    # Generate file
    str_dict = job_dict["str"]
    if True:
        # Suspend in distribute mode
        structure_name, pdb_file_path, psf_file_path = generate_structure(
            r0=Quantity(str_dict["r0"], angstrom),
            l0=Quantity(str_dict["l0"], angstrom),
            w0=Quantity(str_dict["w0"], angstrom),
            ls=Quantity(str_dict["ls"], angstrom),
            ions=ions.copy(),
            wall_charges=wall_charges.copy(),
        )
    # Output
    with open(json_file_path, "w") as f:
        data = json.dumps(job_dict, sort_keys=True, indent=4)
        data = data.encode("utf-8").decode("unicode_escape")
        print(data, file=f)
    return json_file_path


def get_structure(job_dict: dict):
    str_dict = job_dict["str"]
    ions = str_dict["ions"]
    for key, value in ions.items():
        ions[key] = Quantity(value, mol / decimeter**3)
    wall_charges = []
    for key, value in str_dict["wall_charges"].items():
        if not "num_sites" in key:
            wall_charges.append(
                {
                    "z0": Quantity(value["z0"], angstrom),
                    "q": Quantity(value["q"], elementary_charge),
                    "n": value["n"],
                }
            )
    structure_name, pdb_file_path, psf_file_path = generate_structure(
        r0=Quantity(str_dict["r0"], angstrom),
        l0=Quantity(str_dict["l0"], angstrom),
        w0=Quantity(str_dict["w0"], angstrom),
        ls=Quantity(str_dict["ls"], angstrom),
        ions=ions,
        wall_charges=wall_charges,
    )
    return structure_name, pdb_file_path, psf_file_path


def execute_json(json_file_path: str, cuda_index: int = 0):
    try:
        root_dir = os.path.dirname(json_file_path)
        with open(json_file_path, "r") as f:
            job_dict = json.load(f)
        structure_name, pdb_file_path, psf_file_path = get_structure(job_dict)
        # Simulation
        sim_dict = job_dict["simulation"]
        num_simulations = sim_dict["num_simulations"]
        # Copy pdb and psf file
        target_str_dir = check_dir(os.path.join(root_dir, "str"))
        target_pdb_file_path = os.path.join(target_str_dir, structure_name + ".pdb")
        target_psf_file_path = os.path.join(target_str_dir, structure_name + ".psf")
        os.system("cp %s %s" % (pdb_file_path, target_pdb_file_path))
        os.system("cp %s %s" % (psf_file_path, target_psf_file_path))
        simulator = Simulator(
            pdb_file_path=target_pdb_file_path,
            psf_file_path=target_psf_file_path,
            parameter_file_paths=PARAMETER_FILE_PATHS,
            out_dir=root_dir,
            cuda_index=cuda_index,
        )
        for index in range(num_simulations):
            sim_recipe = sim_dict[str(index)]
            args = {}
            for key, value in sim_recipe.items():
                if not "order" in key and not "name" in key:
                    args[key] = sim_recipe[key]
            restart_file_path = os.path.join(
                root_dir, sim_recipe["out_prefix"], "restart.pdb"
            )
            if not os.path.exists(restart_file_path):
                getattr(simulator, sim_recipe["name"])(**args)
            else:
                simulator.load_state(out_dir=os.path.dirname(restart_file_path))
                print("Skip %s" % sim_recipe["out_prefix"])
    except:
        error = "Failed %s at %s" % (
            json_file_path,
            datetime.datetime.now().replace(microsecond=0),
        )
        error += traceback.format_exc()
        return error
    return "Finished %s successfully at %s" % (
        json_file_path,
        datetime.datetime.now().replace(microsecond=0),
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    json_file_path = args[0]
    cuda_index = int(args[1])
    message = execute_json(json_file_path=json_file_path, cuda_index=cuda_index)
    # post(message)
    print(message)

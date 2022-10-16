#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : metad.py
created time : 2022/10/16
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import numpy as np
from itertools import product
from mdpy.unit import *
from mdpy.utils import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CUR_DIR, "../..")
EXECUTION_FILE_PATH = os.path.join(CODE_DIR, "distributor.py")
sys.path.append(CODE_DIR)
from job import generate_json
from str.generator import CC_BOND_LENGTH


def generate_simulation_recipe(center_ion_type, z_range):
    simulation_recipe = [
        {"name": "minimize", "max_iterations": 500, "out_prefix": "01-minimize"},
        {
            "name": "equilibrate_npt",
            "num_steps": 500000,
            "step_size": Quantity(1, femtosecond),
            "temperature": Quantity(300, kelvin),
            "pressure": 1.0,
            "out_prefix": "02-eq-npt",
            "out_freq": 10000,
        },
        {
            "name": "equilibrate_nvt",
            "num_steps": 100000,
            "step_size": Quantity(0.5, femtosecond),
            "temperature": Quantity(300, kelvin),
            "out_prefix": "03-eq-nvt",
            "out_freq": 10000,
        },
        {
            "name": "sample_metadynamics",
            "num_steps": 25000000,
            "step_size": Quantity(1, femtosecond),
            "temperature": Quantity(300, kelvin),
            "center_ion_type": center_ion_type,
            "z_range": z_range,
            "out_prefix": "04-sample-metad",
            "out_freq": 1000,
        },
    ]
    return simulation_recipe


def generate_json_file_path(
    out_dir, r0, w0, l0, ls, ions, wall_charges, center_ion_type
):
    ion_name = "-".join(
        [
            "%s-%.2emolPerL"
            % (key.upper(), check_quantity_value(value, mol / decimeter**3))
            for key, value in ions.items()
            if check_quantity_value(value, mol / decimeter**3) != 0
        ]
    )
    str_name = "pore-r0-%.3fA-w0-%.3fA-l0-%.3fA-ls-%.3fA" % (
        check_quantity_value(r0, angstrom),
        check_quantity_value(w0, angstrom),
        check_quantity_value(l0, angstrom),
        check_quantity_value(ls, angstrom),
    )
    wall_charge_name = "-".join(
        [
            "z0-%.3fA-q-%.2fe-n-%d"
            % (
                check_quantity_value(i["z0"], angstrom),
                check_quantity_value(i["q"], elementary_charge),
                i["n"],
            )
            for i in wall_charges
            if i["q"] != 0
        ]
    )
    if wall_charge_name == "":
        wall_charge_name = "no-wall-charge"
    center_ion_name = "%s" % center_ion_type.lower()
    return os.path.join(
        out_dir,
        wall_charge_name,
        ion_name,
        str_name + "-" + center_ion_name,
        "job.json",
    )


if __name__ == "__main__":
    out_dir = os.path.join(CUR_DIR, "out/metad")
    job_args = []
    r0_list = [
        Quantity(i * CC_BOND_LENGTH * 3 / (2 * np.pi), angstrom) for i in range(8, 20)
    ]
    l0_list = [Quantity(50, angstrom)]
    w0_list = [Quantity(50, angstrom)]
    ls_list = [Quantity(25, angstrom)]
    ions_list = [
        {"POT": Quantity(0.1, mol / decimeter**3)},
    ]
    center_ion_type_list = ["POT", "CLA"]
    wall_charges_list = [[]]
    job_args.extend(
        list(
            product(
                r0_list,
                l0_list,
                w0_list,
                ls_list,
                ions_list,
                wall_charges_list,
                center_ion_type_list,
            )
        )
    )
    z_padding = 10
    z_range = [-25 - z_padding, 25 + z_padding]
    json_file_path_list = []
    for (r0, l0, w0, ls, ions, wall_charges, center_ion_type) in job_args:
        json_file_path = generate_json_file_path(
            out_dir=out_dir,
            r0=r0,
            l0=l0,
            w0=w0,
            ls=ls,
            ions=ions,
            center_ion_type=center_ion_type,
            wall_charges=wall_charges,
        )
        json_file_path_list.append(
            generate_json(
                json_file_path=json_file_path,
                r0=r0,
                l0=l0,
                w0=w0,
                ls=ls,
                wall_charges=wall_charges,
                ions=ions,
                simulation_recipes=generate_simulation_recipe(center_ion_type, z_range),
            )
        )
    command = "python %s %s" % (EXECUTION_FILE_PATH, " ".join(json_file_path_list))
    os.system(command)

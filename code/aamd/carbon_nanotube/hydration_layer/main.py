#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : main.py
created time : 2022/09/23
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
CODE_DIR = os.path.join(CUR_DIR, "..")
EXECUTION_FILE_PATH = os.path.join(CODE_DIR, "manager.py")
sys.path.append(CODE_DIR)
from job import generate_json
from str.generator import CC_BOND_LENGTH


def generate_simulation_recipe(center_ion_type):
    return [
        {"name": "minimize", "max_iterations": 500, "out_prefix": "01-minimize"},
        {
            "name": "equilibrate_nvt",
            "num_steps": 300000,
            "step_size": Quantity(0.01, femtosecond),
            "temperature": Quantity(300, kelvin),
            "out_prefix": "02-eq-nvt",
            "out_freq": 10000,
        },
        {
            "name": "equilibrate_npt",
            "num_steps": 400000,
            "step_size": Quantity(1, femtosecond),
            "temperature": Quantity(300, kelvin),
            "pressure": 1.0,
            "out_prefix": "03-eq-npt",
            "out_freq": 10000,
        },
        {
            "name": "equilibrate_hydration_ion",
            "num_steps": 1000000,
            "step_size": Quantity(0.5, femtosecond),
            "temperature": Quantity(300, kelvin),
            "center_ion_type": center_ion_type,
            "out_prefix": "04-eq-hydration-ion",
            "out_freq": 10000,
        },
        {
            "name": "equilibrate_fixed_hydration_ion",
            "num_steps": 500000,
            "step_size": Quantity(0.01, femtosecond),
            "temperature": Quantity(300, kelvin),
            "center_ion_type": center_ion_type,
            "out_prefix": "05-eq-fixed-hydration-ion",
            "out_freq": 10000,
        },
        {
            "name": "sample_hydration_ion",
            "num_steps": 10000000,
            "step_size": Quantity(1, femtosecond),
            "temperature": Quantity(300, kelvin),
            "center_ion_type": center_ion_type,
            "out_prefix": "06-sample-hydration-ion",
            "out_freq": 10000,
        },
    ]


def generate_json_file_path(
    out_dir, r0, w0, l0, ls, ions, wall_charges, center_ion_type
):
    ion_name = "-".join(
        [
            "%s-%.2emolPerL"
            % (key.upper(), check_quantity_value(value, mol / decimeter**3))
            for key, value in ions.items()
        ]
    )
    str_name = "r0-%.3fA-w0-%.3fA-l0-%.3fA-ls-%.3fA" % (
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
        ]
    )
    center_ion_name = "center-%s" % center_ion_type
    return os.path.join(
        out_dir,
        str_name + "-" + ion_name,
        wall_charge_name,
        center_ion_name,
        "job.json",
    )


if __name__ == "__main__":
    out_dir = os.path.join(CUR_DIR, "out")
    if False:
        json_file_path_list = []
        center_ion_type_list = ["POT", "CLA"]
        r0 = Quantity(5, angstrom)
        l0 = Quantity(50, angstrom)
        w0 = Quantity(50, angstrom)
        ls = Quantity(50, angstrom)
        ions = {"POT": Quantity(0.01, mol / decimeter**3)}
        wall_charges = [{"z0": Quantity(0, angstrom), "q": 0, "n": 0}]
        for center_ion_type in center_ion_type_list:
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
                    simulation_recipes=generate_simulation_recipe(center_ion_type),
                )
            )
        command = "python %s %s" % (EXECUTION_FILE_PATH, " ".join(json_file_path_list))
        os.system(command)
    r0_list = [
        Quantity(i * CC_BOND_LENGTH * 3 / (2 * np.pi), angstrom) for i in range(2, 20)
    ]
    l0_list = [Quantity(50, angstrom)]
    w0_list = [Quantity(50, angstrom)]
    ls_list = [Quantity(50, angstrom)]
    center_ion_type_list = ["POT", "CLA"]
    ions_list = [
        {"POT": Quantity(0.10, mol / decimeter**3)},
    ]
    wall_charges_list = [
        [{"z0": Quantity(0, angstrom), "q": 0.5, "n": 3}],
        [{"z0": Quantity(0, angstrom), "q": 1, "n": 3}],
        [{"z0": Quantity(0, angstrom), "q": 1.5, "n": 3}],
        [{"z0": Quantity(0, angstrom), "q": 2, "n": 3}],
    ]
    job_args = list(
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
    print("%s jobs in total" % (len(job_args)))
    json_file_path_list = []
    for r0, l0, w0, ls, ions, wall_charges, center_ion_type in job_args:
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
                simulation_recipes=generate_simulation_recipe(center_ion_type),
            )
        )
    command = "python %s %s" % (EXECUTION_FILE_PATH, " ".join(json_file_path_list))
    os.system(command)
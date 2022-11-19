#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : long_time.py
created time : 2022/11/19
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


def generate_simulation_recipe():
    sim_step = int(200 * 1e6)
    simulation_recipe = [
        {"name": "minimize", "max_iterations": 500, "out_prefix": "01-minimize"},
        {
            "name": "equilibrate_npt",
            "num_steps": 1000000,
            "step_size": Quantity(1, femtosecond),
            "temperature": Quantity(300, kelvin),
            "pressure": 1.0,
            "out_prefix": "02-eq-npt",
            "out_freq": 10000,
        },
        {
            "name": "equilibrate_nvt",
            "num_steps": 500000,
            "step_size": Quantity(0.1, femtosecond),
            "temperature": Quantity(300, kelvin),
            "out_prefix": "03-eq-nvt",
            "out_freq": 10000,
        },
        {
            "name": "sample_nvt",
            "num_steps": sim_step,
            "step_size": Quantity(1.0, femtosecond),
            "temperature": Quantity(300, kelvin),
            "out_prefix": "04-sample",
            "out_freq": 5000,
        },
    ]
    return simulation_recipe


def generate_json_file_path(out_dir, r0, w0, l0, ls, ions, index=0):
    ion_name = "-".join(
        [
            "%s-%.2emolPerL"
            % (key.upper(), check_quantity_value(value, mol / decimeter**3))
            for key, value in ions.items()
            if check_quantity_value(value, mol / decimeter**3) != 0
        ]
    )
    if ion_name == "":
        ion_name = "no-ion"
    if check_quantity_value(l0, angstrom) != 0:
        str_name = "pore-r0-%.3fA-w0-%.3fA-l0-%.3fA-ls-%.3fA" % (
            check_quantity_value(r0, angstrom),
            check_quantity_value(w0, angstrom),
            check_quantity_value(l0, angstrom),
            check_quantity_value(ls, angstrom),
        )
    else:
        str_name = "no-pore-w0-%.3fA-ls-%.3fA" % (
            check_quantity_value(r0, angstrom),
            check_quantity_value(ls, angstrom),
        )
    return os.path.join(
        out_dir,
        "no-wall-charge-long-time",
        str_name + "-" + ion_name + "-" + str(index),
        "job.json",
    )


if __name__ == "__main__":
    out_dir = os.path.join(CUR_DIR, "out")
    job_args = []
    channel_r0_list = [Quantity(30 * CC_BOND_LENGTH * 3 / (2 * np.pi), angstrom)]
    print(channel_r0_list)
    # Job for ion in bulk
    r0_list = [Quantity(2, angstrom)]
    l0_list = [Quantity(0, angstrom)]
    w0_list = [Quantity(50, angstrom)]
    ls_list = [Quantity(25, angstrom)]
    ions_list = [
        {"POT": Quantity(0.1, mol / decimeter**3)},
    ]
    wall_charges_list = [[]]
    # Job for water in channel
    r0_list = channel_r0_list
    l0_list = [Quantity(50, angstrom)]
    ls_list = [Quantity(25, angstrom)]
    job_args.extend(
        list(
            product(
                r0_list,
                l0_list,
                w0_list,
                ls_list,
                ions_list,
                wall_charges_list,
            )
        )
    )
    print("%s jobs in total" % (len(job_args)))
    json_file_path_list = []
    for (
        r0,
        l0,
        w0,
        ls,
        ions,
        wall_charges,
    ) in job_args:
        for i in range(6):
            json_file_path = generate_json_file_path(
                out_dir=out_dir, r0=r0, l0=l0, w0=w0, ls=ls, ions=ions, index=i
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
                    simulation_recipes=generate_simulation_recipe(),
                )
            )
    command = "python %s %s" % (EXECUTION_FILE_PATH, " ".join(json_file_path_list))
    # os.system(command)

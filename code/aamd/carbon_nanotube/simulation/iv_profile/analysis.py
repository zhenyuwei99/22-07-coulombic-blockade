#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : analysis.py
created time : 2023/01/04
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import mdpy as md
import numpy as np
import matplotlib.pyplot as plt
from mdpy.unit import *
from main import generate_json_file_path


def get_target_dir_list(out_dir, r0, w0, l0, ls, ions, wall_charges):
    json_file_path = generate_json_file_path(
        out_dir=out_dir,
        r0=r0,
        l0=l0,
        w0=w0,
        ls=ls,
        ions=ions,
        wall_charges=wall_charges,
        electric_field=0,
    )
    parent_dir = os.path.dirname(os.path.dirname(json_file_path))
    target_dir_list, electric_field_list = [], []
    for i in os.listdir(parent_dir):
        target_dir = os.path.join(parent_dir, i)
        if os.path.exists(os.path.join(target_dir, "05-sample-nvt-ele/restart.pdb")):
            target_dir_list.append(target_dir)
            electric_field_list.append(float(i.split("VPerNm")[0]))
    electric_field_list, target_dir_list = [
        list(i) for i in zip(*sorted(zip(electric_field_list, target_dir_list)))
    ]
    print(target_dir_list)
    return target_dir_list, electric_field_list


def analysis_current(target_dir, electric_field, ion_type):
    npt_dir = "03-eq-npt"
    sample_file_name = "05-sample-nvt-ele/05-sample-nvt-ele.ion"
    # PBC matrix
    pdb = md.io.PDBParser(os.path.join(target_dir, npt_dir, "restart.pdb"))
    pbc_matrix = pdb.pbc_matrix
    pbc_inv = np.linalg.inv(pbc_matrix)
    # Read ion
    data = []
    with open(os.path.join(target_dir, sample_file_name), "r") as f:
        line = f.readline()
        current_data = []
        while line:
            if line.startswith("Step"):
                current_data = []
                while line != "\n":
                    line = f.readline()
                    if line.startswith(ion_type):
                        current_data.append(float(line.split()[-1]))
                data.append(current_data)
                line = f.readline()
            else:
                line = f.readline()
    data = np.array(data)
    factor = 1 if electric_field >= 0 else -1
    # num_events = np.round((np.abs(data).max(0) / pbc_matrix[2, 2]).sum())
    num_events = np.round(np.abs(data).max(0) / pbc_matrix[2, 2]).sum()
    current = Quantity(num_events, elementary_charge) / Quantity(50, nanosecond)
    current = current.convert_to(ampere).value
    voltage = Quantity(electric_field, volt / nanometer) * Quantity(
        pbc_matrix[2, 2], angstrom
    )
    voltage = voltage.convert_to(volt).value
    return voltage, current * factor


if __name__ == "__main__":
    out_dir = "/home/zhenyuwei/hdd2/22-07-coulombic-blockade/aamd/carbon_nanotube/simulation/iv_profile/out"
    r0 = 10.156
    w0 = 50
    l0 = 50
    ls = 25
    ions = {"POT": Quantity(0.15, mol / decimeter**3)}
    wall_charges = []
    target_dir_list, electric_field_list = get_target_dir_list(
        out_dir=out_dir,
        r0=r0,
        l0=l0,
        w0=w0,
        ls=ls,
        ions=ions,
        wall_charges=wall_charges,
    )
    voltage_list, current_list = [], []
    for target_dir, electric_field in zip(target_dir_list, electric_field_list):
        voltage, current = analysis_current(target_dir, electric_field, "POT")
        voltage_list.append(voltage)
        current_list.append(current)
        print(voltage, current)
    plt.plot(voltage_list, current_list, ".-")
    plt.xlim(0, 20)
    plt.ylim(0, 6e-9)
    plt.show()

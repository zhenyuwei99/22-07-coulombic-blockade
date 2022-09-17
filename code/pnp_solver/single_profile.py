#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : single_profile.py
created time : 2022/09/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
from mdpy.unit import *
from job import *
from utils import *
from analyzer import *


def generate_jobs(
    root_dir,
    voltage_range: list,
    num_jobs: int,
    ion_types: list,
    ion_density: list,
    ion_ls: list,
    r0,
    l0,
    w0,
    grid_width=Quantity(1.0, angstrom),
):
    object_root_dir = os.path.join(root_dir, STR_NAME % (r0, l0, w0))
    json_file_paths = []
    ion_data = {}
    for i, j, k in zip(ion_types, ion_density, ion_ls):
        ion_data["%s_density" % i] = Quantity(j, mol / decimeter ** 3)
        ion_data["%s_ls" % i] = Quantity(k, angstrom)
    for voltage in np.linspace(voltage_range[0], voltage_range[1], num_jobs):
        voltage_name = "%.4fV" % voltage
        job_root_dir = check_dir(os.path.join(object_root_dir, voltage_name))
        json_file_paths.append(
            generate_json(
                json_file_path=os.path.join(job_root_dir, "job.json"),
                r0=Quantity(r0, angstrom),
                l0=Quantity(l0, angstrom),
                w0=Quantity(w0, angstrom),
                grid_width=grid_width,
                voltage=Quantity(voltage, volt),
                **ion_data,
            )
        )
    return json_file_paths, object_root_dir


def excute(
    json_file_paths,
    execution_file_path,
    device_file_path,
    num_devices,
    num_jobs_per_device,
):
    init_device_file(device_file_path, num_devices, num_jobs_per_device)
    os.system(
        "python %s %s %d %d "
        % (execution_file_path, device_file_path, num_devices, num_jobs_per_device)
        + " ".join(json_file_paths)
    )


def analysis(root_dir, voltage_range):
    analyzer = PNPAnalyzer(root_dir)
    current_functions = analyzer.analysis()
    voltage = (
        Quantity(
            np.linspace(voltage_range[0], voltage_range[1], 100, endpoint=True), volt
        )
        .convert_to(default_voltage_unit)
        .value
    )
    current_pred = np.array([f(voltage) for f in current_functions]).sum(0)
    # Visulize
    img_file_path = os.path.join(root_dir, "iv-curve.png")
    fig, ax = plt.subplots(1, 1, figsize=[15, 6])
    voltage = Quantity(voltage, default_voltage_unit).convert_to(volt).value
    current_pred = Quantity(current_pred, default_current_unit).convert_to(ampere).value
    ax.plot(voltage, current_pred, label="PNP solution", color="navy")
    ax.set_xlabel("Voltage (V)", fontsize=20)
    ax.set_ylabel("Current (A)", fontsize=20)
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig(img_file_path)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = check_dir(os.path.join(cur_dir, "out/voltage-range/"))
    execution_file_path = os.path.join(cur_dir, "job.py")
    device_file_path = os.path.join(root_dir, "device.h5")
    voltage_range = [-25, 25]
    json_file_paths, object_root_dir = generate_jobs(
        root_dir=root_dir,
        voltage_range=voltage_range,
        num_jobs=50,
        ion_types=["pot", "cla"],
        ion_density=[1, 1],
        ion_ls=[30, 30],
        r0=20,
        l0=250,
        w0=100,
    )
    excute(
        json_file_paths=json_file_paths,
        execution_file_path=execution_file_path,
        device_file_path=device_file_path,
        num_devices=3,
        num_jobs_per_device=2,
    )
    analysis(root_dir=object_root_dir, voltage_range=voltage_range)

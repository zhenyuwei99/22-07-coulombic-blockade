#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : main.py
created time : 2022/07/31
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import shutil
import multiprocessing as mp
from time import sleep
import openmm.unit as unit
from datetime import datetime
from modeler import Modeler
from simulator import Simulator
from manager import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "data")
OUT_DIR = os.path.join(CUR_DIR, "out")


def create_new_structure():
    # Model
    box_size = [20, 20, 5]
    solvation_box_height = 100
    template_file_path = os.path.join(data_dir, "modeler_template.tcl")
    modeler = Modeler(
        template_file_path=template_file_path,
        box_size=box_size,
        pore_radius=pore_radius,
        solvation_box_height=solvation_box_height,
        ion_concentration=ion_concentration,
        out_dir=str_dir,
    )
    modeler.model()
    # Equilibrium
    simulator = Simulator(
        str_dir=str_dir,
        str_name="str",
        parameter_file_paths=[
            os.path.join(data_dir, "par_sio2.prm"),
            os.path.join(data_dir, "par_water.prm"),
        ],
        out_dir=str_dir,
        cuda_index=1,
    )
    simulator.minimize(max_iterations=2500, out_prefix="00_minimize")
    simulator.equilibrium_nvt(
        num_steps=1000000,
        step_size=0.1 * unit.femtosecond,
        temperature=300 * unit.kelvin,
        langevin_factor=1 / unit.picosecond,
        out_prefix="01_nvt_eq",
        out_freq=1000,
    )
    simulator.equilibrium_npt(
        num_steps=1000000,
        step_size=0.5 * unit.femtosecond,
        temperature=300 * unit.kelvin,
        pressure=1 * unit.bar,
        langevin_factor=1 / unit.picosecond,
        out_prefix="02_npt_eq",
        out_freq=1000,
    )
    simulator.equilibrium_nvt(
        num_steps=1000000,
        step_size=1 * unit.femtosecond,
        temperature=300 * unit.kelvin,
        langevin_factor=1 / unit.picosecond,
        out_prefix="03_nvt_eq",
        out_freq=1000,
    )


def sample(
    device_file_path: str, voltage, length, out_dir: str, str_dir, restart_dir: str
):
    job_name = generate_job_name(voltage)
    if not is_job_finished(job_name, out_dir):
        device, job = get_available_device(device_file_path)
        register_device(device_file_path, device, job)
        print(
            "Submit %s to device-%d-job-%d at %s"
            % (job_name, device, job, datetime.now().replace(microsecond=0))
        )
        job_dir = check_dir(os.path.join(out_dir, job_name))
        simulator = Simulator(
            str_dir=str_dir,
            str_name="str",
            parameter_file_paths=[
                os.path.join(data_dir, "par_sio2.prm"),
                os.path.join(data_dir, "par_water.prm"),
            ],
            out_dir=job_dir,
            cuda_index=device,
        )
        simulator.load_state(restart_dir)
        simulator.equilibrium_nvt_with_external_field(
            num_steps=500000,
            step_size=1 * unit.femtosecond,
            temperature=300 * unit.kelvin,
            voltage=voltage,
            length=length,
            langevin_factor=1 / unit.picosecond,
            out_prefix="eq",
            out_freq=5000,
        )
        simulator.sample_nvt_with_external_field(
            num_steps=5000000,
            step_size=2 * unit.femtosecond,
            temperature=300 * unit.kelvin,
            voltage=voltage,
            length=length,
            langevin_factor=1 / unit.picosecond,
            out_prefix="sample",
            out_freq=1000,
        )
        free_device(device_file_path, device, job)
        print(
            "Finish %s to device-%d-job-%d at %s"
            % (job_name, device, job, datetime.now().replace(microsecond=0))
        )


def check_dir(dir_path: str, restart=False):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if restart:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    return dir_path


def generate_job_name(voltage: unit.Quantity):
    voltage = voltage / unit.volt
    prefix = "minus" if voltage < 0 else "plus"
    voltage = abs(voltage)
    return "%s-%.2f-volt" % (prefix, voltage)


def is_job_finished(job_name: str, out_dir: str):
    if os.path.exists(os.path.join(out_dir, job_name, "sample/restart.pdb")):
        return True
    return False


if __name__ == "__main__":
    pore_radius = 15
    ion_concentration = 0.5
    is_new_structure = False
    # Initialization
    root_dir = os.path.join(OUT_DIR, "%.2f-nm-pore" % pore_radius)
    out_dir = check_dir(os.path.join(root_dir, "out"), restart=True)
    str_dir = check_dir(os.path.join(root_dir, "str"))
    data_dir = check_dir(os.path.join(root_dir, "data"))
    os.system("cp -r %s %s" % (DATA_DIR, root_dir))
    if is_new_structure:
        create_new_structure()
    # Manager
    num_devices = 4
    num_jobs_per_device = 2
    num_total_jobs = num_devices * num_jobs_per_device
    device_file_path = init_device_file(
        file_path=os.path.join(CUR_DIR, "device.h5"),
        num_devices=num_devices,
        num_jobs_per_device=num_jobs_per_device,
    )
    # Submitting jobs
    pool = mp.Pool(num_total_jobs)
    length = 34.740 * 8 * unit.angstrom  # PNP dimension
    restart_dir = os.path.join(str_dir, "03_nvt_eq")
    target_voltage = np.linspace(-4, 4, 16)
    interval = target_voltage.shape[0] // num_total_jobs
    for i in range(interval):
        for voltage in target_voltage[i::interval]:
            voltage = voltage * unit.volt
            # sample(device_file_path, voltage, length, out_dir, str_dir, restart_dir)
            pool.apply_async(
                sample,
                args=(device_file_path, voltage, length, out_dir, str_dir, restart_dir),
            )
            sleep(2)
    pool.close()
    pool.join()

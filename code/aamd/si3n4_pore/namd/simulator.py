#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : simulator.py
created time : 2022/07/11
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np
from mdpy.utils import check_quantity_value
from mdpy.unit import *

MODEL_FILE_NAME = "model.tcl"


class NAMDNanoChannelSimulator:
    def __init__(
        self,
        # Dir parameters
        template_dir: str,
        forcefield_dir: str,
        simulation_dir: str,
        # Simulation parameters
        temperature: Quantity,
        # Other parameters
        is_clear_cache: bool = True,
    ) -> None:
        self._template_dir = template_dir
        self._template_forcefield_dir = forcefield_dir
        self._simulation_dir = simulation_dir
        self._str_dir = os.path.join(self._simulation_dir, "str")
        self._forcefield_dir = os.path.join(self._simulation_dir, "forcefield")
        self._out_dir = os.path.join(self._simulation_dir, "out")
        self._generate_dir(is_clear_cache=is_clear_cache)

    def _generate_dir(self, is_clear_cache: bool) -> None:
        if is_clear_cache:
            # Clean cache
            if os.path.exists(self._simulation_dir):
                os.system("rm -rf %s/*" % self._simulation_dir)
            else:
                os.mkdir(self._simulation_dir)
            # Make dir
            os.mkdir(self._str_dir)
            os.mkdir(self._forcefield_dir)
            os.mkdir(self._out_dir)
            # Copy forcefield dir
            os.system(
                "cp %s/* %s" % (self._template_forcefield_dir, self._forcefield_dir)
            )
        elif not os.path.exists(self._simulation_dir):
            os.mkdir(self._simulation_dir)
            # Make dir
            os.mkdir(self._str_dir)
            os.mkdir(self._forcefield_dir)
            os.mkdir(self._out_dir)
            # Copy forcefield dir
            os.system(
                "cp %s/* %s" % (self._template_forcefield_dir, self._forcefield_dir)
            )

    def _generate_file_from_template(
        self, template_file_name: str, target_file_name: str, *args
    ) -> None:
        template_file_path = os.path.join(self._template_dir, template_file_name)
        target_file_path = os.path.join(self._simulation_dir, target_file_name)
        with open(template_file_path, "r") as f:
            template = f.read()
        with open(target_file_path, "w") as f:
            print(template % args, file=f)

    def model(
        self,
        lattice_diameter: int,
        lattice_height: int,
        pore_radius: Quantity,
        solvation_box_height: Quantity,
        ion_concentration: Quantity,
        num_fixed_ions: int,
        ion_charge: Quantity,
    ) -> None:
        pore_radius = check_quantity_value(pore_radius, angstrom)
        self._generate_file_from_template(
            MODEL_FILE_NAME,
            MODEL_FILE_NAME,
            lattice_diameter,
            lattice_height,
            check_quantity_value(pore_radius, angstrom),
            check_quantity_value(solvation_box_height, angstrom),
            check_quantity_value(ion_concentration, mol / decimeter**3),
        )
        os.system(
            "cd %s && vmd -dispdev text -e %s" % (self._simulation_dir, MODEL_FILE_NAME)
        )

    def minimize(self, input_prefix: str, output_prefix: str, output_freq: int):
        template_file_name = "min.namd"
        target_file_name = output_prefix.split("/")[-1] + ".namd"
        input_prefix = os.path.join(".", input_prefix)
        output_prefix = os.path.join(".", output_prefix)
        cell_data = []
        with open(os.path.join(self._str_dir, "cell.txt"), "r") as f:
            text_data = f.read().split("\n")
        for i in range(4):
            cell_data.extend([float(i) for i in text_data[i].split()])
        cell_data[8] = float(text_data[4])
        self._generate_file_from_template(
            template_file_name,
            target_file_name,
            input_prefix,
            output_prefix,
            output_freq,
            *cell_data,
        )
        os.system(
            "cd %s && namd3 +p6 %s > %s.log"
            % (self._simulation_dir, target_file_name, output_prefix)
        )

    def equilibrate_nvt(
        self,
        input_prefix: str,
        output_prefix: str,
        output_freq: int,
        num_steps: int,
        time_step: Quantity,
    ) -> None:
        template_file_name = "eq_nvt.namd"
        target_file_name = output_prefix.split("/")[-1] + ".namd"
        input_prefix = os.path.join(".", input_prefix)
        output_prefix = os.path.join(".", output_prefix)
        cell_data = []
        with open(os.path.join(self._str_dir, "cell.txt"), "r") as f:
            text_data = f.read().split("\n")
        for i in range(4):
            cell_data.extend([float(i) for i in text_data[i].split()])
        cell_data[8] = float(text_data[4])
        self._generate_file_from_template(
            template_file_name,
            target_file_name,
            input_prefix,
            output_prefix,
            output_freq,
            num_steps,
            check_quantity_value(time_step, femtosecond),
            *cell_data,
        )
        os.system(
            "cd %s && namd3 +p12 %s > %s.log"
            % (self._simulation_dir, target_file_name, output_prefix)
        )

    def sample(self) -> None:
        pass


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(cur_dir, "template")
    forcefield_dir = os.path.join(cur_dir, "forcefield")
    simulation_dir = os.path.join(cur_dir, "simulation")
    simulator = NAMDNanoChannelSimulator(
        template_dir=template_dir,
        forcefield_dir=forcefield_dir,
        simulation_dir=os.path.join(simulation_dir, "test"),
        temperature=Quantity(300, kelvin),
        is_clear_cache=False,
    )
    # simulator.model(
    #     lattice_diameter=12,
    #     lattice_height=20,
    #     pore_radius=Quantity(10, angstrom),
    #     solvation_box_height=Quantity(100, angstrom),
    #     ion_concentration=Quantity(0.15, mol / decimeter**3),
    #     num_fixed_ions=0,
    #     ion_charge=Quantity(1, elementary_charge),
    # )
    # simulator.minimize(
    #     input_prefix="str/str", output_prefix="out/01_min", output_freq=1000
    # )
    simulator.equilibrate_nvt(
        input_prefix="out/01_min",
        output_prefix="out/02_eq_nvt",
        output_freq=5000,
        num_steps=50000,
        time_step=Quantity(0.5, femtosecond),
    )

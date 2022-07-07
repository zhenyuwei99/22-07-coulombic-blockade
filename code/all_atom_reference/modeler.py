#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : modeler.py
created time : 2022/06/29
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "data")
OUT_DIR = os.path.join(CUR_DIR, "str")


class Modeler:
    def __init__(
        self,
        template_file_path: str,
        box_size: list[int],
        pore_radius: float,
        solvation_box_height: float,
        ion_concentration: float,
        out_dir: str = OUT_DIR,
    ) -> None:
        self._template = open(template_file_path, "r").read()
        self._box_size = box_size
        self._pore_radius = pore_radius
        self._solvation_box_height = solvation_box_height
        self._ion_concentration = ion_concentration
        self._out_dir = out_dir

        self._tcl_file_name = "model.tcl"
        self._tcl_file_path = os.path.join(self._out_dir, self._tcl_file_name)

    def _generate_tcl_file(self):
        with open(self._tcl_file_path, "w") as f:
            print(
                self._template
                % (
                    self._box_size[0],
                    self._box_size[1],
                    self._box_size[2],
                    self._pore_radius,
                    self._solvation_box_height,
                    self._ion_concentration,
                ),
                file=f,
            )

    def model(self):
        os.system("rm -rf %s/*.pdb" % self._out_dir)
        os.system("rm -rf %s/*.psf" % self._out_dir)
        self._generate_tcl_file()
        os.system(
            "cd %s && vmd -dispdev text -e %s" % (self._out_dir, self._tcl_file_name)
        )
        os.system("rm -rf %s/*.log" % self._out_dir)
        os.system("rm -rf %s/*.tcl" % self._out_dir)


if __name__ == "__main__":
    # Parameter
    template_file_path = os.path.join(DATA_DIR, "modeler_template.tcl")
    box_size = [20, 20, 5]
    pore_radius = 10
    solvation_box_height = 100
    ion_concentration = 0.15
    # Model
    modeler = Modeler(
        template_file_path=template_file_path,
        box_size=box_size,
        pore_radius=pore_radius,
        solvation_box_height=solvation_box_height,
        ion_concentration=ion_concentration,
    )
    modeler.model()

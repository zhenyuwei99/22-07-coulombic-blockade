#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : str.py
created time : 2022/09/17
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import numpy as np
from mdpy.utils import *
from mdpy.unit import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILE = os.path.join(CUR_DIR, "template.tcl")
SI3N4_LATTICE_MATRIX = np.array([[7.595, 0, 0], [3.798, 6.578, 0], [0, 0, 2.902]])
SI3N4_LATTICE_LENGTH = np.sqrt((SI3N4_LATTICE_MATRIX ** 2).sum(1))
CRYST1 = "CRYST1" + "%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f\n"
STRUCTURE_NAME = "r0-%.3fA-w0-%.3fA-l0-%.3fA-ls-%.3fA"


def generate_structure(r0, w0, l0, ls):
    """
    - `r0` (A): The radius of Si3N4 pore
    - `w0` (A): The width (length on the first and second base vector) of Si3N4 substrate
    - `l0` (A): The thickness of Si3N4 substrate
    - `ls` (A): The thickness of solution on one side
    """
    r0 = check_quantity_value(r0, angstrom)
    w0 = check_quantity_value(w0, angstrom)
    l0 = check_quantity_value(l0, angstrom)
    ls = check_quantity_value(ls, angstrom)

    box_size = [
        int(np.round(w0 / SI3N4_LATTICE_LENGTH[0])),
        int(np.round(w0 / SI3N4_LATTICE_LENGTH[1])),
        int(np.round(l0 / SI3N4_LATTICE_LENGTH[2])),
    ]
    w0 = box_size[0] * SI3N4_LATTICE_LENGTH[0]
    l0 = box_size[2] * SI3N4_LATTICE_LENGTH[2]
    cryst1_line = CRYST1 % (w0, w0, l0, 90, 90, 60)
    structure_name = STRUCTURE_NAME % (r0, w0, l0, ls)
    if not os.path.exists(os.path.join(CUR_DIR, structure_name + ".pdb")):
        with open(TEMPLATE_FILE, "r") as f:
            tcl = f.read()
        tcl = tcl % (structure_name, box_size[0], box_size[1], box_size[2], r0)
        tcl_file = os.path.join(CUR_DIR, structure_name + ".tcl")
        with open(tcl_file, "w") as f:
            print(tcl, file=f)


if __name__ == "__main__":
    generate_structure(5, 50, 50, 20)

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : job.py
created time : 2022/09/21
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import datetime
import h5py
import numpy as np
import openmm.openmm as openmm
import openmm.app as app
import openmm.unit as unit
from mdpy.unit import *


def parse_restrain(pdb_file_path: str):
    restrain_constant = []
    with open(pdb_file_path, "r") as f:
        line = f.readline()
        index = 0
        while line:
            line = f.readline()
            index += 1
            restrain_constant.append(float(line[61:67]))
            if restrain_constant[-1] == 0:
                break
    restrain_constant = (
        np.array(restrain_constant[:-1])
        * (unit.kilocalorie_per_mole / unit.angstrom**2)
        / (unit.kilojoule_per_mole / unit.nanometer**2)
    )
    restrain_index = np.array(list(range(index - 1)), np.int32)
    restrain_origin = pdb.getPositions(asNumpy=True)[:index, :] / unit.nanometer
    num_restrained_particles = index - 1


def equilibrate_nvt(
    pdb_file_path: str,
    psf_file_path: str,
    restart_pdb_file_path: str,
    temperature: Quantity,
    step_size: Quantity,
    num_steps: int,
):
    pass


def equilibrate_npt():
    pass


def sample_nvt():
    pass


if __name__ == "__main__":
    pass

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : current_analyzer.py
created time : 2022/09/25
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import json
import numpy as np
import mdpy as md
from mdpy.unit import *
from mdpy.utils import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STR_DIR = os.path.join(CUR_DIR, "../str")
OUT_DIR = os.path.join(CUR_DIR, "out")
RES_DIR = os.path.join(CUR_DIR, "res")
CODE_DIR = os.path.join(CUR_DIR, "..")
sys.path.append(CODE_DIR)
from str.generator import *


class CurrentAnalyzer:
    def __init__(
        self,
        root_dir: str,
        sample_prefix: str,
        str_dir: str = STR_DIR,
        res_dir: str = RES_DIR,
    ) -> None:
        self._root_dir = root_dir
        self._sample_prefix = sample_prefix
        self._str_dir = str_dir
        self._res_dir = res_dir
        self._target_jobs, self._ele_list = self._get_jobs()
        self._structure_name, self._topology, self._ions = self._get_structure()
        self._res_file_path = os.path.join(self._res_dir, self._structure_name + ".npz")

    def _get_jobs(self):
        target_jobs = os.listdir(self._root_dir)
        ele_list = [float(i.split("VPerNm")[0]) for i in target_jobs]
        ele_list, target_jobs = [
            list(i) for i in zip(*sorted(zip(ele_list, target_jobs)))
        ]
        target_jobs = [os.path.join(self._root_dir, i) for i in target_jobs]
        return target_jobs, ele_list

    def _get_structure(self):
        json_file_path = os.path.join(self._target_jobs[0], "job.json")
        with open(json_file_path, "r") as f:
            job_dict = json.load(f)
        str_dict = job_dict["str"]
        ions = str_dict["ions"]
        for key, value in ions.items():
            ions[key] = Quantity(value, mol / decimeter**3)
        structure_name, pdb_file_path, psf_file_path = generate_structure(
            r0=Quantity(str_dict["r0"], angstrom),
            l0=Quantity(str_dict["l0"], angstrom),
            w0=Quantity(str_dict["w0"], angstrom),
            ls=Quantity(str_dict["ls"], angstrom),
            **ions
        )
        topology = md.io.PSFParser(psf_file_path).topology
        # Get information for ions
        ions = {}
        for key in str_dict["ions"].keys():
            ions[key] = {
                "valence": SUPPORTED_IONS[key],
                "matrix_ids": select(topology, [{"particle name": [[key]]}]),
            }
        # Cla does not contained in the json
        key = "CLA"
        ions[key] = {
            "valence": -1,
            "matrix_ids": select(topology, [{"particle name": [[key]]}]),
        }
        return structure_name, topology, ions

    def analysis(self):
        sin_matrix_ids = select(self._topology, [{"molecule type": [["SIN"]]}])
        out_dir = os.path.join(job, self._sample_prefix)
        for job in self._target_jobs[-2:-1]:
            # Get pbc matrix
            pdb_file_path = os.path.join(out_dir, "restart.pdb")
            with open(pdb_file_path, "r") as f:
                line = f.readline()
                line = f.readline()  # Second line
            x, y, z, alpha, beta, gamma = [float(i) for i in line.split()[1:7]]
            gamma = np.deg2rad(gamma)
            self._pbc_matrix = np.array(
                [
                    [x, 0, 0],
                    [y * np.cos(gamma), y * np.sin(gamma), 0],
                    [0, 0, z],
                ]
            )
            self._pbc_inv = np.linalg.inv(self._pbc_matrix)
            dcd_file_path = os.path.join(out_dir, self._sample_prefix + ".dcd")
            # dcd_file_path = os.path.join(job, self._sample_prefix, "wrapped.dcd")
            positions = md.io.DCDParser(dcd_file_path).positions
            # sin_center = self._analysis_z_center(positions[:, sin_matrix_ids, :])
            sin_center = self._analysis_z_center(positions[:, :, :])
            for key, value in self._ions.items():
                ion_positions = positions[:, value["matrix_ids"], :]
                self._analysis_flux(ion_positions, dividing_z=sin_center)
                print(key, ion_positions.shape)
            print("Finish dcd")

    def _analysis_z_center(self, positions):
        scaled_positions = np.dot(positions, self._pbc_inv)
        scaled_positions -= np.round(scaled_positions)
        positions = np.dot(scaled_positions, self._pbc_matrix)
        z_center = positions[:, :, 2].mean()
        return z_center

    def _analysis_flux(self, positions, dividing_z):
        scaled_positions = np.dot(positions, self._pbc_inv)
        diff = scaled_positions[1:] - scaled_positions[:-1]
        diff -= np.round(diff)
        num_frames = positions.shape[0]
        unwrapped_positions = np.zeros_like(positions)
        unwrapped_positions[0] = scaled_positions[0] - np.round(scaled_positions[0])
        for i in range(1, num_frames):
            unwrapped_positions[i] = unwrapped_positions[i - 1] + diff[i - 1]
        unwrapped_positions = np.dot(unwrapped_positions, self._pbc_matrix)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=[16, 9])
        for i in [1, 5, 24, 58, 60]:
            ax.plot(unwrapped_positions[:, i, 2])
        fig.tight_layout()
        plt.savefig(
            "/home/zhenyuwei/simulation_data/22-07-coulombic-blockade/code/aamd/si3n4_pore/str/test.png"
        )
        positions = self._warp_positions(positions)
        status = positions[:, :, 2] < dividing_z
        print(status.all(0))


if __name__ == "__main__":
    analyzer = CurrentAnalyzer(
        root_dir=os.path.join(
            OUT_DIR, "POT-5.00e-01molPerL/r0-5.000A-w0-50.000A-l0-50.000A-ls-50.000A"
        ),
        sample_prefix="05-sample-nvt-ele",
    )
    analyzer.analysis()

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : rdf_ion_water_analyzer.py
created time : 2023/03/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import time
import cupy as cp
import numpy as np
import mdpy as md
import matplotlib.pyplot as plt
from mdpy.utils import select
from mdpy.analyser import AnalyserResult
from mdpy.environment import *
from mdpy.unit import *


class WaterRDFAnalyzer:
    def __init__(self, ion_type, psf_file_path, dcd_file_path) -> None:
        # Read input
        self._ion_type = ion_type
        self._topology = md.io.PSFParser(psf_file_path).topology
        parser = md.io.DCDParser(dcd_file_path)
        positions = cp.array(parser.positions)
        self._pbc_matrix = cp.array(parser.pbc_matrix, CUPY_FLOAT)
        self._pbc_inv = cp.linalg.inv(self._pbc_matrix).astype(CUPY_FLOAT)
        self._ion_positions = self._get_ion_positions(positions)
        self._water_positions = self._get_water_positions(positions)

    def _get_ion_positions(self, positions):
        index = select(self._topology, [{"particle type": [[self._ion_type.upper()]]}])
        return positions[:, index, :].astype(CUPY_FLOAT)

    def _get_water_positions(self, positions):
        index = select(self._topology, [{"particle type": [["OT"]]}])
        num_waters = len(index)
        o_index = np.zeros(num_waters, NUMPY_INT)
        h1_index = np.zeros(num_waters, NUMPY_INT)
        h2_index = np.zeros(num_waters, NUMPY_INT)

        cur_mol_id = -1
        cur_array_index = -1
        for particle in self._topology.particles:
            mol_id = particle.molecule_id
            if cur_mol_id != mol_id:
                cur_mol_id = mol_id
                cur_array_index += 1
            if particle.particle_name == "OH2":
                o_index[cur_array_index] = particle.matrix_id
            elif particle.particle_name == "H1":
                h1_index[cur_array_index] = particle.matrix_id
            elif particle.particle_name == "H2":
                h2_index[cur_array_index] = particle.matrix_id

        o_mass = CUPY_FLOAT(self._topology.particles[o_index[0]].mass)
        h1_mass = CUPY_FLOAT(self._topology.particles[h1_index[0]].mass)
        h2_mass = CUPY_FLOAT(self._topology.particles[h2_index[0]].mass)
        total_mass = o_mass + h1_mass + h2_mass

        water_positions = (
            positions[:, o_index, :] * o_mass
            + positions[:, h1_index, :] * h1_mass
            + positions[:, h2_index, :] * h2_mass
        ) / total_mass

        return water_positions.astype(CUPY_FLOAT)

    def analysis(self):
        factor = Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton)
        factor = factor.convert_to(1 / angstrom**3).value
        num_frames = self._ion_positions.shape[0]
        num_ions = self._ion_positions.shape[1]
        factor *= num_ions * num_frames
        print(factor)
        dist = []
        for i in range(num_ions):
            dist_vec = self._ion_positions[:, i : i + 1, :] - self._water_positions
            scaled_vec = cp.dot(dist_vec, self._pbc_inv)
            scaled_vec -= cp.round(scaled_vec)
            dist_vec = cp.dot(scaled_vec, self._pbc_matrix)
            dist.append(cp.sqrt((dist_vec**2).sum(2)).reshape(-1))

        dist = cp.hstack(dist).astype(CUPY_FLOAT)
        hist, bin_edges = cp.histogram(dist, bins=250, range=(0.0, 12.0))
        dr = CUPY_FLOAT((bin_edges[1] - bin_edges[0]).get())
        r = bin_edges[:-1] + CUPY_FLOAT(dr * 0.5)
        hist = hist / (CUPY_FLOAT(4 * cp.pi * dr * factor) * r**2)

        plt.plot(r.get(), hist.get(), ".-")
        plt.show()

        title = "RDF between %s --- water" % (self._ion_type)
        description = {
            "rdf": "The mean value of RDF function, unit: dimesionless",
            "r": "The bin center of RDF function, unit: default_length_unit",
        }
        data = {"rdf": hist.get(), "r": r.get()}
        return AnalyserResult(title=title, description=description, data=data)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "out/analysis_out")

    ion_type = "pot"
    res_file_path = os.path.join(out_dir, "rdf-%s-water.npz" % ion_type)
    psf_file_path = "/home/zhenyuwei/Documents/22-07-coulombic-blockade/code/aamd/carbon_nanotube/simulation/hydration_layer/out/no-wall-charge/no-pore-w0-2.000A-ls-25.000A-POT-1.00e-01molPerL/pot-x-0.00A-y-0.00A-z-0.00A/str/no-pore-w0-50.000A-ls-25.000A-pot-1.00e-01molPerL-no-wall-charge.psf"
    dcd_file_path = "/home/zhenyuwei/Documents/22-07-coulombic-blockade/code/aamd/carbon_nanotube/simulation/hydration_layer/out/no-wall-charge/no-pore-w0-2.000A-ls-25.000A-POT-1.00e-01molPerL/pot-x-0.00A-y-0.00A-z-0.00A/04-sample/04-sample.dcd"
    analyzer = WaterRDFAnalyzer(
        ion_type=ion_type, dcd_file_path=dcd_file_path, psf_file_path=psf_file_path
    )
    res = analyzer.analysis()
    res.save(res_file_path)

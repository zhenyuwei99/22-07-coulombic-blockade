#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : ion_rdf_analyzer.py
created time : 2022/10/08
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import mdpy as md
import matplotlib.pyplot as plt
from mdpy.unit import *

PARTICLE_NAME_MAP = {
    "pot": "POT",
    "cla": "CLA",
    "oxygen": "OT",
    "hydrogen": "HT",
}

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "out/analysis_out")
    ion, target = "cla", "oxygen"
    res_file_path = os.path.join(out_dir, "rdf-%s-%s.npz" % (ion, target))
    density_factor = 1 if target == "oxygen" else 2

    if True:
        target_dir = "/home/zhenyuwei/Documents/22-07-coulombic-blockade/code/aamd/carbon_nanotube/simulation/hydration_layer/out/no-wall-charge/no-pore-w0-2.000A-ls-25.000A-POT-1.00e-01molPerL/pot-x-0.00A-y-0.00A-z-0.00A"
        dcd_file_path = os.path.join(target_dir, "04-sample/04-sample.dcd")
        npt_file_path = os.path.join(target_dir, "02-eq-npt/restart.pdb")
        psf_file_path = os.path.join(target_dir, "str")
        psf_file_path = os.path.join(
            psf_file_path,
            [i for i in os.listdir(psf_file_path) if i.endswith("psf")][0],
        )
        pbc_matrix = md.io.PDBParser(npt_file_path).pbc_matrix
        positions = md.io.DCDParser(dcd_file_path).positions
        topology = md.io.PSFParser(psf_file_path).topology
        trajectory = md.core.Trajectory(topology=topology)
        trajectory.set_pbc_matrix(pbc_matrix)
        trajectory.append(positions=positions)

        selection_condition_1 = [{"particle type": [[PARTICLE_NAME_MAP[ion]]]}]
        selection_condition_2 = [{"particle type": [[PARTICLE_NAME_MAP[target]]]}]
        analyser = md.analyser.RDFAnalyser(
            selection_condition_1=selection_condition_1,
            selection_condition_2=selection_condition_2,
            bulk_density=Quantity(
                1.014 * density_factor / 18, kilogram / dalton / decimeter**3
            ),
            cutoff_radius=Quantity(12, angstrom),
            num_bins=400,
        )
        res = analyser.analysis(trajectory)
        res.save(res_file_path)
    else:
        res = md.analyser.load_analyser_result(
            os.path.join(out_dir, "rdf-%s-%s.npz" % (ion, target))
        )
        res1 = md.analyser.load_analyser_result(
            os.path.join(out_dir, "rdf-%s-%s-long.npz" % (ion, target))
        )
    # Visualization
    fig, ax = plt.subplots(1, 1)
    ax.plot(res.data["bin_center"], res.data["mean"], ".-")
    plt.show()

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : parametrization.py
created time : 2022/08/15
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np
from main import check_dir
from nelder_mead import NelderMeadMinimizer


def ls_objective_fun(ls_pot, ls_cla):
    root_dir = check_dir(os.path.join(out_dir, "pot-%.2f-cla-%.2f" % (ls_pot, ls_cla)))
    result = os.popen(
        "python %s/object/ls.py %s %.2f %.2f" % (cur_dir, root_dir, ls_pot, ls_cla)
    )
    result = result.read().split("\n")
    for i in range(len(result)):
        if "Error" in result[i]:
            print(float(result[i].split()[-1]))
            return float(result[i].split()[-1])
    raise KeyError("Failed to catch result")


def ls_boundary_ratio_objective_fun(
    ls_pot, ls_cla, boundary_ratio_pot, boundary_ratio_cla
):
    root_dir = check_dir(
        os.path.join(
            out_dir,
            "pot-ls-%.2f-boundary-ratio-%.3f-cla-ls-%.2f-boundary-ratio-%.3f"
            % (ls_pot, boundary_ratio_pot, ls_cla, boundary_ratio_cla),
        )
    )
    command = "python %s/object/ls_boundary_ratio.py %s %.5f %.5f %.5f %.5f" % (
        cur_dir,
        root_dir,
        ls_pot,
        ls_cla,
        boundary_ratio_pot,
        boundary_ratio_cla,
    )
    print(command)
    result = os.popen(command)
    result = result.read().split("\n")
    for i in range(len(result)):
        if "Error" in result[i]:
            print(float(result[i].split()[-1]))
            return float(result[i].split()[-1])
    raise KeyError("Failed to catch result")


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = check_dir(os.path.join(cur_dir, "out/nelder_mead-02"))

    res_file = os.path.join(out_dir, "minimizer.npz")
    minimizer = NelderMeadMinimizer(
        objective_fun=ls_boundary_ratio_objective_fun,
        num_dimensions=4,
        result_file_path=res_file,
    )
    minimizer.initialize(
        np.array(
            [
                [10, 10, 0.01, 0.01],
                [75, 75, 0.9, 0.9],
                [10, 75, 0.01, 0.9],
                [75, 10, 0.9, 0.01],
                [10, 10, 0.9, 0.9],
            ],
        )
    )
    minimizer.minimize(error_tolerance=1e-2)


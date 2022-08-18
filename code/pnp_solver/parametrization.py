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

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = check_dir(os.path.join(cur_dir, "out/nelder_mead"))

    def objective_fun(ls_pot, ls_cla):
        result = os.popen(
            "python %s/objective.py %s %.2f %.2f" % (cur_dir, root_dir, ls_pot, ls_cla)
        )
        result = result.read().split("\n")
        for i in range(len(result)):
            if "Error" in result[i]:
                print(float(result[i].split()[-1]))
                return float(result[i].split()[-1])
        raise KeyError("Failed to catch result")

    res_file = os.path.join(root_dir, "minimizer.npz")
    minimizer = NelderMeadMinimizer(
        objective_fun=objective_fun, num_dimensions=2, result_file_path=res_file
    )
    minimizer.initialize(np.array([[10, 10], [20, 20], [15, 15]]))
    minimizer.minimize(error_tolerance=1e-2)


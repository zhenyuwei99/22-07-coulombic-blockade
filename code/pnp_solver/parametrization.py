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
import requests
import numpy as np
import scipy.optimize as optimize
from main import check_dir
from nelder_mead import NelderMeadMinimizer
from simulate_annealing import SimulateAnnealingMinimizer


def post_autodl(text):
    resp = requests.post(
        "https://www.autodl.com/api/v1/wechat/message/push",
        json={
            "token": "9380a61c5243",
            "title": "Parameterization",
            "name": "3080x6",
            "content": text,
        },
    )


def dump(*x, end="\n", newline=False):
    global operation
    global log_file
    with open(log_file, "a") as f:
        if newline:
            print("Operation %d:" % operation, *x, file=f, end=end)
            operation += 1
        else:
            print(*x, file=f, end=end)


def ls_objective_fun(args):
    global operation
    ls_pot, ls_cla = args
    root_dir = check_dir(os.path.join(out_dir, "pot-%.4f-cla-%.4f" % (ls_pot, ls_cla)))
    dump("object_fun %s" % args, end=" ", newline=True)
    result = os.popen(
        "python %s/object/ls.py %s %.4f %.4f" % (cur_dir, root_dir, ls_pot, ls_cla)
    )
    result = result.read().split("\n")
    print(result)
    for i in range(len(result)):
        if "Error" in result[i]:
            res = float(result[i].split()[-1])
            dump("get result %.3f" % res, newline=False)
            if operation % 2 == 0 and operation >= 106:
                post_autodl("object_fun %s get result %.3f" % (args, res))
            return res
    raise KeyError("Failed to catch result")


def ls_boundary_ratio_objective_fun(args):
    ls_pot, ls_cla, boundary_ratio_pot, boundary_ratio_cla = args
    dump("object_fun %s" % args, end=" ")
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
    result = os.popen(command)
    result = result.read().split("\n")
    for i in range(len(result)):
        if "Error" in result[i]:
            res = float(result[i].split()[-1])
            dump("get result %.3f" % res)
            return res
    raise KeyError("Failed to catch result")


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = check_dir(os.path.join(cur_dir, "out/ls"))

    res_file = os.path.join(out_dir, "minimizer.npz")
    log_file = os.path.join(out_dir, "minimizer.log")
    operation = 0
    if not True:
        minimizer = NelderMeadMinimizer(
            objective_fun=ls_objective_fun, num_dimensions=2, result_file_path=res_file,
        )
        minimizer.initialize(np.array([[10, 10], [75, 75], [10, 75],]))
        minimizer.minimize()
    if not True:
        minimizer = SimulateAnnealingMinimizer(
            objective_fun=ls_objective_fun,
            num_dimensions=2,
            coordinate_range=np.array([[0, 100], [0, 100]]),
            result_file_path=res_file,
            log_file_path=log_file,
        )
        minimizer.minimize(start_beta=1e4, end_beta=1e-4, decreasing_factor=0.9)
    if True:
        try:
            operation = 0
            open(log_file, "w").close()
            optimize.dual_annealing(
                ls_objective_fun,
                [[0, 100], [0, 100]],
                maxiter=50,
                callback=dump,
                x0=np.array([50, 50]),
                seed=12345,
            )
        except:
            post_autodl("Job failed")


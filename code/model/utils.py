#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : utils.py
created time : 2022/11/10
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import shutil
import numpy as np


def get_sigmoid_length(alpha):
    y = 1
    for _ in range(500):
        y = (y + 1) ** 2 / 100
    return float(-np.log(np.abs(y)) / alpha)


def reasoning_alpha(ls):
    y = 1
    for _ in range(100):
        y = (1 + y**2) / 100
    return float(-np.log(np.abs(y)) / ls)


def mkdir(dir_path):
    try:
        os.mkdir(dir_path)
    except:
        mkdir(os.path.dirname(dir_path))
        mkdir(dir_path)


def check_dir(dir_path: str, restart=False):
    if not os.path.exists(dir_path):
        mkdir(dir_path)
    if restart:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    return dir_path

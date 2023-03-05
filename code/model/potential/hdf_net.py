#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hdf_net.py
created time : 2023/03/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import numpy as np
import cupy as cp
import torch as tc
from model import *
from model.core import Net


class HydrationDistributionFunctionNet:
    def __init__(self, net_file_path: str) -> None:
        self._net: Net = tc.load(net_file_path)

    def __call__(self, dist):
        if isinstance(dist, cp.ndarray):
            dist = dist.get()
        origin_shape = dist.shape
        dist = tc.tensor(dist, device=self._net.device, dtype=TORCH_FLOAT).reshape(
            -1, 1
        )
        pred = tc.zeros_like(dist, device=tc.device("cpu"), dtype=TORCH_FLOAT)
        batch_size = 100000
        num_batches = int(np.ceil(dist.shape[0] / batch_size))
        for i in range(num_batches - 1):
            target = slice(i * batch_size, (i + 1) * batch_size)
            pred[target] = self._net(dist[target]).detach().cpu()
        target = slice(num_batches * batch_size, None)
        pred[target] = self._net(dist[target]).detach().cpu()

        pred = pred.numpy().reshape(origin_shape)
        return cp.array(pred, CUPY_FLOAT)

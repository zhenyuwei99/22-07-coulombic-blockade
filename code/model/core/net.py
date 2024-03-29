#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : net.py
created time : 2023/02/21
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import torch as tc
import torch.nn as nn
from model import *


class Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_list: list[int],
        output_dim: int,
        device=tc.device("cuda", 0),
    ) -> None:
        super(Net, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim_list = hidden_dim_list
        self._num_hidden_layers = len(hidden_dim_list)
        self._output_dim = output_dim
        self._device = device
        self._module_list = nn.ModuleList(self._get_layer_list()).to(self._device)
        self.to(self._device)

    def _get_layer_list(self):
        activate = nn.Sigmoid
        layer_list = []
        layer_list.append(
            nn.Linear(self._input_dim, self._hidden_dim_list[0], dtype=TORCH_FLOAT)
        )
        layer_list.append(activate())
        for index in range(self._num_hidden_layers - 1):
            layer_list.append(
                nn.Linear(
                    self._hidden_dim_list[index],
                    self._hidden_dim_list[index + 1],
                    dtype=TORCH_FLOAT,
                )
            )
            layer_list.append(activate())
        layer_list.append(
            nn.Linear(self._hidden_dim_list[-1], self._output_dim, dtype=TORCH_FLOAT)
        )
        return layer_list

    def forward(self, x: tc.Tensor):
        y = x
        for module in self._module_list:
            y = module(y)
        return y

    @property
    def device(self):
        return self._device

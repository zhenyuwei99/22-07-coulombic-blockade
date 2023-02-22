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


class Net(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim_list: list[int], output_dim: int, device
    ) -> None:
        super(Net, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim_list = hidden_dim_list
        self._num_hidden_layers = len(hidden_dim_list)
        self._device = device
        self._output_dim = output_dim
        self._layer_list = self._get_layer_list()

    def _get_layer_list(self):
        layer_list = []
        layer_list.append(
            nn.Linear(self._input_dim, self._hidden_dim_list[0], device=self._device)
        )
        layer_list.append(nn.Sigmoid())
        for index in range(1, self._num_hidden_layers - 1):
            layer_list.append(
                nn.Linear(
                    self._hidden_dim_list[index],
                    self._hidden_dim_list[index + 1],
                    device=self._device,
                )
            )
            layer_list.append(nn.Sigmoid())
        layer_list.append(
            nn.Linear(self._hidden_dim_list[-1], self._output_dim, device=self._device)
        )
        return layer_list

    def forward(self, x: tc.Tensor):
        y = x
        for layer in self._layer_list:
            y = layer(y)
        y.requires_grad_(True)
        return y

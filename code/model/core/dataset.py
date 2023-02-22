#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : dataset.py
created time : 2023/02/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import torch as tc
from model import *
from model.exceptions import *


class DataSet:
    def __init__(self, device=tc.device("cuda", index=0)) -> None:
        self._device = device
        # Attribute
        self._sample = tc.tensor([], requires_grad=True, device=self._device)
        self._num_samples = 0
        self._label = {}
        self._coefficient_fun = {}

    def add_samples(self, *coordinates):
        sample = []
        for coordinate in coordinates:
            sample.append(coordinate.reshape(-1, 1))
        sample = tc.hstack(sample).to(self._device)
        sample.requires_grad_(True)
        if self._num_samples == 0:
            self._sample = sample
        else:
            self._sample = tc.vstack([self._sample, sample])
        self._num_samples = self._sample.shape[0]

    def add_labels(self, name, **label_data):
        if name in self._label.keys():
            for key, val in label_data.items():
                dtype = TORCH_INT if key == "index" else TORCH_FLOAT
                val = val.type(dtype).to(self._device)
                self._label[name][key] = tc.vstack([self._label[name][key], val])
        else:
            cur_label = {}
            for key, val in label_data.items():
                dtype = TORCH_INT if key == "index" else TORCH_FLOAT
                cur_label[key] = val.type(dtype).to(self._device)
            self._label[name] = cur_label

    def add_coefficient_fun(self, name, fun):
        self._coefficient_fun[name] = fun

    def get_coefficient(self, name):
        if not name in self._coefficient_fun.keys():
            raise DataSetPoorDefinedError(
                "Coefficient function %s has not defined." % name
            )
        return self._coefficient_fun[name](self._sample)

    def get_label(self, name):
        if not name in self._label.keys():
            raise DataSetPoorDefinedError("Label %s has not defined." % name)
        pass

    def to(self, device):
        self._device = device
        self._sample = self._sample.to(self._device)
        for key_label, val in self._label.items():
            for key_data, tensor in val.items():
                self._label[key_label][key_data] = tensor.to(self._device)

    @property
    def x(self):
        return self._sample

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def label(self):
        return self._label

    @property
    def device(self):
        return self._device

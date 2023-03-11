#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_dataset.py
created time : 2023/02/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import pytest
import torch as tc
import numpy as np
from torch.autograd import grad
from model.core import DataSet
from model.exceptions import *


class TestDataSet:
    def setup(self):
        self.dataset = DataSet()

    def teardown(self):
        del self.dataset

    def test_attribute(self):
        assert self.dataset.num_samples == 0
        assert self.dataset.label == {}
        assert self.dataset.x.requires_grad == True

    def test_exceptions(self):
        with pytest.raises(DataSetPoorDefinedError):
            self.dataset.get_coefficient("phi")
        with pytest.raises(DataSetPoorDefinedError):
            self.dataset.get_label("inner")

    def test_add_samples(self):
        h = 0.5
        r, z = tc.meshgrid(
            tc.arange(0, 10 + h, h), tc.arange(-10, 10 + h, h), indexing="ij"
        )
        self.dataset.add_samples(r, z)
        assert self.dataset.num_samples == np.prod(r.shape)
        assert self.dataset.x.device == self.dataset.device
        assert self.dataset.x.requires_grad == True

        r, z = tc.meshgrid(
            tc.arange(0, 10 + h, h), tc.arange(-10, 10 + h, h), indexing="ij"
        )
        self.dataset.add_samples(r, z)
        assert self.dataset.num_samples == np.prod(r.shape) * 2
        assert self.dataset.x.device == self.dataset.device
        assert self.dataset.x.requires_grad == True

    def test_add_labels(self):
        h = 0.5
        r, z = tc.meshgrid(
            tc.arange(0, 10 + h, h), tc.arange(-10, 10 + h, h), indexing="ij"
        )
        self.dataset.add_samples(r, z)
        field = tc.zeros_like(r)
        field[1:, :-1] = 1
        index = tc.argwhere(field == 1)
        index = (index[:, 0] * r.shape[1] + index[:, 1])[:, None]
        self.dataset.add_labels(name="inner", index=index, value=index)
        assert self.dataset.label["inner"]["value"].device == self.dataset.device

        index = tc.argwhere(field != 1)
        index = (index[:, 0] * r.shape[1] + index[:, 1])[:, None]
        self.dataset.add_labels(name="inner", index=index, value=index)

        assert self.dataset.label["inner"]["value"].shape[0] == np.prod(r.shape)
        assert self.dataset.label["inner"]["value"].device == self.dataset.device

    def test_coefficient(self):
        h = 0.5
        r, z = tc.meshgrid(
            tc.arange(0, 10 + h, h), tc.arange(-10, 10 + h, h), indexing="ij"
        )
        self.dataset.add_samples(r, z)

        self.dataset.add_coefficient_fun("a", lambda x: x[:, 0] + 2 * x[:, 1])
        a = self.dataset.get_coefficient("a")
        dadr = grad(a.sum(), self.dataset.x)[0]

        assert tc.allclose(dadr[:, 0], tc.ones_like(dadr[:, 0]))
        assert tc.allclose(dadr[:, 1], 2 * tc.ones_like(dadr[:, 1]))
        assert a.device == self.dataset.device

    def test_to(self):
        h = 0.5
        r, z = tc.meshgrid(
            tc.arange(0, 10 + h, h), tc.arange(-10, 10 + h, h), indexing="ij"
        )
        self.dataset.add_samples(r, z)
        field = tc.zeros_like(r)
        field[1:, :-1] = 1
        index = tc.argwhere(field != 1)
        index = (index[:, 0] * r.shape[1] + index[:, 1])[:, None]
        self.dataset.add_labels(name="inner", index=index, value=index)

        device = tc.device("cpu")
        assert not self.dataset.x.device == device

        self.dataset.to(device)
        assert self.dataset.x.device == device

        r, z = tc.meshgrid(
            tc.arange(0, 10 + h, h), tc.arange(-10, 10 + h, h), indexing="ij"
        )
        self.dataset.add_samples(r, z)
        assert self.dataset.x.device == device

        index = tc.argwhere(field == 1)
        index = (index[:, 0] * r.shape[1] + index[:, 1])[:, None]
        self.dataset.add_labels(name="inner", index=index, value=index)


if __name__ == "__main__":
    test = TestDataSet()
    test.setup()
    test.test_coefficient()
    test.teardown()

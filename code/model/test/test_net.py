#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_net.py
created time : 2023/02/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import pytest
import torch as tc
from model.core import Net


class TestNet:
    def setup(self):
        self.net = Net(2, [32, 32, 32], 1)

    def teardown(self):
        del self.net

    def test_attributes(self):
        device = tc.device("cpu")
        for params in self.net.parameters():
            assert params.requires_grad
            assert params.is_leaf
            assert params.device == device

    def test_exceptions(self):
        pass

    def test_to(self):
        device = tc.device("cuda", index=0)
        self.net.to(device)
        for params in self.net.parameters():
            assert params.device == device

    def test_forward(self):
        device = tc.device("cpu")
        x = tc.rand(10, 2, requires_grad=True)
        y = self.net(x)
        assert x.device == device
        assert y.device == device

        device = tc.device("cuda", 0)
        self.net.to(device)

        with pytest.raises(RuntimeError):
            y = self.net(x)
        x = x.to(device)
        y = self.net(x)
        assert x.device == device
        assert y.device == device

        assert y.requires_grad


if __name__ == "__main__":
    test = TestNet()
    test.setup()
    test.test_forward()
    test.teardown()

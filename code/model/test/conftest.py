#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : conftest.py
created time : 2021/09/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest

test_order = ["grid", "grid_parser", "grid_writer"]


def pytest_collection_modifyitems(items):
    current_index = 0
    for test in test_order:
        indexes = []
        for id, item in enumerate(items):
            if "test_" + test + ".py" in item.nodeid:
                indexes.append(id)
        for id, index in enumerate(indexes):
            items[current_index + id], items[index] = (
                items[index],
                items[current_index + id],
            )
        current_index += len(indexes)

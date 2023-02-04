#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : exceptions.py
created time : 2023/02/04
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""


class ArrayDimError(Exception):
    """This error occurs when:
    - The dimension of argument does not meet the requirement

    Used in:
    - model.core.grid
    """

    pass


class FileFormatError(Exception):
    """This error occurs when:
    - file suffix or prefix appears in an unexpected way

    Used in:
    - model.core.grid_writer
    - model.core.grid_parser
    """

    pass


class GridPoorDefinedError(Exception):
    """This error occurs when:
    - Grid's requirement has not been satisfied

    Used in:
    - model.core.grid
    """

    pass

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hyd_prl.py
created time : 2023/03/07
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import numpy as np
from mdpy.unit import *


PARAM = {
    "na": {"r": [2.9, 5.1, 7.5], "u": [-1.51, -0.72, -0.30]},
    "k": {"r": [3.3, 5.6, 7.8], "u": [-1.15, -0.61, -0.27]},
    "cl": {"r": [3.1, 4.9, 7.1], "u": [-1.73, -0.68, -0.31]},
}


class HydrationPotentialPRL:
    def __init__(self, ion_type: str) -> None:
        self._ion_type = ion_type
        self._r = PARAM[self._ion_type]["r"]
        self._u = PARAM[self._ion_type]["u"]
        self._u = (
            Quantity(self._u, elementary_charge * volt)
            .convert_to(default_energy_unit)
            .value
        )

    def evaluate(self, rp):
        res = 0
        for r, u in zip(self._r, self._u):
            if rp >= r:
                continue
            f = 1 - np.sqrt(1 - (rp / r) ** 2)
            res += (f - 1) * u
        return res


if __name__ == "__main__":
    potential = HydrationPotentialPRL("k")
    potential.evaluate(5.0)

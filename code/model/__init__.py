__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__email__ = "zhenyuwei99@gmail.com"
__copyright__ = "Copyright 2021-2021, Southeast University and Zhenyu Wei"
__license__ = "GPLv3"

import torch
import numba as nb
import cupy as cp
import numpy as np
from mdpy.unit import *
from mdpy.utils import check_quantity_value

CC_BOND_LENGTH = 1.418
DIFFUSION_UNIT = default_length_unit**2 / default_time_unit
MOBILITY_UNIT = default_length_unit**2 / default_time_unit / default_voltage_unit
VAL_UNIT = default_charge_unit
ION_DICT = {
    "k": {
        "d": Quantity(1.96e-9, meter**2 / second),
        "mu": Quantity(7.62e-8, meter**2 / second / volt),
        "val": Quantity(1, elementary_charge),
        "name": "pot",
    },
    "na": {
        "d": Quantity(1.33e-9, meter**2 / second),
        "mu": Quantity(5.19e-8, meter**2 / second / volt),
        "val": Quantity(1, elementary_charge),
        "name": "sod",
    },
    "ca": {
        "d": Quantity(0.79e-9, meter**2 / second),
        "mu": Quantity(6.16e-8, meter**2 / second / volt),
        "val": Quantity(2, elementary_charge),
        "name": "carbon",
    },
    "cl": {
        "d": Quantity(2.03e-9, meter**2 / second),
        "mu": Quantity(7.91e-8, meter**2 / second / volt),
        "val": Quantity(-1, elementary_charge),
        "name": "cla",
    },
}

convert_factor = 2 ** (5 / 6)
VDW_DICT = {
    "c": {
        "sigma": Quantity(1.992 * convert_factor, angstrom),
        "epsilon": Quantity(0.070, kilocalorie_permol),
    },
    "k": {
        "sigma": Quantity(1.764 * convert_factor, angstrom),
        "epsilon": Quantity(0.087, kilocalorie_permol),
    },
    "na": {
        "sigma": Quantity(1.411 * convert_factor, angstrom),
        "epsilon": Quantity(0.047, kilocalorie_permol),
    },
    "ca": {
        "sigma": Quantity(1.367 * convert_factor, angstrom),
        "epsilon": Quantity(0.120, kilocalorie_permol),
    },
    "cl": {
        "sigma": Quantity(2.270 * convert_factor, angstrom),
        "epsilon": Quantity(0.150, kilocalorie_permol),
    },
}
NP_DENSITY_UPPER_THRESHOLD = (
    (Quantity(4, mol / decimeter**3) * NA)
    .convert_to(1 / default_length_unit**3)
    .value
)
NP_DENSITY_LOWER_THRESHOLD = (
    (Quantity(0.001, mol / decimeter**3) * NA)
    .convert_to(1 / default_length_unit**3)
    .value
)

PRECISION = "single"

CUPY_BIT = cp.uint32
NUMBA_BIT = nb.uint32
NUMPY_BIT = np.uint32
UNIT_FLOAT = np.float128
if PRECISION == "single":
    CUPY_FLOAT = cp.float32
    NUMBA_FLOAT = nb.float32
    NUMPY_FLOAT = np.float32
    TORCH_FLOAT = torch.float32
    CUPY_INT = cp.int32
    NUMBA_INT = nb.int32
    NUMPY_INT = np.int32
    TORCH_INT = torch.int32
    CUPY_UINT = cp.uint32
    NUMBA_UINT = nb.uint32
    NUMPY_UINT = np.uint32
elif PRECISION == "double":
    CUPY_FLOAT = cp.float64
    NUMBA_FLOAT = nb.float64
    NUMPY_FLOAT = np.float64
    TORCH_FLOAT = torch.float64
    CUPY_INT = cp.int64
    NUMBA_INT = nb.int64
    NUMPY_INT = np.int64
    TORCH_INT = torch.int64
    CUPY_UINT = cp.uint64
    NUMBA_UINT = nb.uint64
    NUMPY_UINT = np.uint64

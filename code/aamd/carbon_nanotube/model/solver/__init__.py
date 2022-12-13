__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__email__ = "zhenyuwei99@gmail.com"
__copyright__ = "Copyright 2021-2021, Southeast University and Zhenyu Wei"
__license__ = "GPLv3"

from mdpy.unit import *

CC_BOND_LENGTH = 1.418
DIFFUSION_UNIT = default_length_unit**2 / default_time_unit
VAL_UNIT = default_charge_unit
ION_DICT = {
    "k": {
        "d": Quantity(1.96e-9, meter**2 / second),
        "val": Quantity(1, elementary_charge),
    },
    "na": {
        "d": Quantity(1.33e-9, meter**2 / second),
        "val": Quantity(1, elementary_charge),
    },
    "ca": {
        "d": Quantity(0.79e-9, meter**2 / second),
        "val": Quantity(2, elementary_charge),
    },
    "cl": {
        "d": Quantity(2.03e-9, meter**2 / second),
        "val": Quantity(-1, elementary_charge),
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
    (Quantity(7, mol / decimeter**3) * NA)
    .convert_to(1 / default_length_unit**3)
    .value
)
NP_DENSITY_LOWER_THRESHOLD = (
    (Quantity(0.001, mol / decimeter**3) * NA)
    .convert_to(1 / default_length_unit**3)
    .value
)

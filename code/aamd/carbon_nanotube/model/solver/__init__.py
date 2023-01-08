__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__email__ = "zhenyuwei99@gmail.com"
__copyright__ = "Copyright 2021-2021, Southeast University and Zhenyu Wei"
__license__ = "GPLv3"

from mdpy.unit import *
from mdpy.utils import check_quantity_value

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
    (Quantity(4, mol / decimeter**3) * NA)
    .convert_to(1 / default_length_unit**3)
    .value
)
NP_DENSITY_LOWER_THRESHOLD = (
    (Quantity(0.001, mol / decimeter**3) * NA)
    .convert_to(1 / default_length_unit**3)
    .value
)


def generate_name(ion_types, r0, z0, zs, w0, rho, grid_width, voltage, is_pnp):
    """
    - `ion_types` (list[str]): list of ion types
    - `r0` (Quantity or float): The radius of pore, if float in unit of angstrom
    - `z0` (Quantity or float): The half-thickness of pore, if float in unit of angstrom
    - `zs` (Quantity or float): The thickness of solvent above the pore, if float in unit of angstrom
    - `w0` (Quantity or float): The width of pore, if float in unit of angstrom
    - `grid_width` (Quantity or float): The grid width, if float in unit of angstrom
    - `rho` (Quantity or float): The bulk density of solution, if float in unit of mol/L
    - `voltage` (Quantity or float): The external voltage, if float in unit of V
    - `is_pnp` (bool): Bool value for including external potential or not. True: not include
    """
    name = ["pnp" if is_pnp else "mpnp"]
    name.extend(ion_types)
    name.append("%.2fmolPerL" % check_quantity_value(rho, mol / decimeter**3))
    name.append("r0-%.2fA" % check_quantity_value(r0, angstrom))
    name.append("z0-%.2fA" % check_quantity_value(z0, angstrom))
    name.append("zs-%.2fA" % check_quantity_value(zs, angstrom))
    name.append("w0-%.2fA" % check_quantity_value(w0, angstrom))
    name.append("h-%.2fA" % check_quantity_value(grid_width, angstrom))
    name.append("%.2fV" % check_quantity_value(voltage, volt))
    name = "-".join(name)
    return name

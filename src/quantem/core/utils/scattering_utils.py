import math


def electron_wavelength_angstrom(E_eV):
    """returns relativistic electron wavelength in Angstroms."""
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458.0
    h = 6.62607e-34

    lam = (
        h / math.sqrt(2 * m * e * E_eV) / math.sqrt(1 + e * E_eV / 2 / m / c**2) * 1e10
    )
    return lam

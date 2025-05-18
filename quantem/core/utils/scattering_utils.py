import math


def electron_wavelength_angstrom(energy: float) -> float:
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458.0
    h = 6.62607e-34

    lam = (
        h
        / math.sqrt(2.0 * m * e * energy)
        / math.sqrt(1 + e * energy / 2.0 / m / c**2)
        * 1e10
    )
    return lam

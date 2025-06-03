import math


def electron_wavelength_angstrom(E_eV):
    """returns relativistic electron wavelength in Angstroms."""
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458.0
    h = 6.62607e-34

    lam = h / math.sqrt(2 * m * e * E_eV) / math.sqrt(1 + e * E_eV / 2 / m / c**2) * 1e10
    return lam


def angular_to_reciprocal_sampling(angular_sampling, E_eV):
    """ """
    if isinstance(angular_sampling, float):
        squeeze = True
        angular_sampling = (angular_sampling,)
    else:
        squeeze = False

    wavelength = electron_wavelength_angstrom(E_eV)
    reciprocal_sampling = tuple(da * 1e-3 / wavelength for da in angular_sampling)

    return reciprocal_sampling[0] if squeeze else reciprocal_sampling


def reciprocal_to_angular_sampling(reciprocal_sampling, E_eV):
    """ """
    if isinstance(reciprocal_sampling, float):
        squeeze = True
        reciprocal_sampling = (reciprocal_sampling,)
    else:
        squeeze = False

    wavelength = electron_wavelength_angstrom(E_eV)
    angular_sampling = tuple(dk * 1e3 * wavelength for dk in reciprocal_sampling)

    return angular_sampling[0] if squeeze else angular_sampling

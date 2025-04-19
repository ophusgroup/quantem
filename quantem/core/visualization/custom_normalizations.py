from dataclasses import dataclass

import matplotlib as mpl
import numpy as np

"""
Custom normalization based on astropy's visualization routines.

Original implementation:
https://github.com/astropy/astropy/blob/main/astropy/visualization/mpl_normalize.py

Licensed under a 3-clause BSD style license.
"""


class BaseInterval:
    """
    Base class for the interval classes, which when called with an array of values,
    return an interval clipped to the [0:1] range.
    """

    def __call__(self, values):
        """
        Transform values using this interval.

        Parameters
        ----------
        values : array-like
            The input values.

        Returns
        -------
        result : ndarray
            The transformed values.
        """
        vmin, vmax = self.get_limits(values)

        # subtract vmin
        values = np.subtract(values, vmin)

        # divide by interval
        if (vmax - vmin) != 0.0:
            np.true_divide(values, vmax - vmin, out=values)

        # clip to [0:1]
        np.clip(values, 0.0, 1.0, out=values)
        return values

    def inverse(self, values):
        """
        Pseudo-inverse interval transform. Note this does not recover
        the original range due to clipping. Used for colorbars.

        Parameters
        ----------
        values : array-like
            The input values.

        Returns
        -------
        result : ndarray
            The transformed values.
        """
        vmin, vmax = self.get_limits(values)

        values = np.multiply(values, vmax - vmin)
        np.add(values, vmin, out=values)
        return values


@dataclass
class ManualInterval(BaseInterval):
    """
    Interval based on user-specified values.

    Parameters
    ----------
    vmin : float, optional
        The minimum value in the scaling.
    vmax : float, optional
        The maximum value in the scaling.
    """

    vmin: float | None = None
    vmax: float | None = None

    def get_limits(self, values):
        # Avoid overhead of preparing array if both limits have been specified
        # manually, for performance.

        if self.vmin is not None and self.vmax is not None:
            return self.vmin, self.vmax

        # Make sure values is a Numpy array
        values = np.asarray(values).ravel()

        # Filter out invalid values (inf, nan)
        values = values[np.isfinite(values)]
        vmin = np.min(values) if self.vmin is None else self.vmin
        vmax = np.max(values) if self.vmax is None else self.vmax

        return vmin, vmax


@dataclass
class CenteredInterval(BaseInterval):
    """
    Centered interval based on user-specified halfrange.

    Parameters
    ----------
    vcenter : float
        The center value in the scaling.
    half_range : float, optional
        The half range in the scaling.
    """

    vcenter: float = 0.0
    half_range: float | None = None

    def get_limits(self, values):
        if self.half_range is not None:
            return self.vcenter - self.half_range, self.vcenter + self.half_range

        values = np.asarray(values).ravel()
        values = values[np.isfinite(values)]
        vmin = np.min(values)
        vmax = np.max(values)

        half_range = np.maximum(
            np.abs(vmin - self.vcenter), np.abs(vmax - self.vcenter)
        )

        return self.vcenter - half_range, self.vcenter + half_range


@dataclass
class QuantileInterval(BaseInterval):
    """
    Interval based on a keeping a specified fraction of pixels.

    Parameters
    ----------
    lower_quantile : float or None
        The lower quantile below which to ignore pixels. If None, then
        defaults to 0.
    upper_quantile : float or None
        The upper quantile above which to ignore pixels. If None, then
        defaults to 1.
    """

    lower_quantile: float = 0.02
    upper_quantile: float = 0.98

    def get_limits(self, values):
        # Make sure values is a Numpy array
        values = np.asarray(values).ravel()

        # Filter out invalid values (inf, nan)
        values = values[np.isfinite(values)]
        vmin, vmax = np.quantile(values, (self.lower_quantile, self.upper_quantile))

        return vmin, vmax


@dataclass
class LinearStretch:
    r"""
    A linear stretch with a slope and offset.

    The stretch is given by:

    .. math::
        y = slope * x + intercept

    Parameters
    ----------
    slope : float, optional
        The ``slope`` parameter used in the above formula.  Default is 1.
    intercept : float, optional
        The ``intercept`` parameter used in the above formula.  Default is 0.
    """

    slope: float = 1.0
    intercept: float = 0.0

    def __call__(self, values, copy=True):
        if self.slope == 1.0 and self.intercept == 0.0:
            return values

        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        if self.slope != 1.0:
            np.multiply(values, self.slope, out=values)
        if self.intercept != 0.0:
            np.add(values, self.intercept, out=values)
        return values

    @property
    def inverse(self):
        return LinearStretch(1 / self.slope, -self.intercept / self.slope)


@dataclass
class PowerLawStretch:
    r"""
    A power stretch.

    The stretch is given by:

    .. math::
        y = x^{power}

    Parameters
    ----------
    power : float
        The power index (see the above formula).  ``power`` must be greater
        than 0.
    """

    power: float = 1.0

    def __post_init__(self):
        if self.power <= 0.0:
            raise ValueError("power must be > 0")

    def __call__(self, values, copy=True):
        if self.power == 1.0:
            return values

        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        np.power(values, self.power, out=values)
        return values

    @property
    def inverse(self):
        return PowerLawStretch(1.0 / self.power)


@dataclass
class LogarithmicStretch:
    r"""
    A logarithmic stretch.

    The stretch is given by:

    .. math::
        y = \frac{\log{(a x + 1)}}{\log{(a + 1)}}

    Parameters
    ----------
    a : float
        The ``a`` parameter used in the above formula.  ``a`` must be
        greater than 0.  Default is 1000.
    """

    a: float = 1000.0

    def __post_init__(self):
        if self.a <= 0:
            raise ValueError("a must be > 0")

    def __call__(self, values, copy=True):
        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        np.multiply(values, self.a, out=values)
        np.add(values, 1.0, out=values)
        np.log(values, out=values)
        np.true_divide(values, np.log(self.a + 1.0), out=values)
        return values

    @property
    def inverse(self):
        return InverseLogarithmicStretch(self.a)


@dataclass
class InverseLogarithmicStretch:
    r"""
    Inverse transformation for `LogarithmicStretch`.

    The stretch is given by:

    .. math::
        y = \frac{e^{y \log{a + 1}} - 1}{a} \\
        y = \frac{e^{y} (a + 1) - 1}{a}

    Parameters
    ----------
    a : float, optional
        The ``a`` parameter used in the above formula.  ``a`` must be
        greater than 0.  Default is 1000.
    """

    a: float = 1000.0

    def __post_init__(self):
        if self.a <= 0:
            raise ValueError("a must be > 0")

    def __call__(self, values, copy=True):
        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        np.multiply(values, np.log(self.a + 1.0), out=values)
        np.exp(values, out=values)
        np.subtract(values, 1.0, out=values)
        np.true_divide(values, self.a, out=values)
        return values

    @property
    def inverse(self):
        return LogarithmicStretch(self.a)


@dataclass
class InverseHyperbolicSineStretch:
    r"""
    An asinh stretch.

    The stretch is given by:

    .. math::
        y = \frac{{\rm asinh}(x / a)}{{\rm asinh}(1 / a)}.

    Parameters
    ----------
    a : float, optional
        The ``a`` parameter used in the above formula. The value of this
        parameter is where the asinh curve transitions from linear to
        logarithmic behavior, expressed as a fraction of the normalized
        image. The stretch becomes more linear as the ``a`` value is
        increased. ``a`` must be greater than 0. Default is 0.1.
    """

    a: float = 0.1

    def __post_init__(self):
        if self.a <= 0:
            raise ValueError("a must be > 0")

    def __call__(self, values, copy=True):
        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        # map to [-1,1]
        np.multiply(values, 2.0, out=values)
        np.subtract(values, 1.0, out=values)

        np.true_divide(values, self.a, out=values)
        np.arcsinh(values, out=values)

        # map from [-1,1]
        np.true_divide(values, np.arcsinh(1.0 / self.a) * 2.0, out=values)
        np.add(values, 0.5, out=values)
        return values

    @property
    def inverse(self):
        return HyperbolicSineStretch(1.0 / np.arcsinh(1.0 / self.a))


@dataclass
class HyperbolicSineStretch:
    r"""
    A sinh stretch.

    The stretch is given by:

    .. math::
        y = \frac{{\rm sinh}(x / a)}{{\rm sinh}(1 / a)}

    Parameters
    ----------
    a : float, optional
        The ``a`` parameter used in the above formula. The stretch
        becomes more linear as the ``a`` value is increased. ``a`` must
        be greater than 0. Default is 1/3.
    """

    a: float = 1.0 / 3.0

    def __post_init__(self):
        if self.a <= 0:
            raise ValueError("a must be > 0")

    def __call__(self, values, copy=True):
        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)

        # map to [-1,1]
        np.subtract(values, 0.5, out=values)
        np.multiply(values, 2.0, out=values)

        np.true_divide(values, self.a, out=values)
        np.sinh(values, out=values)

        # map from [-1,1]
        np.true_divide(values, np.sinh(1.0 / self.a) * 2.0, out=values)
        np.add(values, 0.5, out=values)
        return values

    @property
    def inverse(self):
        return InverseHyperbolicSineStretch(1.0 / np.sinh(1.0 / self.a))


class CustomNormalization(mpl.colors.Normalize):
    """ """

    def __init__(
        self,
        interval_type: str = "quantile",
        stretch_type: str = "linear",
        *,
        data: np.ndarray | None = None,
        lower_quantile: float = 0.02,
        upper_quantile: float = 0.98,
        vmin: float | None = None,
        vmax: float | None = None,
        vcenter: float = 0.0,
        half_range: float | None = None,
        power: float = 1.0,
        logarithmic_index: float = 1000.0,
        asinh_linear_range: float = 0.1,
    ):
        """ """
        super().__init__(vmin=vmin, vmax=vmax, clip=False)
        if interval_type == "quantile":
            self.interval = QuantileInterval(
                lower_quantile=lower_quantile, upper_quantile=upper_quantile
            )
        elif interval_type == "manual":
            self.interval = ManualInterval(vmin=vmin, vmax=vmax)
        elif interval_type == "centered":
            self.interval = CenteredInterval(
                vcenter=vcenter,
                half_range=half_range,
            )
        else:
            raise ValueError("unrecognized interval_type.")

        if stretch_type == "power" or power != 1.0:
            self.stretch = PowerLawStretch(power)
        elif stretch_type == "linear":
            self.stretch = LinearStretch()
        elif stretch_type == "logarithmic":
            self.stretch = LogarithmicStretch(logarithmic_index)
        elif stretch_type == "asinh":
            self.stretch = InverseHyperbolicSineStretch(asinh_linear_range)
        else:
            raise ValueError("unrecognized stretch_type.")

        self.vmin = vmin
        self.vmax = vmax

        if data is not None:
            self._set_limits(data)

    def _set_limits(self, data):
        """ """
        self.vmin, self.vmax = self.interval.get_limits(data)
        self.interval = ManualInterval(
            self.vmin, self.vmax
        )  # set explicitly with ManualInterval
        return None

    def __call__(self, values):
        values = self.interval(values)
        self.stretch(values, copy=False)
        return np.ma.masked_invalid(values)

    def inverse(self, values):
        values = self.stretch.inverse(values)
        values = self.interval.inverse(values)
        return values

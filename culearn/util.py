import cProfile
import functools
import marshmallow_dataclass as mm
import numpy as np
import pandas as pd
import os
import pstats
import scipy
import time
import warnings

from datetime import datetime, timedelta
from itertools import combinations
from kneed import KneeLocator
from multiprocessing.pool import ThreadPool
from sklearn.linear_model import LinearRegression
from statsmodels.api import tsa
from typing import *
from workalendar.core import CoreCalendar
from workalendar.registry import registry as calendar_registry


def applymap(df: pd.DataFrame, f: Callable[[Any], Any]):
    """"Preserves backward compatibility by applying either map or applymap function to data frame."""
    try:
        return df.map(f)
    except:
        return df.applymap(f)


def tupleclass(c):
    """Class decorator that wraps around dataclasses and marshmallow_dataclass to present class as tuple."""
    return mm.dataclass(c, repr=True, eq=True, order=True, frozen=True, unsafe_hash=True)


def ignore_warnings(f: Callable) -> Callable:
    """Function decorator that ignores function warnings."""

    @functools.wraps(f)
    def decorate(*args, **kwargs):
        warn = warnings.warn
        warnings.warn = lambda *a, **b: None
        np_settings = np.seterr(all='ignore')
        result = f(*args, **kwargs)
        np.seterr(**np_settings)
        warnings.warn = warn
        return result

    return decorate


def profile(f: str, n: int = 50, path: str = 'profiling_results'):
    """
    Runs profiling and prints results.

    :param f: Function call.
    :param n: Top n results to print.
    :param path: Path to file that with be used to temporary store the profiling results while executing the function.
    :return: None
    """
    cProfile.run(f, path)
    stats = pstats.Stats(path)
    stats.sort_stats('cumtime')
    stats.print_stats(n)
    os.remove(path)


def parallel(f: Callable, param: Iterable):
    """Runs the function in parallel for each set of parameters."""
    with ThreadPool() as pool:
        return pool.map(lambda p: f(p), param)


@tupleclass
class Point2D:
    x: Union[int, float] = 0
    y: Union[int, float] = 0


class File:
    @staticmethod
    def make_dir(*directory: str) -> bool:
        """Creates directory if it doesn't exist and returns true, or returns false if directory does exist."""
        _dir = os.path.join(*directory)
        if os.path.exists(_dir):
            return False

        os.makedirs(_dir)
        return True

    @staticmethod
    def paths(*directory: str, recursive=False) -> Sequence[str]:
        """Returns file paths for all files from the specified directory and all subdirectories if recursive = True."""
        files = []
        for _directory, _, _files in os.walk(str(os.path.join(*directory))):
            files.extend([os.path.join(_directory, file) for file in _files])
            if not recursive:
                break
        return files


class Time:
    @staticmethod
    def now():
        """Returns the current unix time."""
        return time.time()

    @staticmethod
    def py(pd_time) -> datetime:
        """Transforms pandas to python datetime."""
        return pd.Timestamp(pd_time).to_pydatetime()

    @staticmethod
    def unix(seconds: float) -> datetime:
        """Transforms unix time to datetime."""
        return datetime(1970, 1, 1) + timedelta(seconds=seconds)

    @staticmethod
    def calendar(iso_code: str) -> CoreCalendar:
        """Returns calendar for ISO code."""
        return calendar_registry.get(iso_code)()

    @staticmethod
    def calendars() -> Mapping[str, CoreCalendar]:
        """Returns ISO codes mapped to calendars."""
        return calendar_registry.get_calendars(include_subregions=True)


class Math:
    @staticmethod
    def qcf(p, n, mean, stdev, k3=None, k4=None):
        """
        Calculates the quantile value based on Cornish–Fisher expansion.

        :param p: Quantile probability.
        :param n: Number of samples.
        :param mean: Mean value.
        :param stdev: Standard deviation.
        :param k3: 3rd cumulant (standardized skewness).
        :param k4: 4rd cumulant (standardized excess kurtosis).
        :return: Quantile value.
        """

        def qcf3(q):
            return q + 1 / np.sqrt(n) * (((q ** 2 - 1) * k3) / 6)

        def qcf4(q):
            return qcf3(q) + 1 / n * ((((q ** 3 - 3 * q) * k4) / 24) - ((((2 * q ** 3) - 5 * q) * k3 ** 2) / 36))

        def quantile():
            q = scipy.stats.norm.ppf(p)
            if k3 is None:
                return q
            if k4 is None:
                return qcf3(q)
            return qcf4(q)

        return mean + stdev * quantile()

    @staticmethod
    def pinball(p, x, q):
        """
        Calculates the Pinball loss for predicted quantile.

        :param p: Quantile probability.
        :param x: True value.
        :param q: Predicted quantile.
        :return: Pinball loss.
        """
        return (1 - p) * (q - x) * (x < q) + p * (x - q) * (x >= q)

    @staticmethod
    def winkler(p, x, l, u):
        """
        Calculates the Winkler loss for prediction interval.

        :param p: Interval probability.
        :param x: True value.
        :param l: Lower bound of prediction interval.
        :param u: Upper bound of prediction interval.
        :return: Winkler loss.
        """
        a = 1 - p
        width = u - l
        penal = 2 * ((l - x) / a) * (x < l) + 2 * (x - u) / a * (x > u)
        return width + penal

    @staticmethod
    def subsets(x: set) -> list:
        """Returns all non-empty subsets of x."""
        s = [*x]
        for k in range(2, len(x) + 1):
            s.extend(combinations(x, k))
        return s

    @staticmethod
    def knee(curve: Sequence[float],
             shape: Literal['convex', 'concave'] = 'convex',
             direction: Literal['decreasing', 'increasing'] = 'decreasing'):
        """Returns the knee of the curve using :class:`KneeLocator`."""
        if len(curve) == 0:
            return Point2D()

        p = KneeLocator(range(len(curve)), curve, curve=shape, direction=direction)
        if p.knee is None or p.knee_y is None:
            return Point2D(len(curve) - 1, curve[-1])

        return Point2D(int(p.knee), float(p.knee_y))

    @staticmethod
    def pacf(x: pd.DataFrame, max_lag: Optional[int] = None) -> pd.DataFrame:
        """
        Partial AutoCorrelation Function (PACF)

        :param x: Time series values indexed by timestamp (row) and name (column).
        :param max_lag: Maximum lag to analyse (default: None => 5% of the time series length).
        :return: PACF results indexed by time series (columns) and lag (row).
        """
        nlags = int(len(x) * 0.05) if max_lag is None else max_lag
        result = pd.DataFrame([tsa.pacf(x, nlags=nlags) for _, x in x.items()], index=x.columns).T
        return applymap(result, lambda s: s if -1 <= s <= 1 else 0)

    @staticmethod
    @ignore_warnings
    def mcorr(x: pd.DataFrame, y: pd.DataFrame) -> Mapping[str, float]:
        """
        Returns the multiple correlation coefficient between all independent variables x and each dependent variable y.

        :param x: Independent variables.
        :param y: Dependent variables.
        :return: Names of y variables mapped to coefficients.
        """
        corr = {}
        for y_col in y.columns:
            mlr = LinearRegression()
            mlr.fit(x, y[y_col])
            y_pred = mlr.predict(x)
            corr[y_col] = scipy.stats.pearsonr(y_pred, y[y_col])[0]
        return corr

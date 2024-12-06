import pandas as pd

from abc import abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from functools import reduce
from itertools import chain
from math import gcd
from typing import *

from culearn.util import tupleclass, Time


class StrMixin:
    """Returns type name as __str__."""

    def __str__(self):
        return type(self).__name__


class StrExtMixin:
    """Returns __repr__ as __str__."""

    def __str__(self):
        return repr(self)


@tupleclass
class TimeInterval(StrExtMixin):
    start: datetime
    end: datetime

    @property
    def delta(self) -> timedelta:
        """Length of the time interval as time delta."""
        return self.end - self.start

    @property
    def length(self) -> float:
        """Length of the time interval in seconds."""
        return self.delta.total_seconds()

    def overlap(self, other):
        """Finds an overlap with the other time interval"""
        if not isinstance(other, TimeInterval):
            return None

        def normalize(i: TimeInterval):
            return i if i.start <= i.end else TimeInterval(i.end, i.start)

        def has_overlap(a: TimeInterval, b: TimeInterval):
            return a.start < b.end and a.end > b.start

        def get_overlap(a: TimeInterval, b: TimeInterval):
            return TimeInterval(max(a.start, b.start), min(a.end, b.end))

        def try_get_overlap(a: TimeInterval, b: TimeInterval):
            return get_overlap(a, b) if a == b or has_overlap(a, b) else None

        return try_get_overlap(normalize(self), normalize(other))

    def contains(self, t: Union[datetime, pd.DatetimeIndex, pd.Index, pd.Series, pd.Timestamp, any]):
        """Returns start <= t < end."""
        if isinstance(t, TimeInterval):
            return self.start <= t.start <= self.end and self.start <= t.end <= self.end

        return (pd.Timestamp(self.start) <= t) & (t < pd.Timestamp(self.end))  # type: ignore

    @staticmethod
    def from_index(i: pd.Index):
        return TimeInterval(Time.py(i.min()), Time.py(i.max()))


class TimeResolution(timedelta):
    """Time resolution with equidistant time steps."""

    @property
    def length(self) -> float:
        """Length of resolution time step in seconds."""
        return self.total_seconds()

    def step(self, t: datetime) -> TimeInterval:
        """Finds the time step that the specified timestamp belongs to."""
        anchor = Time.unix(0)
        start = anchor + timedelta(seconds=self.length * int((t - anchor) / self))
        return TimeInterval(start, start + self)

    def steps(self, i: TimeInterval) -> Iterable[TimeInterval]:
        """Finds the time steps that the specified interval overlaps with."""
        t = i.start
        while t < i.end:
            s = self.step(t)
            yield s
            t = s.end

    def gcd(self, other):
        """Returns the Greatest Common Divisor between self and other."""
        return TimeResolution(seconds=gcd(int(self.length), int(other.length))) \
            if isinstance(other, type(self)) else self


class TimeTree(StrMixin):
    def __init__(self, *resolutions: TimeResolution):
        """Hierarchy of time resolutions."""

        def _gcd(*r: TimeResolution) -> TimeResolution:
            return reduce(lambda a, b: a.gcd(b), list(r))

        remaining = set(resolutions)
        self.root = _gcd(*remaining)

        self.__parent2children = defaultdict(set)
        remaining.discard(self.root)
        for parent in sorted(remaining):
            for child in list(remaining):
                if child > parent and _gcd(child, parent) == parent:
                    self.__parent2children[parent].add(child)
                    remaining.discard(child)

        self.__parent2children[self.root].update(remaining)

    def __traverse(self, parent: TimeResolution, callback: Callable[[TimeResolution], any]):
        callback(parent)
        for child in self(parent):
            self.__traverse(child, callback)

    def __call__(self, resolution: TimeResolution) -> Iterable[TimeResolution]:
        """
        Returns derived resolutions.

        :param resolution: Higher resolution.
        :return: Lower (derived) resolutions.
        """
        return self.__parent2children[resolution]

    def __iter__(self):
        """Returns all resolutions from the hierarchy."""
        result = []
        self.__traverse(self.root, result.append)
        for resolution in result:
            yield resolution


@tupleclass
class TimeSeriesID(StrExtMixin):
    """
    Composite time series ID, where:

    - source: Time series source (e.g., energy consumer or cluster).
    - value_type: Type of time series values (e.g., active or reactive power)
    - category: Source category (e.g., residential or commercial energy consumer).
    """
    source: str = 'NA'
    value_type: str = 'NA'
    category: str = 'NA'


@tupleclass
class TimeSeriesTuple(StrExtMixin):
    timestamp: datetime
    value: float

    def __round__(self, n=None):
        return TimeSeriesTuple(self.timestamp, round(self.value, n))


@tupleclass
class TimeSeriesSegment(TimeInterval):
    """
    Representation of a time series segment in the form of central moments, where:

    - length: Length of the time series segment.
    - count: Number of time series values inside the segment.
    - mean: Mean of time series values over the segment.
    - variance: Variance of time series values over the segment.
    - skewness: Skewness of time series values over the segment (not standardized).
    - kurtosis: Kurtosis of time series values over the segment (not standardized, not excess).
    """
    start: datetime
    end: datetime
    length: float = 0
    count: int = 0
    mean: float = 0
    variance: float = 0
    skewness: float = 0
    kurtosis: float = 0

    def to_csv(self):
        return self.start, self.end, self.length, self.count, self.mean, self.variance, self.skewness, self.kurtosis

    @staticmethod
    def from_csv(row: Sequence[str]):
        t = []
        t.extend([Time.py(_) for _ in row[:2]])
        t.append(float(row[2]))
        t.append(int(row[3]))
        t.extend([float(_) for _ in row[4:8]])
        return TimeSeriesSegment(*t)

    def __round__(self, n=None):
        return TimeSeriesSegment(self.start,
                                 self.end,
                                 self.length,
                                 self.count,
                                 round(self.mean, n),
                                 round(self.variance, n),
                                 round(self.skewness, n),
                                 round(self.kurtosis, n))


class TimeSeries(StrMixin):
    def __init__(self, ts_id: TimeSeriesID):
        """Abstract time series."""
        self.ts_id = ts_id


class PandasTimeSeries(TimeSeries):
    """Abstract pandas time series."""

    @abstractmethod
    def series(self) -> pd.Series:
        """Returns time series as a pandas series."""
        pass


class StreamingTimeSeries(TimeSeries):
    """Abstract streaming time series."""

    @abstractmethod
    def stream(self) -> Iterable[TimeSeriesTuple]:
        """Returns time series as a stream of tuples."""
        pass


class TimeSeriesInMemory(PandasTimeSeries, StreamingTimeSeries):
    def __init__(self, ts_id: TimeSeriesID, values: pd.Series):
        """
        Time series in whose values and timestamps are stored in memory.

        :param ts_id: Time series ID.
        :param values: Time series values with time index.
        """
        if type(values.index) != pd.DatetimeIndex:
            raise Exception('Index of values has to be pandas.DatetimeIndex.')

        super().__init__(ts_id)
        self.values = values

    def series(self) -> pd.Series:
        return self.values

    def stream(self) -> Iterable[TimeSeriesTuple]:
        for t, v in self.values.items():
            yield TimeSeriesTuple(Time.py(t), float(v))


class TimeSeriesDataFrame(pd.DataFrame, StrMixin):
    def __init__(self, ts_id: TimeSeriesID, *args, **kwargs):
        """Wraps around a data frame with time series values."""
        super().__init__(*args, **kwargs)
        self.ts_id = ts_id

    def select(self, interval: TimeInterval):
        """Returns time series values in the specified interval"""
        return TimeSeriesDataFrame(self.ts_id, self[interval.contains(self.index)])


@tupleclass
class PredictionDataset(StrMixin):
    """
    Dataset with input and output values for time series prediction, where:

    - x: Input time series indexed by timestamp (usually small number of time series).
    - y: Output time series (abstraction over significantly large number of time series).
    """
    x: pd.DataFrame
    y: Sequence[TimeSeries]


@tupleclass
class PredictionInterval(StrMixin):
    """Wraps around prediction interval probability, and lower and upper bounds."""
    p: float
    lower: pd.Series
    upper: pd.Series


class TimeSeriesPrediction(list, StrMixin):
    def __init__(self, ts_id: TimeSeriesID, *args, **kwargs):
        """Wraps around one or more prediction intervals for one time series."""
        super().__init__(*args, **kwargs)
        self.ts_id = ts_id

    def to_frame(self) -> TimeSeriesDataFrame:
        """Returns data frame with 'Lower <probability>% PI' and 'Upper <probability>% PI' columns."""
        i = [[pi.lower.to_frame(f'Lower {100 * pi.p}% PI'),
              pi.upper.to_frame(f'Upper {100 * pi.p}% PI')] for pi in self]
        return TimeSeriesDataFrame(self.ts_id, pd.concat(chain.from_iterable(i), axis=1))


class MultiSeries(StrMixin):
    """Abstract multiresolution time series representation."""

    @abstractmethod
    def last(self) -> Optional[TimeSeriesTuple]:
        """Returns the last time series tuple."""
        pass

    @abstractmethod
    def buffer(self, resolution: TimeResolution) -> Optional[TimeSeriesSegment]:
        """Returns buffered time series segment for the specified resolution."""
        pass

    @abstractmethod
    def disc(self, resolution: TimeResolution) -> Sequence[TimeSeriesSegment]:
        """Returns time series segments from disc for the specified resolution."""
        pass

    @abstractmethod
    def update_last(self, last: Optional[TimeSeriesTuple]):
        """Replaces the last time series tuple with the new one."""
        pass

    @abstractmethod
    def update_buffer(self, resolution: TimeResolution, segment: Optional[TimeSeriesSegment]):
        """Replaces the buffered time series segment with the new one."""
        pass

    @abstractmethod
    def extend_disc(self, resolution: TimeResolution, *segment: TimeSeriesSegment):
        """Extends disc with the new time series segment."""
        pass

    def __call__(self, resolution: TimeResolution) -> Sequence[TimeSeriesSegment]:
        """Reads time series segments from both disc and buffer for the specified resolution."""
        disk = self.disc(resolution)
        buffer = self.buffer(resolution)
        return list(disk) if buffer is None else list(disk) + [buffer]


class MultiSeriesDictionary(MultiSeries):
    def __init__(self):
        """
        :class:`MultiSeries` with in-memory (dictionary-based) buffer and disc, primarily for testing purposes.
        Note that custom :class:`MultiSeries` implementation can be used if the data should be stored to database.
        """
        self.__last: Optional[TimeSeriesTuple] = None
        self.__buffer: MutableMapping[TimeResolution, TimeSeriesSegment] = {}
        self.__disc = defaultdict(list)

    def last(self) -> Optional[TimeSeriesTuple]:
        return self.__last

    def buffer(self, resolution: TimeResolution) -> Optional[TimeSeriesSegment]:
        return self.__buffer.get(resolution)

    def disc(self, resolution: TimeResolution) -> Sequence[TimeSeriesSegment]:
        return self.__disc[resolution]

    def update_last(self, last: Optional[TimeSeriesTuple]):
        self.__last = last

    def update_buffer(self, resolution: TimeResolution, segment: Optional[TimeSeriesSegment]):
        self.__buffer[resolution] = segment

    def extend_disc(self, resolution: TimeResolution, *segment: TimeSeriesSegment):
        self.__disc[resolution].extend(segment)

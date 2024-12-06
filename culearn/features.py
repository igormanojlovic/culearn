import numpy as np
import scipy

from collections import namedtuple
from math import sqrt
from npeet.entropy_estimators import mi
from sklearn import decomposition
from workalendar.core import CoreCalendar

from culearn.base import *
from culearn.util import Math, applymap, ignore_warnings


class TimeEncoder(StrMixin):
    def __init__(self, *features: str):
        """
        Abstract time feature encoder.

        :param features: Feature names that will be used to create named tuples.
        """
        self.features = features

    @abstractmethod
    def _encode(self, t: datetime) -> Sequence[float]:
        pass

    def __call__(self, *t: datetime):
        """Transforms the specified timestamp into numeric values."""
        features = namedtuple(f'{type(self).__name__}Tuple', self.features)
        encoded = [features(*self._encode(_)) for _ in t]
        return encoded[0] if len(encoded) == 1 else encoded


class PeriodicTimeEncoder(TimeEncoder):
    def __init__(self):
        t = type(self).__name__
        super().__init__(f'{t}X', f'{t}Y')

    @abstractmethod
    def _length(self) -> int:
        """Returns the cycle length."""
        pass

    @abstractmethod
    def _index(self, t: datetime) -> int:
        """Extracts the index of the timestamp position on the cycle."""
        pass

    def _encode(self, t: datetime) -> Sequence[float]:
        degree = 2 * np.pi * self._index(t) / self._length()
        return [(np.sin(degree)+1)/2, (np.cos(degree)+1)/2]


class MonthOfYear(PeriodicTimeEncoder):
    """Encodes month of year as 2D Cartesian coordinates on a unit circle (clock positions)."""

    def _length(self) -> int:
        return 12

    def _index(self, t: datetime) -> int:
        return t.month - 1


class WeekOfYear(PeriodicTimeEncoder):
    """Encodes week of year as 2D Cartesian coordinates on a unit circle (clock positions)."""

    def _length(self) -> int:
        return 52

    def _index(self, t: datetime) -> int:
        c = t.isocalendar()
        w = c[1] if isinstance(c, tuple) else c.week # Backward compatibility fix.
        return w - 1


class DayOfWeek(PeriodicTimeEncoder):
    """Encodes day of week as 2D Cartesian coordinates on a unit circle (clock positions)."""

    def _length(self) -> int:
        return 7

    def _index(self, t: datetime) -> int:
        return t.weekday()


class TimeOfDay(PeriodicTimeEncoder):
    """Encodes time of day as 2D Cartesian coordinates on a unit circle (clock positions)."""

    def _length(self) -> int:
        return 24 * 3600

    def _index(self, t: datetime) -> int:
        return t.hour * 3600 + t.minute * 60 + t.second


class Holiday(TimeEncoder):
    def __init__(self, calendar: CoreCalendar):
        """
        Encodes holidays as integers (0=non-holiday, 1=holiday).

        :param calendar: Work calendar for specific geographic region.
        """
        super().__init__(type(self).__name__)
        self.calendar = calendar

    def _encode(self, t: datetime) -> Sequence[float]:
        return [int(self.calendar.is_holiday(t))]


class DayType(TimeEncoder):
    def __init__(self, calendar: CoreCalendar):
        """
        Encodes days of week and holidays as integers (0=workday, 1=weekend, 2=holiday).

        :param calendar: Work calendar for specific geographic region.
        """
        super().__init__(type(self).__name__)
        self.calendar = calendar

    def _encode(self, t: datetime) -> Sequence[float]:
        if self.calendar.is_holiday(t):
            return [2]
        if self.calendar.is_working_day(t):
            return [0]
        return [1]


class TimeEncoders(TimeEncoder):
    def __init__(self, *encoders: TimeEncoder):
        """Encodes multiple time features using the specified time encoders."""
        super().__init__(*list(chain.from_iterable([_.features for _ in encoders])))
        self.encoders = encoders

    def _encode(self, t: datetime) -> Sequence[float]:
        return list(chain.from_iterable([_._encode(t) for _ in self.encoders]))

    def __str__(self):
        return f'{type(self).__name__}{tuple(self.encoders)}'


class PiecewiseFunction(StrExtMixin):
    def __init__(self, a: TimeSeriesTuple, b: TimeSeriesTuple):
        """Abstract piecewise function between two subsequent time series tuples."""
        self.a = a
        self.b = b

    @abstractmethod
    def integrate(self, i: TimeInterval) -> float:
        """Integrates the function over the specified time interval."""
        pass


class PiecewiseLinearFunction(PiecewiseFunction):
    def __init__(self, a: TimeSeriesTuple, b: TimeSeriesTuple):
        """Piecewise linear function between two subsequent time series tuples."""
        super().__init__(a, b)
        xa = self.__x(a.timestamp)
        xb = self.__x(b.timestamp)
        ya = a.value
        yb = b.value
        self.__slope = 0 if xa == xb else (yb - ya) / (xb - xa)
        self.__intercept = ya - self.__slope * xa

    def __x(self, t: datetime):
        return TimeInterval(self.a.timestamp, t).length

    def __integrate(self, t: datetime) -> float:
        x = self.__x(t)
        return self.__slope * x ** 2 / 2 + self.__intercept * x

    def integrate(self, i: TimeInterval) -> float:
        return self.__integrate(i.end) - self.__integrate(i.start)


class PiecewiseStepFunction(PiecewiseFunction):
    def __init__(self, a: TimeSeriesTuple, b: TimeSeriesTuple):
        """Piecewise step function between two subsequent time series tuples."""
        super().__init__(a, b)

    def integrate(self, i: TimeInterval) -> float:
        def _length(interval: TimeInterval):
            return 0 if interval is None else interval.length

        inner = TimeInterval(self.a.timestamp, self.b.timestamp).overlap(i)
        outer = TimeInterval(self.b.timestamp, datetime.max).overlap(i)
        return _length(inner) * self.a.value + _length(outer) * self.b.value


class PiecewiseApproximation(StrMixin):
    """Abstract approximation of piecewise functions connecting the time series tuples."""

    @abstractmethod
    def represent(self, a: TimeSeriesTuple, b: TimeSeriesTuple) -> PiecewiseFunction:
        """Represents a time series between two subsequent tuples."""
        pass

    @abstractmethod
    def aggregate(self, f: PiecewiseFunction, i: TimeInterval) -> TimeSeriesSegment:
        """Aggregates the specified time series representation function over the specified time interval."""
        pass

    @abstractmethod
    def _merge(self, a: TimeSeriesSegment, b: TimeSeriesSegment) -> TimeSeriesSegment:
        """Merges two time series approximations into one."""
        pass

    def merge(self, *segments: TimeSeriesSegment) -> TimeSeriesSegment:
        """Merges multiple time series approximations into one."""
        return reduce(lambda a, b: self._merge(a, b), [s for s in segments if s is not None])


class PiecewiseStatisticalApproximation(PiecewiseApproximation):
    def __init__(self, piecewise_function: Type[PiecewiseFunction] = PiecewiseLinearFunction):
        """
        Statistical approximation of piecewise functions connecting the time series tuples, based on [1]_ [2]_.

        References
        -----
        .. [1] Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić, Darko Čapko:
               Hierarchical multiresolution representation of streaming time series,
               Big Data Research 26: 100256 (2021),
               https://doi.org/10.1016/j.bdr.2021.100256
        .. [2] Qinglin Cai, Ling Chen, Jianling Sun:
               Piecewise statistic approximation based similarity measure for time series,
               Knowledge-Based Systems 85: 181–195 (2015),
               https://doi.org/10.1016/j.knosys.2015.05.005
        """
        self.piecewise_function = piecewise_function

    def represent(self, a: TimeSeriesTuple, b: TimeSeriesTuple) -> PiecewiseFunction:
        return self.piecewise_function(a, b)

    def aggregate(self, f: PiecewiseFunction, i: TimeInterval) -> TimeSeriesSegment:
        length = i.length
        if length == 0:
            return TimeSeriesSegment(i.start, i.end)

        count = int(i.contains(f.a.timestamp)) + int(i.contains(f.b.timestamp))
        mean = f.integrate(i) / length

        def moment(degree: int):
            dev_a = TimeSeriesTuple(f.a.timestamp, (f.a.value - mean) ** degree)
            dev_b = TimeSeriesTuple(f.b.timestamp, (f.b.value - mean) ** degree)
            return self.represent(dev_a, dev_b).integrate(i) / length

        variance = moment(2)
        skewness = moment(3)
        kurtosis = moment(4)
        return TimeSeriesSegment(i.start, i.end, length, count, mean, variance, skewness, kurtosis)

    def _merge(self, a: TimeSeriesSegment, b: TimeSeriesSegment) -> TimeSeriesSegment:
        i = TimeInterval(min(a.start, b.start), max(a.end, b.end))
        length = a.length + b.length
        if length == 0:
            return TimeSeriesSegment(i.start, i.end)

        count = a.count + b.count
        mean = (a.length * a.mean + b.length * b.mean) / length

        def moment(ma: float, mb: float, degree: int):
            sum_dev_a = a.length * (ma + (a.mean - mean) ** degree)
            sum_dev_b = b.length * (mb + (b.mean - mean) ** degree)
            return (sum_dev_a + sum_dev_b) / length

        variance = moment(a.variance, b.variance, 2)
        skewness = moment(a.skewness, b.skewness, 3)
        kurtosis = moment(a.kurtosis, b.kurtosis, 4)
        return TimeSeriesSegment(i.start, i.end, length, count, mean, variance, skewness, kurtosis)


@tupleclass
class HMTSR(StrMixin):
    """
    Hierarchical Multiresolution Time Series Representation (HMTSR) model [1]_: generic solution for multiresolution
    streaming time series (STS) representation based on two generic data structures (buffer and disk) and a one-pass
    stream processing algorithm that offers high processing speed at low RAM consumption.

    References
    -----
    .. [1] Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić, Darko Čapko:
           Hierarchical multiresolution representation of streaming time series,
           Big Data Research 26: 100256 (2021),
           https://doi.org/10.1016/j.bdr.2021.100256
    """
    resolutions: TimeTree
    multiseries: MultiSeries
    approximation: PiecewiseApproximation = PiecewiseStatisticalApproximation()

    def __derive(self, parent_resolution: TimeResolution, parent_step: TimeInterval, child_segment: TimeSeriesSegment):
        for derived_resolution in self.resolutions(parent_resolution):
            parent_buffer = self.multiseries.buffer(derived_resolution)
            parent_buffer = self.approximation.merge(parent_buffer, child_segment)
            derived_step = derived_resolution.step(parent_step.start)
            if parent_step.end == derived_step.end:
                self.multiseries.extend_disc(derived_resolution, parent_buffer)
                self.__derive(derived_resolution, derived_step, parent_buffer)
                parent_buffer = None

            self.multiseries.update_buffer(derived_resolution, parent_buffer)

    def process(self, t: TimeSeriesTuple):
        """
        Applies one-pass stream processing algorithm that incrementally builds the multiresolution STS representation.
        Note that the STS tuples must be ordered by timestamp to correctly build the representation.
        """
        resolution = self.resolutions.root
        last = self.multiseries.last()
        if last is None:
            self.multiseries.update_last(t)
            return

        if t.timestamp <= last.timestamp:
            return

        representation = self.approximation.represent(last, t)
        interval = TimeInterval(last.timestamp, t.timestamp)
        buffer = self.multiseries.buffer(resolution)
        for step in resolution.steps(interval):
            segment = self.approximation.aggregate(representation, step.overlap(interval))
            buffer = self.approximation.merge(buffer, segment)
            if step.end <= t.timestamp:
                self.multiseries.extend_disc(resolution, buffer)
                self.__derive(resolution, step, buffer)
                buffer = TimeSeriesSegment(buffer.end, buffer.end)

        self.multiseries.update_buffer(resolution, buffer)
        self.multiseries.update_last(t)
        return self


class Approximator(StrMixin):
    @abstractmethod
    def fit(self, x: TimeSeries, resolution: TimeResolution):
        """Approximates time series values at the specified time resolution."""
        pass

    @abstractmethod
    def transform(self, interval: TimeInterval) -> pd.DataFrame:
        """
        Transforms time series values in the specified interval into raw central moments at
        previously specified time resolution. The moments are indexed by timestamp (row)
        and name (columns: 'count', 'mean', 'variance', 'skewness', 'kurtosis').
        """
        pass

    def fit_transform(self, x: TimeSeries, resolution: TimeResolution, interval: TimeInterval) -> pd.DataFrame:
        """
        Transforms time series values in the specified interval into raw central moments at
        previously specified time resolution. The moments are indexed by timestamp (row)
        and name (columns: 'count', 'mean', 'variance', 'skewness', 'kurtosis').
        """
        self.fit(x, resolution)
        return self.transform(interval)


class SeriesApproximator(Approximator):
    def __init__(self, skip_zeros=True):
        """
        Approximates :class:`PandasTimeSeries` utilizing pandas resampling at transform-time.

        :param skip_zeros: Whether to skip zeros when approximating time series values.
        """
        self.skip_zeros = skip_zeros
        self.__x: Optional[PandasTimeSeries] = None
        self.__resolution: Optional[TimeSeries] = None

    def fit(self, x: TimeSeries, resolution: TimeResolution):
        self.__x = x
        self.__resolution = resolution

    def transform(self, interval: TimeInterval) -> pd.DataFrame:
        x = self.__x.series()
        if self.skip_zeros:
            x = x.replace(0, np.nan)

        x_mean = x[interval.contains(x.index)].dropna().resample(self.__resolution).mean()
        if len(x_mean) == 0:
            return pd.DataFrame()

        x_moments = pd.DataFrame(np.ones_like(x_mean), index=x_mean.index, columns=['count'])
        x_moments['mean'] = x_mean
        x_moments['variance'] = x_moments['skewness'] = x_moments['kurtosis'] = 0
        return x_moments


class StreamApproximator(Approximator):
    def __init__(self, struct: MultiSeries, skip_zeros=True):
        """
        Approximates :class:`StreamingTimeSeries` utilizing :class:`HMTSR` at fit-time.

        Use the :class:`StreamApproximator` only if the original data time resolution is significantly higher than the
        expected time resolution. Otherwise, use the :class:`SeriesApproximator`.

        :param struct: Data structure for storing time series approximation.
        :param skip_zeros: Whether to skip zeros when approximating time series values.
        """
        self.skip_zeros = skip_zeros
        self.struct = struct
        self.__approx: Optional[HMTSR] = None

    def __skip_tuple(self, t: TimeSeriesTuple):
        return self.skip_zeros and t.value == 0

    def __skip_segment(self, s: TimeSeriesSegment):
        return self.skip_zeros and s.count == 0

    def __process(self, x: StreamingTimeSeries):
        for t in x.stream():
            if not self.__skip_tuple(t):
                self.__approx.process(t)

    def __select(self, interval: TimeInterval) -> Iterable[TimeSeriesSegment]:
        for s in self.__approx.multiseries(self.__approx.resolutions.root):
            if not self.__skip_segment(s) and s.overlap(interval):
                yield s

    def fit(self, x: TimeSeries, resolution: TimeResolution):
        self.__approx = HMTSR(TimeTree(resolution), self.struct)
        self.__process(x)  # type: ignore

    def transform(self, interval: TimeInterval) -> pd.DataFrame:
        max_length = self.__approx.resolutions.root.length
        moments = pd.DataFrame([(s.start, s.length / max_length, s.mean, s.variance, s.skewness, s.kurtosis)
                                for s in self.__select(interval)])
        moments.columns = ['timestamp', 'count', 'mean', 'variance', 'skewness', 'kurtosis']
        moments.set_index('timestamp', inplace=True)
        return moments


class _FeatureMixin(StrMixin):
    @staticmethod
    def to_index(scores: Sequence[float], ratio: Optional['float > 0 and float < 1'] = None) -> Sequence[int]:
        """
        Returns indexes of the most important features according to the feature scores.

        :param scores: Feature scores.
        :param ratio: Ratio of the most important features to select (default: None => apply the knee method).
        :return: Indexes of the most important features.
        """

        def n_cumsum(s: Sequence):
            cumsum_scores = np.cumsum(s / np.sum(s))
            return len(cumsum_scores[cumsum_scores <= ratio]) + 1

        def n_select(s: Sequence):
            n = Math.knee(s).x if ratio is None else n_cumsum(s)
            return min(max(3, n), len(scores))

        index_score = sorted(zip(range(len(scores)), scores), key=lambda _: _[1], reverse=True)
        sorted_index = [i for i, _ in index_score]
        sorted_scores = [s for _, s in index_score]
        return sorted_index[:n_select(sorted_scores)]


class FeatureExtractor(_FeatureMixin):
    """Abstract feature extractor."""

    @abstractmethod
    def fit(self, x: pd.DataFrame):
        """
        Evaluates the features.

        :param x: Input values indexed by object (row) and feature (column).
        """
        pass

    @abstractmethod
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the most important features.

        :param x: Input values indexed by object (row) and feature (column).
        :return: Transformed input values indexed by object (row) and extracted feature (column).
        """
        pass

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates the features and then extracts the most important ones.

        :param x: Input values indexed by object (row) and feature (column).
        :return: Transformed input values indexed by object (row) and extracted feature (column).
        """
        return self.fit(x).transform(x)

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """
        Returns feature extraction scores indexed by 'feature', with the 'score' and 'selected' columns showing the
        score and whether the feature is selected, respectively.
        """
        pass


class PCA(FeatureExtractor):
    def __init__(self, ratio: Optional['float > 0 and float < 1'] = None):
        """
        Performs feature extraction utilizing Principal Component Analysis (PCA). The features extracted by PCA
        (principal components) are selected according to the percentage of the explained variance in the data.

        :param ratio: Ratio of the most important features to select (default: None => apply the knee method).
        """
        self.ratio = ratio

        self.__pca = decomposition.PCA()
        self.scores = pd.Series(dtype=float)
        self.indexes = []

    def fit(self, x: pd.DataFrame):
        self.__pca.fit(x)
        self.scores = pd.Series(self.__pca.explained_variance_ratio_)
        self.indexes = self.to_index(self.scores, self.ratio)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        transformed = self.__pca.transform(x)
        return pd.DataFrame(transformed[:, self.indexes], index=x.index)

    def summary(self) -> pd.DataFrame:
        """
        Returns feature extraction scores indexed by 'feature', with the 'score' and 'selected' columns showing the
        score and whether the feature is selected, respectively.
        """
        scores = self.scores.to_frame('score')
        scores['selected'] = False
        scores.loc[self.indexes, 'selected'] = True
        return scores.rename_axis('feature')


class FeatureSelector(_FeatureMixin):
    """Abstract feature selector."""

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Evaluates input features.

        :param x: Input values indexed by object (row) and feature (column).
        :param y: Output values indexed by object (row) and feature (column).
        """
        pass

    @abstractmethod
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Selects the most important input features.

        :param x: Input values indexed by object (row) and feature (column).
        :return: Subset of input values indexed by object (row) and selected feature (column).
        """
        pass

    def fit_transform(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates input features and then selects the most important ones.

        :param x: Input values indexed by object (row) and feature (column).
        :param y: Output values indexed by object (row) and feature (column).
        :return: Subset of input values indexed by object (row) and selected feature (column).
        """
        return self.fit(x, y).transform(x)

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """
        Returns feature selection scores indexed by 'x' and 'y' features, with the 'score' and 'selected' columns
        showing the score and whether the x feature is selected, respectively.
        """
        pass


class _FilteringFeatureSelector(FeatureSelector):
    def __init__(self, ratio: 'float > 0 and float < 1'):
        """
        Abstract feature subset selector.

        :param ratio: Ratio of the most important features to select.
        """
        self.ratio = ratio
        self.scores = pd.DataFrame()
        self.indexes = []

    @abstractmethod
    def _fit(self, x: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Returns feature scores indexed by x feature names (rows) and y feature names (columns)."""
        pass

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        scores = []
        for y_column in y:
            scores.append(self._fit(x, y[y_column]))
        self.scores = pd.concat(scores, axis=1)
        self.scores.index = x.columns
        self.scores.columns = y.columns
        self.indexes = self.to_index(self.scores.mean(axis=1), self.ratio)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.iloc[:, self.indexes]

    def summary(self) -> pd.DataFrame:
        scores = self.scores.reset_index()
        scores['selected'] = False
        scores.loc[self.indexes, 'selected'] = True
        return scores.melt(['index', 'selected']) \
            .rename(columns={'index': 'x', 'variable': 'y', 'value': 'score'}) \
            .set_index(['x', 'y'])[['score', 'selected']]


class JMIM(_FilteringFeatureSelector):
    def __init__(self, ratio: 'float > 0 and float < 1' = 0.9, k: 'int > 1' = 3):
        """
        Performs feature selection utilizing the Joint Mutual Information Maximisation (JMIM) [1]_
        with the k-Nearest Neighbours (kNN) based entropy estimation.

        References
        -----
        .. [1] Mohamed Bennasar, Yulia Hicksm, Rossitza Setchi,
               Feature selection using joint mutual information maximisation,
               Expert Systems with Applications 42 (22): 8520–8532 (2015),
               https://doi.org/10.1016/j.eswa.2015.07.007

        :param ratio: Ratio of the most important features to select.
        :param k: Number of the nearest neighbours in kNN.
        """
        super().__init__(ratio)
        self.k = k

    def _fit(self, x: pd.DataFrame, y: pd.Series) -> pd.Series:
        m = pd.Series([mi(x[i], y, k=self.k) for i in x], index=x.columns)
        s = [min([m[k] + mi(x[i], y, x[k], k=self.k) for k in x if k != i]) for i in x]
        return pd.Series(s, index=x.columns)


class DummyFeatureSelector(FeatureSelector):
    def __init__(self):
        """Returns all features without the selection, primarily for testing purposes."""
        self.x_columns = []
        self.y_columns = []

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        self.x_columns = list(x.columns)
        self.y_columns = list(y.columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x

    def summary(self) -> pd.DataFrame:
        s = [{'x': x, 'y': y, 'score': 0, 'selected': True} for x in self.x_columns for y in self.y_columns]
        return pd.DataFrame(s).set_index(['x', 'y'])


class LagSelector(StrMixin):
    def __init__(self, lag=-1):
        """
        Abstract lag selector.

        :param lag: Initial lag, before the selection.
        """
        self.lag = lag

    @abstractmethod
    def _fit_predict(self, x: pd.DataFrame) -> int:
        """Calculates time series lag scores and recommends the lag."""
        pass

    def fit(self, x: pd.DataFrame):
        """
        Calculates time series lag scores and recommended lag.

        :param x: Time series values.
        """
        self.lag = self._fit_predict(x)
        return self

    def fit_predict(self, x: pd.DataFrame) -> int:
        """
        Calculates time series lag scores and recommends the lag.

        :param x: Time series values.
        :return: Recommended lag.
        """
        return self.fit(x).lag

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """
        Returns lag selection scores indexed by 'feature' and 'lag', with the 'score', 'selected', and 'ci' columns
        showing the score, whether the feature is selected, and the confidence interval radius, respectively.
        """
        pass


class PACF(LagSelector):
    def __init__(self, max_lag: 'int > 0' = None, p: '0 < float < 1' = 0.99):
        """
        Recommends time series lag according to the results of Partial AutoCorrelation Function (PACF).
        The recommendation is based on finding the knee of a curve defined by a cumulative sum of the
        PACF results outside the confidence interval for the specified probability.

        :param max_lag: Maximum lag to analyse (default: None => 5% of the time series length).
        :param p: Autocorrelation confidence interval probability.
        """
        super().__init__()
        self.max_lag = max_lag
        self.p = p

        self.scores = pd.DataFrame()
        self.ci = float('nan')

    @staticmethod
    @ignore_warnings
    def __leg(scores, ci):
        leg = applymap(scores, lambda s: 0 if abs(s) < ci else pow(s - ci, 2))
        leg = leg.abs().iloc[::-1].cumsum(axis=0).iloc[::-1]
        leg = 1 - np.cumsum(leg / np.sum(leg, axis=0), axis=0)
        return leg.mean(axis=1)

    def _fit_predict(self, x: pd.DataFrame):
        self.scores = Math.pacf(x, self.max_lag)
        self.ci = scipy.stats.norm.ppf(0.5 + self.p / 2) / sqrt(len(x))
        return Math.knee(self.__leg(self.scores, self.ci)).x

    def summary(self) -> pd.DataFrame:
        selected = pd.DataFrame(self.scores.index <= self.lag, columns=['selected'])
        merged = pd.concat((self.scores, selected), axis=1).reset_index() \
            .melt(['index', 'selected']).rename(columns={'index': 'lag', 'variable': 'feature', 'value': 'score'}) \
            .set_index(['feature', 'lag'])[['score', 'selected']]
        merged['ci'] = self.ci
        return merged


class DummyLagSelector(LagSelector):
    def __init__(self, lag):
        """Returns the specified lag without the selection, primarily for testing purposes."""
        super().__init__(lag)
        self.x_columns = []

    def _fit_predict(self, x: pd.DataFrame) -> int:
        self.x_columns = list(x.columns)
        return self.lag

    def summary(self) -> pd.DataFrame:
        s = [{'feature': x, 'lag': lag, 'score': 0, 'selected': True, 'ci': 0}
             for x in self.x_columns for lag in range(self.lag + 1)]
        return pd.DataFrame(s).set_index(['feature', 'lag'])

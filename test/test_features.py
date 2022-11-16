from unittest import TestCase

from culearn.data import *
from culearn.features import *
from culearn.util import *
from numpy.testing import assert_array_equal
from parameterized import parameterized
from workalendar.europe import Serbia


def pf(a: (int, float), b: (int, float), f: Type[PiecewiseFunction] = PiecewiseLinearFunction):
    _a = TimeSeriesTuple(Time.unix(a[0]), a[1])
    _b = TimeSeriesTuple(Time.unix(b[0]), b[1])
    return f(_a, _b)


class FakeTimeEncoder(TimeEncoder):
    def __init__(self, value: float):
        super().__init__(f'Fake{value}')
        self.value = value

    def _encode(self, t: datetime) -> tuple:
        return tuple([self.value])


class TestPeriodicEncoder:
    @abstractmethod
    def subTest(self, msg=''):
        pass

    @abstractmethod
    def assertEqual(self, a, b, msg=''):
        pass

    @abstractmethod
    def _encoder(self) -> PeriodicTimeEncoder:
        pass

    @abstractmethod
    def _sample_min(self) -> (datetime, int, float, float):
        pass

    @abstractmethod
    def _sample_q1(self) -> (datetime, int, float, float):
        pass

    @abstractmethod
    def _sample_q2(self) -> (datetime, int, float, float):
        pass

    @abstractmethod
    def _sample_q3(self) -> (datetime, int, float, float):
        pass

    @abstractmethod
    def _sample_max(self) -> (datetime, int, float, float):
        pass

    def _samples(self):
        yield self._sample_min()
        yield self._sample_q1()
        yield self._sample_q2()
        yield self._sample_q3()
        yield self._sample_max()

    def test_length(self):
        expected = self._sample_max()[1] + 1
        actual = self._encoder()._length()
        self.assertEqual(expected, actual)

    def test_encode(self):
        for t, _, x, y in self._samples():
            with self.subTest(t):
                actual = self._encoder()(t)
                round_actual = [round(_, 9) for _ in list(actual)]
                self.assertEqual([x, y], round_actual)


class TestMonthOfYear(TestCase, TestPeriodicEncoder):
    def _encoder(self) -> PeriodicTimeEncoder:
        return MonthOfYear()

    def _sample_min(self) -> (str, int, float, float):
        return datetime(2000, 1, 1), 0, 0.5, 1

    def _sample_q1(self) -> (str, int, float, float):
        return datetime(2000, 4, 1), 3, 1, 0.5

    def _sample_q2(self) -> (str, int, float, float):
        return datetime(2000, 7, 1), 6, 0.5, 0

    def _sample_q3(self) -> (str, int, float, float):
        return datetime(2000, 10, 1), 9, 0, 0.5

    def _sample_max(self) -> (str, int, float, float):
        return datetime(2000, 12, 1), 11, 0.25, 0.933012702


class TestWeekOfYear(TestCase, TestPeriodicEncoder):
    def _encoder(self) -> PeriodicTimeEncoder:
        return WeekOfYear()

    def _sample_min(self) -> (str, int, float, float):
        return datetime(2000, 1, 3), 0, 0.5, 1

    def _sample_q1(self) -> (str, int, float, float):
        return datetime(2000, 4, 3), 13, 1, 0.5

    def _sample_q2(self) -> (str, int, float, float):
        return datetime(2000, 7, 3), 26, 0.5, 0

    def _sample_q3(self) -> (str, int, float, float):
        return datetime(2000, 10, 4), 39, 0, 0.5

    def _sample_max(self) -> (str, int, float, float):
        return datetime(2000, 12, 31), 51, 0.43973166, 0.996354437


class TestDayOfWeek(TestCase, TestPeriodicEncoder):
    def _encoder(self) -> PeriodicTimeEncoder:
        return DayOfWeek()

    def _sample_min(self) -> (str, int, float, float):
        return datetime(2000, 1, 3), 0, 0.5, 1

    def _sample_q1(self) -> (str, int, float, float):
        return datetime(2000, 1, 4), 1, 0.890915741, 0.811744901

    def _sample_q2(self) -> (str, int, float, float):
        return datetime(2000, 1, 6), 3, 0.71694187, 0.049515566

    def _sample_q3(self) -> (str, int, float, float):
        return datetime(2000, 1, 8), 5, 0.012536044, 0.388739533

    def _sample_max(self) -> (str, int, float, float):
        return datetime(2000, 1, 9), 6, 0.109084259, 0.811744901


class TestTimeOfDay(TestCase, TestPeriodicEncoder):
    def _encoder(self) -> PeriodicTimeEncoder:
        return TimeOfDay()

    def _sample_min(self) -> (str, int, float, float):
        return datetime(2000, 1, 1), 0, 0.5, 1

    def _sample_q1(self) -> (str, int, float, float):
        return datetime(2000, 1, 1, 6, 0), 6 * 3600, 1, 0.5

    def _sample_q2(self) -> (str, int, float, float):
        return datetime(2000, 1, 1, 12, 0), 12 * 3600, 0.5, 0

    def _sample_q3(self) -> (str, int, float, float):
        return datetime(2000, 1, 1, 18, 0), 18 * 3600, 0, 0.5

    def _sample_max(self) -> (str, int, float, float):
        return datetime(2000, 1, 1, 23, 59, 59), 24 * 3600 - 1, 0.499963639, 0.999999999


class TestHoliday(TestCase):
    @parameterized.expand([
        [datetime(2000, 1, 1), 1],
        [datetime(2000, 1, 2), 1],
        [datetime(2000, 1, 3), 0],
        [datetime(2000, 5, 1), 1],
        [datetime(2000, 12, 31), 0],
    ])
    def test_encode(self, t: datetime, expected: int):
        actual = Holiday(Serbia())(t)
        with self.subTest(t):
            self.assertEqual([expected], list(actual))


class TestDayType(TestCase):
    @parameterized.expand([
        [datetime(2000, 1, 1), 2],
        [datetime(2000, 1, 2), 2],
        [datetime(2000, 1, 3), 0],
        [datetime(2000, 1, 4), 0],
        [datetime(2000, 1, 5), 0],
        [datetime(2000, 1, 6), 0],
        [datetime(2000, 1, 7), 2],
        [datetime(2000, 1, 8), 1],
        [datetime(2000, 1, 9), 1],
        [datetime(2000, 5, 1), 2],
        [datetime(2000, 12, 29), 0],
        [datetime(2000, 12, 30), 1],
        [datetime(2000, 12, 31), 1],
    ])
    def test_encode(self, t: datetime, expected: int):
        actual = DayType(Serbia())(t)
        with self.subTest(t):
            self.assertEqual([expected], list(actual))


class TestTimeEncoders(TestCase):
    def test_encode(self):
        encoder = TimeEncoders(FakeTimeEncoder(1), FakeTimeEncoder(2), FakeTimeEncoder(3),
                               MonthOfYear(), WeekOfYear(), DayOfWeek(), TimeOfDay(),
                               Holiday(Serbia()), DayType(Serbia()))
        actual = list(encoder(datetime(2000, 1, 1)))

        with self.subTest('length'):
            self.assertEqual(13, len(actual))

        with self.subTest('values'):
            self.assertEqual([1, 2, 3], actual[:3])


class TestPiecewiseLinearFunction(TestCase):
    @parameterized.expand([
        # Interval length is zero:
        [(1, 0), (1, 0), (1, 1), 0],
        [(1, 0), (3, 0), (2, 2), 0],
        [(1, 1), (3, 1), (2, 2), 0],
        # Horizontal above zero:
        [(1, 1), (4, 1), (2, 3), 1],
        [(1, 3), (4, 3), (2, 3), 3],
        [(1, 3), (4, 3), (1, 4), 9],
        [(1, 3), (4, 3), (0, 5), 15],
        # Horizontal below zero:
        [(1, -1), (4, -1), (2, 3), -1],
        [(1, -3), (4, -3), (2, 3), -3],
        [(1, -3), (4, -3), (1, 4), -9],
        [(1, -3), (4, -3), (0, 5), -15],
        # Decreasing around zero:
        [(1, 3), (5, -3), (0, 6), 0],
        [(1, 3), (5, -3), (1, 5), 0],
        [(1, 3), (5, -3), (1, 3), 3],
        [(1, 3), (5, -3), (3, 5), -3],
        # Increasing around zero:
        [(1, -3), (5, 3), (0, 6), 0],
        [(1, -3), (5, 3), (1, 5), 0],
        [(1, -3), (5, 3), (1, 3), -3],
        [(1, -3), (5, 3), (3, 5), 3],
        # Decreasing above zero:
        [(1, 5), (5, 2), (0, 6), 21],
        [(1, 5), (5, 2), (1, 5), 14],
        [(1, 5), (5, 2), (1, 3), 8.5],
        [(1, 5), (5, 2), (3, 5), 5.5],
        # Increasing above zero:
        [(1, 2), (5, 5), (0, 6), 21],
        [(1, 2), (5, 5), (1, 5), 14],
        [(1, 2), (5, 5), (1, 3), 5.5],
        [(1, 2), (5, 5), (3, 5), 8.5],
        # Decreasing below zero:
        [(1, -2), (5, -5), (0, 6), -21],
        [(1, -2), (5, -5), (1, 5), -14],
        [(1, -2), (5, -5), (1, 3), -5.5],
        [(1, -2), (5, -5), (3, 5), -8.5],
        # Increasing below zero:
        [(1, -5), (5, -2), (0, 6), -21],
        [(1, -5), (5, -2), (1, 5), -14],
        [(1, -5), (5, -2), (1, 3), -8.5],
        [(1, -5), (5, -2), (3, 5), -5.5],
    ])
    def test_integrate(self, a: (int, float), b: (int, float), i: (int, int), expected: float):
        _f = pf(a, b)
        _i = TimeInterval(Time.unix(i[0]), Time.unix(i[1]))
        with self.subTest(f'{_f}.integrate({_i})'):
            self.assertEqual(expected, _f.integrate(_i))


class TestPiecewiseStepFunction(TestCase):
    @parameterized.expand([
        # Segment is zero:
        [(1, 0), (1, 0), (1, 1), 0],
        [(1, 0), (3, 0), (2, 2), 0],
        [(1, 1), (3, 1), (2, 2), 0],
        # Interval before segment:
        [(3, 4), (5, 6), (1, 3), 0],
        [(3, 4), (5, 6), (2, 3), 0],
        [(3, 4), (5, 6), (3, 3), 0],
        # Interval inside segment:
        [(3, 4), (5, 6), (3, 4), 4],
        [(3, 4), (5, 6), (4, 5), 4],
        [(3, 4), (5, 6), (3, 5), 8],
        # Interval after segment:
        [(3, 4), (5, 6), (5, 6), 6],
        [(3, 4), (5, 6), (5, 7), 12],
        [(3, 4), (5, 6), (5, 8), 18],
        # Interval over segment:
        [(3, 4), (5, 6), (3, 6), 14],
        [(3, 4), (5, 6), (3, 7), 20],
        [(3, 4), (5, 6), (2, 7), 20],
        [(3, 4), (5, 6), (1, 8), 26],
    ])
    def test_integrate(self, a: (int, float), b: (int, float), i: (int, int), expected: float):
        _f = pf(a, b, PiecewiseStepFunction)
        _i = TimeInterval(Time.unix(i[0]), Time.unix(i[1]))
        with self.subTest(f'{_f}.integrate({_i})'):
            self.assertEqual(expected, _f.integrate(_i))


class TestHMTSR(TestCase):
    def test_hmlpsa(self):
        res30 = TimeResolution(seconds=30)
        res60 = TimeResolution(seconds=60)
        res90 = TimeResolution(seconds=90)
        res120 = TimeResolution(seconds=120)
        res150 = TimeResolution(seconds=150)
        tree = TimeTree(res30, res60, res90, res120, res150)
        hmlpsa = HMTSR(tree, MultiSeriesDictionary())
        hmlpsa.process(TimeSeriesTuple(Time.unix(11), 6))
        hmlpsa.process(TimeSeriesTuple(Time.unix(44), 7))
        hmlpsa.process(TimeSeriesTuple(Time.unix(56), 1))
        hmlpsa.process(TimeSeriesTuple(Time.unix(146), 10))
        hmlpsa.process(TimeSeriesTuple(Time.unix(170), 7.26))

        with self.subTest('last'):
            self.assertEqual(TimeSeriesTuple(Time.unix(170), 7.26), hmlpsa.multiseries.last())

        with self.subTest(res30):
            actual = hmlpsa.multiseries(res30)
            expected = [TimeSeriesSegment(Time.unix(11), Time.unix(30), 19, 1, 6.29, 0.21, 0.09, 0.08),
                        TimeSeriesSegment(Time.unix(30), Time.unix(60), 30, 2, 4.93, 7.72, -5.13, 79.36),
                        TimeSeriesSegment(Time.unix(60), Time.unix(90), 30, 0, 2.9, 13.49, 70.15, 546.75),
                        TimeSeriesSegment(Time.unix(90), Time.unix(120), 30, 0, 5.9, 20.09, -16.07, 416.47),
                        TimeSeriesSegment(Time.unix(120), Time.unix(150), 30, 1, 8.84, 8.88, -55.59, 442.72),
                        TimeSeriesSegment(Time.unix(150), Time.unix(170), 20, 0, 8.4, 1.82, 0.83, 3.71)]
            self.assertEqual(expected, [round(a, 2) for a in actual])

        with self.subTest(res60):
            actual = hmlpsa.multiseries(res60)
            expected = [TimeSeriesSegment(Time.unix(11), Time.unix(60), 49, 3, 5.46, 5.25, -2.97, 48.85),
                        TimeSeriesSegment(Time.unix(60), Time.unix(120), 60, 0, 4.4, 19.04, 27.04, 486.67),
                        TimeSeriesSegment(Time.unix(120), Time.unix(150), 30, 1, 8.84, 8.88, -55.59, 442.72)]
            self.assertEqual(expected, [round(a, 2) for a in actual])

        with self.subTest(res90):
            actual = hmlpsa.multiseries(res90)
            expected = [TimeSeriesSegment(Time.unix(11), Time.unix(90), 79, 3, 4.48, 9.92, 23.85, 240.87),
                        TimeSeriesSegment(Time.unix(90), Time.unix(150), 60, 1, 7.37, 16.65, -35.83, 434.28)]
            self.assertEqual(expected, [round(a, 2) for a in actual])

        with self.subTest(res120):
            actual = hmlpsa.multiseries(res120)
            expected = [TimeSeriesSegment(Time.unix(11), Time.unix(120), 109, 3, 4.87, 13.12, 13.58, 289.93)]
            self.assertEqual(expected, [round(a, 2) for a in actual])

        with self.subTest(res150):
            actual = hmlpsa.multiseries(res150)
            expected = [TimeSeriesSegment(Time.unix(11), Time.unix(150), 139, 4, 5.73, 14.87, 4.3, 343.25)]
            self.assertEqual(expected, [round(a, 2) for a in actual])

    def test_unordered_tuples(self):
        ts1 = TimeSeriesTuple(Time.unix(11), 6)
        ts2 = TimeSeriesTuple(Time.unix(44), 7)
        hmlpsa = HMTSR(TimeTree(TimeResolution(seconds=30)), MultiSeriesDictionary())
        hmlpsa.process(ts1)
        hmlpsa.process(ts2)
        expected_segments = hmlpsa.multiseries(hmlpsa.resolutions.root)
        hmlpsa.process(ts1)
        actual_segments = hmlpsa.multiseries(hmlpsa.resolutions.root)

        with self.subTest('last'):
            self.assertEqual(ts2, hmlpsa.multiseries.last())

        with self.subTest('segments'):
            self.assertEqual(list(expected_segments), list(actual_segments))


class TestApproximator(TestCase):
    @parameterized.expand([
        [SeriesApproximator()],
        [StreamApproximator(MultiSeriesDictionary())]
    ])
    def test_fit_transform(self, a: Approximator):
        gds = GeneratedDataSource(y_count=1)
        ts = gds.dataset().y[0]
        ts_transformed = a.fit_transform(ts, TimeResolution(days=1), gds.interval)
        with self.subTest('columns'):
            self.assertEqual(['count', 'mean', 'variance', 'skewness', 'kurtosis'], list(ts_transformed.columns))
        with self.subTest('rows'):
            self.assertEqual(gds.interval.delta.days, len(ts_transformed))
        with self.subTest('count'):
            self.assertEqual(gds.interval.delta.days, round(ts_transformed['count'].sum()))
        with self.subTest('mean'):
            self.assertIsNot(0, ts_transformed['mean'].sum())


class TestPCA(TestCase):
    @parameterized.expand([[PCA()], [PCA(0.9)]])
    def test_extract(self, f: PCA):
        gds = GeneratedDataSource()
        x = gds(100).T
        x_extracted = f.fit_transform(x)
        summary = f.summary()
        with self.subTest(f'extracted {len(x_extracted)} rows'):
            self.assertEqual(len(x), len(x_extracted))
        with self.subTest(f'extracted {len(x_extracted.columns)} columns'):
            self.assertLess(len(x_extracted.columns), len(x.columns))
        with self.subTest('scores index'):
            self.assertEqual(len(x), len(f.scores))
        with self.subTest('summary index'):
            self.assertEqual(['feature'], list(summary.index.names))
        with self.subTest('summary columns'):
            self.assertEqual(['score', 'selected'], list(summary.columns))
        with self.subTest('summary rows'):
            assert_array_equal(f.scores.index, summary.index)


class TestJMIM(TestCase):
    def test_select(self):
        gds = GeneratedDataSource()
        x = gds(10)
        y = gds(2, 'Y')
        f = JMIM()
        x_selected = f.fit_transform(x, y)
        summary = f.summary()
        with self.subTest(f'selected {len(x_selected)} rows'):
            self.assertEqual(len(x), len(x_selected))
        with self.subTest(f'selected {len(x_selected.columns)} columns'):
            self.assertLess(len(x_selected.columns), len(x.columns))
        with self.subTest('scores index'):
            assert_array_equal(x.columns, f.scores.index)
        with self.subTest('scores columns'):
            assert_array_equal(y.columns, f.scores.columns)
        with self.subTest('summary index'):
            self.assertEqual(['x', 'y'], list(summary.index.names))
        with self.subTest('summary columns'):
            self.assertEqual(['score', 'selected'], list(summary.columns))
        with self.subTest('summary rows'):
            assert_array_equal(len(x.columns) * len(y.columns), len(summary))


class TestPACF(TestCase):
    @parameterized.expand([
        [None, 0.99],
        [24 * 7 * 3, 0.99],
        [24 * 7 * 2, 0.99],
    ])
    def test_recommend(self, max_lag: int, p: float):
        gds = GeneratedDataSource()
        x = gds(2)
        f = PACF(max_lag, p)
        lag = f.fit_predict(x)
        summary = f.summary()
        with self.subTest(f'lag(max_lag={max_lag}, p={p}) > 0'):
            self.assertGreater(lag, 0)
        with self.subTest('scores columns'):
            assert_array_equal(x.columns, f.scores.columns)
        with self.subTest('scores rows'):
            self.assertGreaterEqual(len(f.scores), lag)
        with self.subTest('summary index'):
            self.assertEqual(['feature', 'lag'], list(summary.index.names))
        with self.subTest('summary columns'):
            self.assertEqual(['score', 'selected', 'ci'], list(summary.columns))
        with self.subTest('summary rows'):
            self.assertEqual(len(f.scores) * len(x.columns), len(summary))

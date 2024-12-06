from itertools import product
from unittest import TestCase

import pandas as pd

from culearn.base import *
from numpy.testing import assert_array_equal
from parameterized import parameterized


class TestBase:
    @abstractmethod
    def subTest(self, msg=''):
        pass

    @abstractmethod
    def assertTrue(self, a, msg=''):
        pass

    @abstractmethod
    def assertEqual(self, a, b, msg=''):
        pass

    @abstractmethod
    def assertNotEqual(self, a, b, msg=''):
        pass

    @abstractmethod
    def assertIsNone(self, x, msg=''):
        pass


class TestEquatable(TestBase):
    @abstractmethod
    def samples(self) -> Iterable:
        pass

    @staticmethod
    def implies(a, b) -> bool:
        return not a or b

    @staticmethod
    def symmetry(a) -> bool:
        return a == a

    def reflexivity(self, a, b) -> bool:
        return self.implies(a == b, b == a)

    def transitivity(self, a, b, c) -> bool:
        return self.implies(a == b and b == c, a == c)

    def test_symmetry(self):
        for a in self.samples():
            with self.subTest(a):
                self.assertTrue(self.symmetry(a))
            with self.subTest(f'hash({a})'):
                self.assertTrue(self.symmetry(hash(a)))

    def test_reflexivity(self):
        for a, b in product(self.samples(), self.samples()):
            with self.subTest(f'{a} == {b}'):
                self.assertTrue(self.reflexivity(a, b))
            with self.subTest(f'hash({a}) == hash({b})'):
                self.assertTrue(self.reflexivity(hash(a), hash(b)))

    def test_transitivity(self):
        for a, b, c in product(self.samples(), self.samples(), self.samples()):
            with self.subTest(f'{a} == {b} == {c}'):
                self.assertTrue(self.transitivity(a, b, c))
            with self.subTest(f'hash({a}) == hash({b}) == hash({c})'):
                self.assertTrue(self.transitivity(hash(a), hash(b), hash(c)))

    def test_inequality(self):
        samples = list(self.samples())
        for i in range(len(samples)):
            for j in range(len(samples)):
                if i == j:
                    continue

                a = samples[i]
                b = samples[j]
                with self.subTest(f'{a} != {b}'):
                    self.assertNotEqual(a, b)
                with self.subTest(f'hash({a}) != hash({b})'):
                    self.assertNotEqual(hash(a), hash(b))

    def test_less_than(self):
        for a, b in product(self.samples(), self.samples()):
            with self.subTest(f'{a} < {b}'):
                self.assertTrue(self.implies(a < b, a == b or b > a))

    def test_greater_than(self):
        for a, b in product(self.samples(), self.samples()):
            with self.subTest(f'{a} > {b}'):
                self.assertTrue(self.implies(a > b, a == b or b < a))


class TestTimeInterval(TestCase, TestEquatable):
    def samples(self) -> Iterable:
        yield TimeInterval(Time.unix(1), Time.unix(1))
        yield TimeInterval(Time.unix(2), Time.unix(1))
        yield TimeInterval(Time.unix(1), Time.unix(2))

    @parameterized.expand([
        [Time.unix(0), False],
        [Time.unix(1), True],
        [Time.unix(2), True],
        [Time.unix(3), False],
        [TimeInterval(Time.unix(0), Time.unix(1)), False],
        [TimeInterval(Time.unix(0), Time.unix(3)), False],
        [TimeInterval(Time.unix(1), Time.unix(3)), True],
        [TimeInterval(Time.unix(1), Time.unix(2)), True],
        [TimeInterval(Time.unix(2), Time.unix(3)), True],
        [TimeInterval(Time.unix(3), Time.unix(1)), True],
        [TimeInterval(Time.unix(1), Time.unix(4)), False],
        [TimeInterval(Time.unix(3), Time.unix(4)), False],
        [TimeInterval(Time.unix(0), Time.unix(4)), False],
    ])
    def test_contains(self, other: any, expected: bool):
        i = TimeInterval(Time.unix(1), Time.unix(3))
        self.assertEqual(expected, i.contains(other))

    @parameterized.expand([
        [(1, 1), (1, 1), (1, 1)],
        [(1, 3), (2, 4), (2, 3)],
        [(1, 6), (2, 4), (2, 4)],
        [(1, 2), (2, 3), None],
        [(1, 2), (4, 5), None],
    ])
    def test_overlap(self, a: (int, int), b: (int, int), expected: (int, int)):
        def _assert(_a: TimeInterval, _b: TimeInterval, _c: TimeInterval):
            with self.subTest(f'{_a} vs {_b}'):
                self.assertEqual(_c, _a.overlap(_b))

        def _test_all(_a: TimeInterval, _b: TimeInterval, _c: TimeInterval):
            _assert(_a, _b, _c)
            _assert(_b, _a, _c)
            _assert(TimeInterval(_a.end, _a.start), _b, _c)
            _assert(TimeInterval(_b.end, _b.start), _a, _c)
            _assert(_a, TimeInterval(_b.end, _b.start), _c)
            _assert(_b, TimeInterval(_a.end, _a.start), _c)
            _assert(TimeInterval(_a.end, _a.start), TimeInterval(_b.end, _b.start), _c)
            _assert(TimeInterval(_b.end, _b.start), TimeInterval(_a.end, _a.start), _c)

        _test_all(TimeInterval(Time.unix(a[0]), Time.unix(a[1])),
                  TimeInterval(Time.unix(b[0]), Time.unix(b[1])),
                  None if expected is None else TimeInterval(Time.unix(expected[0]), Time.unix(expected[1])))


class TestTimeResolution(TestCase, TestEquatable):
    def samples(self) -> Iterable:
        yield TimeResolution(1)
        yield TimeResolution(2)

    @parameterized.expand([
        [1800, Time.unix(0), TimeInterval(Time.unix(0), Time.unix(1800))],
        [1800, Time.unix(2000), TimeInterval(Time.unix(1800), Time.unix(3600))],
        [3600, Time.unix(0), TimeInterval(Time.unix(0), Time.unix(3600))],
        [3600, Time.unix(2000), TimeInterval(Time.unix(0), Time.unix(3600))],
        [3600, Time.unix(4000), TimeInterval(Time.unix(3600), Time.unix(7200))],
    ])
    def test_step(self, seconds: float, t: datetime, expected: TimeInterval):
        r = TimeResolution(seconds=seconds)
        actual = r.step(t)
        with self.subTest(f'{r}.step({t})'):
            self.assertEqual(expected, actual)

    @parameterized.expand([
        [3600, TimeInterval(Time.unix(2000), Time.unix(3600)), [TimeInterval(Time.unix(0), Time.unix(3600))]],
        [3600, TimeInterval(Time.unix(2000), Time.unix(4000)), [TimeInterval(Time.unix(0), Time.unix(3600)),
                                                                TimeInterval(Time.unix(3600), Time.unix(7200))]],
    ])
    def test_steps(self, seconds: float, i: TimeInterval, expected: Sequence[TimeInterval]):
        r = TimeResolution(seconds=seconds)
        actual = list(r.steps(i))
        with self.subTest(f'{r}.steps({i})'):
            self.assertEqual(expected, actual)


class TestTimeTree(TestCase, TestBase):
    @parameterized.expand([
        [[1], 1, {}],
        [[1, 1, 2, 2], 1, {1: [2]}],
        [[5, 10, 15, 20], 5, {5: [10, 15], 10: [20]}],
        [[10, 15, 20], 5, {5: [10, 15], 10: [20]}],
    ])
    def test_init(self,
                  seconds: Sequence[int],
                  expected_root: int,
                  expected_tree: Mapping[int, Sequence[int]]):
        resolutions = [TimeResolution(seconds=s) for s in seconds]
        tree = TimeTree(*resolutions)

        with self.subTest(f'root({seconds})'):
            self.assertEqual(TimeResolution(seconds=expected_root), tree.root)

        for parent, children in expected_tree.items():
            with self.subTest(f'children({parent})'):
                expected_children = {TimeResolution(seconds=child) for child in children}
                self.assertEqual(expected_children, set(tree(TimeResolution(seconds=parent))))

        with self.subTest(f'iter'):
            self.assertSetEqual(set(resolutions + [tree.root]), set(tree))


class TestTimeSeriesID(TestCase, TestEquatable):
    def samples(self) -> Iterable:
        yield TimeSeriesID('A')
        yield TimeSeriesID('A', 'A')
        yield TimeSeriesID('A', 'A', 'A')
        yield TimeSeriesID('B', 'A', 'A')
        yield TimeSeriesID('A', 'B', 'A')
        yield TimeSeriesID('A', 'A', 'B')


class TestTimeSeriesTuple(TestCase, TestEquatable):
    def samples(self) -> Iterable:
        yield TimeSeriesTuple(Time.unix(1), 1)
        yield TimeSeriesTuple(Time.unix(2), 1)
        yield TimeSeriesTuple(Time.unix(1), 2)


class TestTimeSeriesSegment(TestCase, TestEquatable):
    def samples(self) -> Iterable:
        yield TimeSeriesSegment(Time.unix(1), Time.unix(1), 1, 1, 1, 1, 1, 1)
        yield TimeSeriesSegment(Time.unix(2), Time.unix(1), 1, 1, 1, 1, 1, 1)
        yield TimeSeriesSegment(Time.unix(1), Time.unix(2), 1, 1, 1, 1, 1, 1)
        yield TimeSeriesSegment(Time.unix(1), Time.unix(1), 2, 1, 1, 1, 1, 1)
        yield TimeSeriesSegment(Time.unix(1), Time.unix(1), 1, 2, 1, 1, 1, 1)
        yield TimeSeriesSegment(Time.unix(1), Time.unix(1), 1, 1, 2, 1, 1, 1)
        yield TimeSeriesSegment(Time.unix(1), Time.unix(1), 1, 1, 1, 2, 1, 1)
        yield TimeSeriesSegment(Time.unix(1), Time.unix(1), 1, 1, 1, 1, 2, 1)
        yield TimeSeriesSegment(Time.unix(1), Time.unix(1), 1, 1, 1, 1, 1, 2)


class TestTimeSeriesInMemory(TestCase):
    def test_index(self):
        values = pd.Series(range(10), pd.DatetimeIndex(range(10)))
        ts = TimeSeriesInMemory(TimeSeriesID(), values)
        assert_array_equal(values, ts.series())

    def test_series(self):
        values = pd.Series(range(10))
        self.assertRaises(Exception, lambda: TimeSeriesInMemory(TimeSeriesID(), values))

    def test_stream(self):
        values = [3, 4]
        timestamps = [datetime(2020, 1, 1), datetime(2021, 1, 1)]
        ts = TimeSeriesInMemory(TimeSeriesID(), pd.Series(values, pd.DatetimeIndex(timestamps)))
        tuples = list(ts.stream())
        self.assertEqual(len(values), len(tuples), "Number of tuples.")
        self.assertEqual(values[0], tuples[0].value, "First value.")
        self.assertEqual(values[1], tuples[1].value, "Second value.")
        self.assertEqual(timestamps[0], tuples[0].timestamp, "First timestamp.")
        self.assertEqual(timestamps[1], tuples[1].timestamp, "Second timestamp.")


class TestTimeSeriesPrediction(TestCase):
    def test_to_frame(self):
        p = TimeSeriesPrediction(TimeSeriesID('A', 'B', 'C'))
        p.append(PredictionInterval(0.5, pd.Series([1, 2]), pd.Series([3, 4])))
        p.append(PredictionInterval(0.9, pd.Series([5, 6]), pd.Series([7, 8])))
        actual = p.to_frame()
        expected = pd.DataFrame([[1, 3, 5, 7],
                                 [2, 4, 6, 8]],
                                columns=['Lower 50% PI', 'Upper 50% PI', 'Lower 90% PI', 'Upper 90% PI'])
        with self.subTest('ID'):
            self.assertEqual(p.ts_id, actual.ts_id)
        with self.subTest('data'):
            assert_array_equal(expected, actual)

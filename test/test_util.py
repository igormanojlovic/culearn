from unittest import TestCase

from culearn.util import *
from numpy.testing import assert_array_almost_equal
from parameterized import parameterized


class TestMath(TestCase):
    @staticmethod
    def a(x: float, rep=2) -> np.array:
        return np.array([x] * rep)

    def d(self, x: float, rep=2) -> np.array:
        return pd.DataFrame(self.a(x, rep))

    def s(self, x: float, rep=2) -> np.array:
        return pd.Series(self.a(x, rep))

    @parameterized.expand([
        [0.1, 50, 100, 10, 5, 1, 87.65],
        [0.5, 50, 100, 10, 5, 1, 98.82],
        [0.9, 50, 100, 10, 5, 1, 113.86],
        [0.1, 50, 100, 10, -5, 1, 86.14],
        [0.5, 50, 100, 10, -5, 1, 101.18],
        [0.9, 50, 100, 10, -5, 1, 112.35],
    ])
    def test_qcf(self, p: float, n: int, mean: float, stdev: float, k3: float, k4: float, q: float):
        with self.subTest(f'qcf({p}, {n}, {mean}, {stdev}, {k3}, {k4}) with scalars'):
            self.assertAlmostEqual(q, Math.qcf(p, n, mean, stdev, k3, k4), 2, 'float')
        for f in [self.a, self.d, self.s]:
            with self.subTest(f'qcf({p}, {n}, {mean}, {stdev}, {k3}, {k4}) with {type(f(0)).__name__}'):
                assert_array_almost_equal(f(q), Math.qcf(p, f(n), f(mean), f(stdev), f(k3), f(k4)), 2, str(f))

    def test_qcf_quantile_order(self):
        q = [Math.qcf(_ / 100, 10, 1.23, 0.99, 0.88, 0.77) for _ in range(1, 100)]
        self.assertEqual(q, list(sorted(q)))

    @parameterized.expand([
        [0.1, 95, 100, 4.5],
        [0.1, 105, 100, 0.5],
        [0.9, 95, 100, 0.5],
        [0.9, 105, 100, 4.5],
    ])
    def test_pinball(self, p: float, x: float, q: float, e: float):
        with self.subTest(f'pinball({p}, {x}, {q}) with scalars'):
            self.assertAlmostEqual(e, Math.pinball(p, x, q), 2, 'float')
        for f in [self.a, self.d, self.s]:
            with self.subTest(f'pinball({p}, {x}, {q}) with {type(f(0)).__name__}'):
                assert_array_almost_equal(f(e), Math.pinball(p, f(x), f(q)), 2, str(f))

    @parameterized.expand([
        [0.95, 100, 95, 105, 10],
        [0.95, 95, 100, 105, 205],
        [0.95, 105, 95, 100, 205],
    ])
    def test_winkler(self, p: float, x: float, l: float, u: float, e: float):
        with self.subTest(f'winkler({p}, {x}, {l}, {u}) with scalars'):
            self.assertAlmostEqual(e, Math.winkler(p, x, l, u), 2, 'float')

        for f in [self.a, self.d, self.s]:
            with self.subTest(f'winkler({p}, {x}, {l}, {u}) with {type(f(0)).__name__}'):
                assert_array_almost_equal(f(e), Math.winkler(p, f(x), f(l), f(u)), 2, str(f))

from unittest import TestCase

from culearn.data import *
from culearn.regression import *
from numpy.testing import assert_array_equal
from parameterized import parameterized


class TestKerasModel(TestCase):
    @parameterized.expand([
        [DCNN(depth=0, n_filters=16, shape=2, avg=True), layers.Input(shape=(128, 128))],
        [DCNN(depth=1, n_filters=16, shape=2, avg=True), layers.Input(shape=(128, 128))],
        [DCNN(depth=2, n_filters=16, shape=2, avg=True), layers.Input(shape=(128, 128))],
        [DCNN(depth=3, n_filters=16, shape=2, avg=True), layers.Input(shape=(128, 128))],
        [DCNN(depth=0, n_filters=16, shape=(2, 2), avg=True), layers.Input(shape=(128, 128, 128))],
        [DCNN(depth=1, n_filters=16, shape=(2, 2), avg=True), layers.Input(shape=(128, 128, 128))],
        [DCNN(depth=2, n_filters=16, shape=(2, 2), avg=True), layers.Input(shape=(128, 128, 128))],
        [DCNN(depth=3, n_filters=16, shape=(2, 2), avg=True), layers.Input(shape=(128, 128, 128))],
        [DCNN(depth=0, n_filters=16, shape=(2, 2, 2), avg=True), layers.Input(shape=(128, 128, 128, 128))],
        [DCNN(depth=1, n_filters=16, shape=(2, 2, 2), avg=True), layers.Input(shape=(128, 128, 128, 128))],
        [DCNN(depth=2, n_filters=16, shape=(2, 2, 2), avg=True), layers.Input(shape=(128, 128, 128, 128))],
        [DCNN(depth=3, n_filters=16, shape=(2, 2, 2), avg=True), layers.Input(shape=(128, 128, 128, 128))],
        [DCNN(depth=0, n_filters=16, shape=2, avg=False), layers.Input(shape=(128, 128))],
        [DCNN(depth=1, n_filters=16, shape=2, avg=False), layers.Input(shape=(128, 128))],
        [DCNN(depth=2, n_filters=16, shape=2, avg=False), layers.Input(shape=(128, 128))],
        [DCNN(depth=3, n_filters=16, shape=2, avg=False), layers.Input(shape=(128, 128))],
        [DCNN(depth=0, n_filters=16, shape=(2, 2), avg=False), layers.Input(shape=(128, 128, 128))],
        [DCNN(depth=1, n_filters=16, shape=(2, 2), avg=False), layers.Input(shape=(128, 128, 128))],
        [DCNN(depth=2, n_filters=16, shape=(2, 2), avg=False), layers.Input(shape=(128, 128, 128))],
        [DCNN(depth=3, n_filters=16, shape=(2, 2), avg=False), layers.Input(shape=(128, 128, 128))],
        [DCNN(depth=0, n_filters=16, shape=(2, 2, 2), avg=False), layers.Input(shape=(128, 128, 128, 128))],
        [DCNN(depth=1, n_filters=16, shape=(2, 2, 2), avg=False), layers.Input(shape=(128, 128, 128, 128))],
        [DCNN(depth=2, n_filters=16, shape=(2, 2, 2), avg=False), layers.Input(shape=(128, 128, 128, 128))],
        [DCNN(depth=3, n_filters=16, shape=(2, 2, 2), avg=False), layers.Input(shape=(128, 128, 128, 128))],
        [SAE(depth=0, edge=64, middle=1), layers.Input(shape=128)],
        [SAE(depth=1, edge=64, middle=1), layers.Input(shape=128)],
        [SAE(depth=2, edge=64, middle=1), layers.Input(shape=128)],
        [SAE(depth=3, edge=64, middle=1), layers.Input(shape=128)],
        [SAE(depth=0, edge=64, middle=4), layers.Input(shape=128)],
        [SAE(depth=1, edge=64, middle=4), layers.Input(shape=128)],
        [SAE(depth=2, edge=64, middle=4), layers.Input(shape=128)],
        [SAE(depth=3, edge=64, middle=4), layers.Input(shape=128)],
        [S2S_LSTM(sae_depth=0), [layers.Input(shape=(128, 4)), layers.Input(shape=(64, 16))]],
        [S2S_LSTM(sae_depth=1), [layers.Input(shape=(128, 4)), layers.Input(shape=(64, 16))]],
        [S2S_LSTM(sae_depth=2), [layers.Input(shape=(128, 4)), layers.Input(shape=(64, 16))]],
        [S2S_LSTM(sae_depth=3), [layers.Input(shape=(128, 4)), layers.Input(shape=(64, 16))]],
        [S2S_GRU(sae_depth=0), [layers.Input(shape=(128, 4)), layers.Input(shape=(64, 16))]],
        [S2S_GRU(sae_depth=1), [layers.Input(shape=(128, 4)), layers.Input(shape=(64, 16))]],
        [S2S_GRU(sae_depth=2), [layers.Input(shape=(128, 4)), layers.Input(shape=(64, 16))]],
        [S2S_GRU(sae_depth=3), [layers.Input(shape=(128, 4)), layers.Input(shape=(64, 16))]],
        [DTSN(), [layers.Input(shape=(168, 1)), layers.Input(shape=(24, 10))]],
        [DTSN(), [layers.Input(shape=(168, 2)), layers.Input(shape=(24, 10))]],
        [DTSN(), [layers.Input(shape=(168, 4)), layers.Input(shape=(24, 10))]]
    ])
    def test_call(self, model: keras.Model, inputs):
        with self.subTest(type(model).__name__):
            self.assertIsInstance(model.call(inputs), KerasTensor)


class TestS2S(TestCase):
    @staticmethod
    def select(df: pd.DataFrame, interval: TimeInterval) -> pd.DataFrame:
        return df[interval.contains(df.index)]

    def test_transform(self):
        gds = GeneratedDataSource()
        x = gds(2)
        x_past, x_future = S2S.transform(x.to_numpy(), 48, 24)

        with self.subTest('shape of past'):
            self.assertEqual((8689, 48, 2), np.shape(x_past))

        with self.subTest('shape of future'):
            self.assertEqual((8689, 24, 2), np.shape(x_future))

        for i_series in range(2):
            with self.subTest(f'first lookback in series {i_series}'):
                sub_interval = TimeInterval(gds.interval.start, gds.interval.start + timedelta(2))
                assert_array_equal(self.select(x, sub_interval).iloc[:, i_series], x_past[0, :, i_series])

            with self.subTest(f'first horizon in series {i_series}'):
                sub_interval = TimeInterval(gds.interval.start + timedelta(2), gds.interval.start + timedelta(3))
                assert_array_equal(self.select(x, sub_interval).iloc[:, i_series], x_future[0, :, i_series])

            with self.subTest(f'last lookback in series {i_series}'):
                sub_interval = TimeInterval(gds.interval.end - timedelta(3), gds.interval.end - timedelta(1))
                assert_array_equal(self.select(x, sub_interval).iloc[:, i_series], x_past[-1, :, i_series])

            with self.subTest(f'last horizon in series {i_series}'):
                sub_interval = TimeInterval(gds.interval.end - timedelta(1), gds.interval.end)
                assert_array_equal(self.select(x, sub_interval).iloc[:, i_series], x_future[-1, :, i_series])


class TestTimeSeriesRegressor(TestCase):
    @parameterized.expand([
        [TimeSeriesRegressor(24, DeepS2S(epochs=2))],
        [TimeSeriesRegressor(24 * 14, DeepS2S(epochs=1))],
        [TimeSeriesRegressor(24, ShallowS2S('eSVM', True))],
        [TimeSeriesRegressor(24, ShallowS2S('nSVM', True))],
        [TimeSeriesRegressor(24, ShallowS2S('XGB', True))],
        [TimeSeriesRegressor(24, ShallowS2S(lambda: SVR(), True))],
        [TimeSeriesRegressor(24 * 14, ShallowS2S('eSVM', True))],
    ])
    def test_fit_predict(self, f: TimeSeriesRegressor):
        gds = GeneratedDataSource()
        x = gds(10)
        y = gds(2, 'Y')

        with self.subTest(f'{f}: predict'):
            self.assertRaises(Exception, lambda: f.predict(x, y))

        x_train = x.iloc[:-f.horizon, :]
        y_train = y.iloc[:-f.horizon, :]

        with self.subTest(f'{f}: fit'):
            self.assertEqual(f, f.fit(x_train, y_train))

        with self.subTest(f'{f}: history'):
            self.assertEqual(1, len(f.history))

        x_test = x.iloc[-f.horizon:, :]
        y_test = y.iloc[-f.horizon - f.lookback:-f.horizon, :]
        y_true = y.iloc[-f.horizon:, :]

        with self.subTest(f'{f}: len(x) < horizon ({f.horizon})'):
            self.assertEqual(0, len(f.predict(x_test.iloc[:-1, :], y_test)))

        with self.subTest(f'{f}: len(y) < lookback ({f.lookback})'):
            self.assertEqual(0, len(f.predict(x_test, y_test.iloc[:-1, :])))

        with self.subTest(f'{f}: predict'):
            y_pred = f.predict(x_test, y_test)
            self.assertEqual(y_true.shape, y_pred.shape)

        x_update = x.iloc[-f.horizon - f.lookback:, :]
        y_update = y.iloc[-f.horizon - f.lookback:, :]

        with self.subTest(f'{f}: update'):
            self.assertEqual(f, f.fit(x_update, y_update))

        with self.subTest(f'{f}: history'):
            self.assertEqual(2, len(f.history))

        with self.subTest(f'{f}: predict'):
            y_pred = f.predict(x_test, y_test)
            self.assertEqual(y_true.shape, y_pred.shape)

        summary = f.summary()
        with self.subTest(f'{f}: summary'):
            self.assertEqual(len(summary), sum([len(h) for h in f.history]))


class TestTimeSeriesGroupRegressor(TestCase):
    @parameterized.expand([
        [TimeSeriesRegressor(24, DeepS2S(epochs=1))],
        [TimeSeriesRegressor(24, ShallowS2S())],
    ])
    def test_fit_predict(self, base: TimeSeriesRegressor):
        gds = GeneratedDataSource()
        x = gds(10)
        y1 = gds(2)
        y2 = gds(2)
        f = TimeSeriesGroupRegressor(base)

        with self.subTest(f'{f}: fit'):
            x_train = x.iloc[:-base.horizon, :]
            y_train = [y1.iloc[:-base.horizon, :],
                       y2.iloc[:-base.horizon, :]]
            self.assertEqual(f, f.fit(x_train, y_train))

        with self.subTest(f'{f}: predict'):
            x_test = x.iloc[-base.horizon:, :]
            y_test = [y1.iloc[-base.horizon - base.lookback:-base.horizon, :],
                      y2.iloc[-base.horizon - base.lookback:-base.horizon, :]]
            y_pred = f.predict(x_test, y_test)
            self.assertEqual([y1.iloc[-base.horizon:, :].shape,
                              y2.iloc[-base.horizon:, :].shape],
                             [y_pred[0].shape,
                              y_pred[1].shape])

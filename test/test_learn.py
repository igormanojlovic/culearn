from unittest import TestCase

from culearn.data import *
from culearn.learn import *
from culearn.regression import *
from numpy.testing import assert_array_equal, assert_array_almost_equal


def assert_cumulants(cumulants: Iterable[TimeSeriesDataFrame], ids: Iterable, index: Iterable):
    missing = set(ids)
    columns = ['mean', 'stdev', 'skewness', 'kurtosis']
    for c in cumulants:
        missing.remove(c.ts_id)
        assert_array_equal(columns, c.columns, f'Columns of {c.ts_id}')
        assert_array_equal(index, c.index, f'Index of {c.ts_id}')

    assert_array_equal(missing, [], 'missing')


def assert_intervals(intervals: Iterable[TimeSeriesPrediction], ids: Iterable, p: Iterable, index: Iterable):
    missing = set(ids)
    for i in intervals:
        missing.remove(i.ts_id)
        assert_array_equal(p, [_.p for _ in i], f'p for {i.ts_id}')
        for pi in i:
            assert_array_equal(index, pi.lower.index, f'Index of lower {p} interval for {i.ts_id}')
            assert_array_equal(index, pi.upper.index, f'Index of upper {p} interval for {i.ts_id}')

    assert_array_equal(missing, [], 'missing')


class RegressorCounter:
    fit = 0
    predict = 0

    def count_fit(self):
        self.fit += 1

    def count_predict(self):
        self.predict += 1

    def reset(self):
        self.fit = 0
        self.predict = 0


class FakeS2S(S2S):
    def __init__(self, counter: RegressorCounter):
        self.counter = counter

    def compile(self, x_count: int, y_count: int, lookback: int, horizon: int) -> any:
        return self

    def fit(self, model, x_future: np.array, y_past: np.array, y_future: np.array) -> pd.DataFrame:
        model.counter.count_fit()
        return pd.DataFrame({'fake_metric': [1, 2, 3]}).rename_axis('fake_index')

    def predict(self, model, x_future: np.array, y_past: np.array) -> np.array:
        model.counter.count_predict()
        return np.empty((np.shape(y_past)[0], np.shape(x_future)[1], np.shape(y_past)[2]))


class TestCumulantTransform(TestCase):
    def test_functions(self):
        gds = GeneratedDataSource(y_id = lambda i: TimeSeriesID(str(i), str(i % 2), str(int(i % 4 / 2))))
        ds = gds.dataset()
        x = ds.y
        f = CumulantTransform()
        p = [0.75, 0.95, 0.99]
        p_quantiles = [0.005, 0.025, 0.125, 0.875, 0.975, 0.995]
        index = ds.x.index

        cumulants = f.fit_transform(x, gds.resolution, gds.interval)

        with self.subTest('clusters'):
            self.assertLess(len(f.clusters), len(x))
            self.assertGreater(len(f.clusters), 0)

        with self.subTest('members'):
            self.assertSetEqual(set([_.ts_id for _ in x]), set(chain.from_iterable(f.clusters.values())))

        with self.subTest('factors'):
            self.assertSetEqual(set([_.ts_id for _ in x]), set(f.factors.keys()))

        with self.subTest('cumulants'):
            assert_cumulants(cumulants, f.clusters.keys(), index)

        with self.subTest('inverse: clusters'):
            intervals = f.inverse_transform(cumulants, p, clusters=True, members=False)
            assert_intervals(intervals, f.clusters.keys(), p, index)

        with self.subTest('inverse: members'):
            intervals = f.inverse_transform(cumulants, p, clusters=False, members=True)
            assert_intervals(intervals, [_.ts_id for _ in x], p, index)

        with self.subTest('inverse: all'):
            intervals = f.inverse_transform(cumulants, p, clusters=True, members=True)
            assert_intervals(intervals, list(f.clusters.keys()) + [_.ts_id for _ in x], p, index)

        pred_lines_count = len(f.clusters) * len(p) * 2

        with self.subTest('figure: clusters'):
            figure = f.figure(cumulants, p=p)
            self.assertEqual(pred_lines_count, len(figure.data))

        with self.subTest('figure: all'):
            figure = f.figure(cumulants, p=p, show_actual=True)
            self.assertEqual(len(x) + pred_lines_count, len(figure.data))

        with self.subTest('evaluate'):
            ps, ws = f.evaluate(cumulants, p)
            self.assertSetEqual(set([_.ts_id for _ in x]), set(ps.index), 'pinball score: index')
            assert_array_equal(ps.index, ws.index, 'winkler score: index')
            assert_array_almost_equal(p_quantiles, ps.columns, 3, 'pinball score: columns')
            assert_array_equal(p, ws.columns, 'winkler score: columns')

        name_score_index_columns = [
            ['clustering_score', f.clustering_score, ['category', 'k'], ['score', 'selected', 'cardinality']],
            ['extractor_score', f.extractor_score, ['category', 'feature'], ['score', 'selected']],
        ]

        for score_name, score_df, score_index, score_columns in name_score_index_columns:
            with self.subTest(f'evaluate: {score_name} index'):
                self.assertEqual(score_index, list(score_df.index.names))
            with self.subTest(f'evaluate: {score_name} columns'):
                self.assertEqual(score_columns, list(score_df.columns))


class TestCumulantLearner(TestCase):
    @ignore_warnings
    def test_functions(self):
        horizon = 24
        gds = GeneratedDataSource(y_id = lambda i: TimeSeriesID(str(i), str(i % 2), str(int(i % 4 / 2))))
        counter = RegressorCounter()

        f = CumulantLearner(gds.dataset(),
                            gds.resolution,
                            CumulantTransform(),
                            lambda: TimeSeriesRegressor(horizon,
                                                        base=FakeS2S(counter),
                                                        x_selector=lambda: DummyFeatureSelector(),
                                                        y_selector=lambda: DummyLagSelector(48)))

        update_step = 5
        update_count = 2
        fit_interval = TimeInterval(gds.interval.start, gds.interval.end - timedelta(update_count * update_step))
        pred_interval = TimeInterval(fit_interval.end, gds.interval.end)
        eval_index = f.dataset.x[pred_interval.contains(f.dataset.x.index)].index
        pred_index = eval_index[:horizon]
        p = [0.75, 0.95, 0.99]
        p_quantiles = [0.005, 0.025, 0.125, 0.875, 0.975, 0.995]

        with self.subTest('fit'):
            self.assertEqual(f, f.fit(fit_interval), 'return')
            regressors_count = len({_.source for _ in f.transformer.clusters.keys()})
            self.assertEqual(regressors_count, counter.fit, 'calls')

        with self.subTest('update'):
            self.assertEqual(f, f.update(pred_interval), 'return')
            self.assertEqual(2 * regressors_count, counter.fit, 'calls')

        with self.subTest('predict_cumulants'):
            cumulants = f.predict_cumulants(pred_interval.start)
            self.assertEqual(regressors_count, counter.predict, 'calls')
            assert_cumulants(cumulants, f.transformer.clusters.keys(), pred_index)

        with self.subTest('predict: clusters'):
            intervals = f.predict(pred_interval.start, p, clusters=True, members=False)
            self.assertEqual(2 * regressors_count, counter.predict, 'calls')
            assert_intervals(intervals, f.transformer.clusters.keys(), p, pred_index)

        with self.subTest('predict: members'):
            intervals = f.predict(pred_interval.start, p, clusters=False, members=True)
            self.assertEqual(3 * regressors_count, counter.predict, 'calls')
            assert_intervals(intervals, [_.ts_id for _ in f.dataset.y], p, pred_index)

        with self.subTest('predict: all'):
            intervals = f.predict(pred_interval.start, p, clusters=True, members=True)
            self.assertEqual(4 * regressors_count, counter.predict, 'calls')
            expected_ids = list(f.transformer.clusters.keys()) + [_.ts_id for _ in f.dataset.y]
            assert_intervals(intervals, expected_ids, p, pred_index)

        counter.reset()
        e = f.evaluate(fit_interval, pred_interval, update_step, p)

        with self.subTest('evaluate: update'):
            self.assertEqual(regressors_count * update_count, counter.fit, 'calls')

        with self.subTest('evaluate: predict'):
            self.assertEqual(regressors_count * pred_interval.delta.days, counter.predict, 'calls')

        name_score_index_columns = [
            ['clustering_score', e.clustering_score, ['category', 'k'], ['score', 'selected', 'cardinality']],
            ['extractor_score', e.extractor_score, ['category', 'feature'], ['score', 'selected']],
            ['x_selector_score', e.x_selector_score, ['cluster', 'x', 'y'], ['score', 'selected']],
            ['y_selector_score', e.y_selector_score, ['cluster', 'feature', 'lag'], ['score', 'selected', 'ci']],
            ['regressor_score', e.regressor_score, ['cluster', 'fit', 'fake_index'], ['fake_metric']],
            ['winkler_score', e.winkler_score, ['member'], p],
        ]

        for score_name, score_df, score_index, score_columns in name_score_index_columns:
            with self.subTest(f'evaluate: {score_name} index'):
                self.assertEqual(score_index, list(score_df.index.names))
            with self.subTest(f'evaluate: {score_name} columns'):
                self.assertEqual(score_columns, list(score_df.columns))

        with self.subTest(f'evaluate: pinball_score index'):
            self.assertEqual(['member'], list(e.pinball_score.index.names))
        with self.subTest(f'evaluate: pinball_score columns'):
            assert_array_almost_equal(p_quantiles, list(e.pinball_score.columns), 3)

        counter.reset()
        f = CumulantLearner(f.dataset, f.resolution, CumulantTransform(), f.regressor)
        f.evaluate(fit_interval, pred_interval, update_step, p)
        new_regressors_count = len({_.source for _ in f.transformer.clusters.keys()})

        with self.subTest('evaluate: fit and update'):
            self.assertEqual(new_regressors_count * (1 + update_count), counter.fit, 'calls')

        with self.subTest('evaluate: predict'):
            self.assertEqual(new_regressors_count * pred_interval.delta.days, counter.predict, 'calls')

        pred_lines_count = len(f.transformer.clusters) * len(p) * 2

        with self.subTest('figure: clusters'):
            figure = f.figure(pred_interval.start, p=p)
            self.assertEqual(pred_lines_count, len(figure.data))

        with self.subTest('figure: all'):
            figure = f.figure(pred_interval.start, p=p, show_actual=True)
            self.assertEqual(len(f.dataset.y) + pred_lines_count, len(figure.data))

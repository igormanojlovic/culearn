from unittest import TestCase

from culearn.clustering import *
from culearn.data import *
from parameterized import parameterized


class TestClustering(TestCase):
    @parameterized.expand([
        [TSGA()],
        [TSGA(km_method=None)],
        [TSGA(km_method='kmeans')],
        [TSGA(km_method='kmeans++')],
        [TSGA(km_method='bisect')],
        [TSGA(km_method='bisect++')],
        [TSGA(km_method='hartigan')],
        [TSGA(km_method='kshape')],
        [TSGA(hc_method='average')],
        [TSGA(hc_method='ward')],
        [TSGA(score='CH')],
        [TSGA(score='DB')],
        [TSGA(score='S')],
        [TSGA(score='CH', best=max)],
        [TSGA(score='DB', best=min)],
        [TSGA(score='S', best=max)],
        [TSGA(score=calinski_harabasz_score, best=max)],
        [TSGA(score=davies_bouldin_score, best=min)],
        [TSGA(score=silhouette_score, best=max)],
    ])
    def test_cluster(self, f: Clustering):
        gds = GeneratedDataSource()
        x = gds(100).T

        y_pred = f.fit_predict(x)
        summary = f.summary()
        with self.subTest(f'{f}: y reference'):
            self.assertIs(f.y, y_pred)
        with self.subTest(f'{f}: y length'):
            self.assertEqual(f.k, len(set(f.y)))
        with self.subTest(f'{f}: k >= k_min'):
            self.assertGreaterEqual(f.k, min(f.k2score.keys()))
        with self.subTest(f'{f}: k <= k_max'):
            self.assertLessEqual(f.k, max(f.k2score.keys()))
        with self.subTest(f'{f}: k score'):
            self.assertEqual(f.k2score[f.k], f.best(list(f.k2score.values())))
        with self.subTest(f'{f}: cardinality vs objects'):
            self.assertSetEqual({len(x)}, {sum(c) for c in f.k2cardinality.values()})
        with self.subTest(f'{f}: cardinality vs score'):
            self.assertSetEqual(set(f.k2score.keys()), set(f.k2cardinality.keys()))
        with self.subTest(f'{f}: summary index'):
            self.assertEqual(['k'], list(summary.index.names))
        with self.subTest(f'{f}: summary columns'):
            self.assertEqual(['score', 'selected', 'cardinality'], list(summary.columns))
        with self.subTest(f'{f}: summary rows'):
            self.assertEqual(len(f.k2score), len(summary))

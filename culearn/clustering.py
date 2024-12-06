from collections import Counter
from math import floor

from hkmeans import HKMeans
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from tslearn.clustering import KShape

from culearn.features import *
from culearn.util import ignore_warnings, parallel


class Clustering(StrMixin):
    __score_functions = {
        'ch': calinski_harabasz_score,
        'db': davies_bouldin_score,
        's': silhouette_score,
    }

    def __init__(self,
                 k_min: 'int > 1',
                 k_max: Optional['int > 1'],
                 extractor: Callable[[], FeatureExtractor],
                 score: Union[str, Callable[[pd.DataFrame, pd.Series], float]],
                 best: Callable[[Sequence[float]], float]):
        """
        Abstract search for an optimal number of clusters.

        :param k_min: Minimum number of clusters.
        :param k_max: Maximum number of clusters (default: None => square root of the number of x rows).
        :param extractor: Returns feature extraction algorithm (default: PCA).
        :param score: Function that takes dataset x and labels y as inputs and returns clustering score as output,
               or a case-insensitive abbreviation for one of the following score functions: <br/>
            - 'CH': Calinski-Harabasz score, <br/>
            - 'DB': Davies-Bouldin score, <br/>
            - 'S': Silhouette score. <br/>
        :param best: Returns the best score from the sequence (e.g., max for CH and S, min for DB).
        """
        self.k_min = k_min
        self.k_max = k_max
        self.extractor = extractor()
        self.score = score
        self.best = best

        self.k = 0
        self.k2cardinality: MutableMapping[int, tuple] = {}
        self.k2score: MutableMapping[int, float] = {}
        self.y = pd.Series(dtype=float)

    @abstractmethod
    def _fit(self, x: pd.DataFrame, k: int) -> pd.Series:
        """
        Runs clustering for the specified number of clusters, which is k_max in the first call and [k_min, k_max) in all
        the other calls.

        :param x: Input values indexed by object (row) and feature (column).
        :param k: Less than the maximum number of clusters.
        :return: Clusters indexed by object.
        """
        pass

    @ignore_warnings
    def fit(self, x: pd.DataFrame):
        """
        Runs iterative search for an optimal number of clusters.

        :param x: Input values indexed by object (row) and feature (column).
        """
        if len(x) <= self.k_min:
            self.k = len(x)
            self.y = pd.Series(x.index, index=x.index)
            self.k2score[self.k] = float('nan')
            self.k2cardinality[self.k] = tuple([1] * self.k)
            return self

        x_extracted = self.extractor.fit_transform(x)
        k_range = range(self.k_min, floor(sqrt(len(x))) if self.k_max is None else self.k_max)
        f_score = self.__score_functions.get(self.score.lower()) if isinstance(self.score, str) else self.score

        def cardinality(y: pd.Series) -> Tuple:
            return tuple(Counter(y).values())

        def fit_one(k: int):
            y = self._fit(x_extracted, k)
            score = f_score(x_extracted, y)
            return k, y, score

        def fit_all():
            k2y = {k_range.stop: self._fit(x_extracted, k_range.stop)}
            k2score = {k_range.stop: f_score(x_extracted, k2y[k_range.stop])}
            k2cardinality = {k_range.stop: cardinality(k2y[k_range.stop])}
            k_y_score = parallel(fit_one, k_range)
            for k, y, score in k_y_score:
                k2y[k] = y
                k2score[k] = score
                k2cardinality[k] = cardinality(y)

            best_score = self.best(list(k2score.values()))
            best_k = [k for k, s in k2score.items() if s == best_score][0]
            return best_k, k2cardinality, k2score, k2y[best_k]

        self.k, self.k2cardinality, self.k2score, self.y = fit_all()
        return self

    def fit_predict(self, x: pd.DataFrame) -> pd.Series:
        """
        Runs iterative search for an optimal number of clusters.

        :param x: Input values indexed by object (row) and feature (column).
        :return: Clusters indexed by object.
        """
        return self.fit(x).y

    def summary(self) -> pd.DataFrame:
        """
        Returns clustering scores indexed by the number of the obtained clusters 'k', with the 'score', 'selected' and
        'cardinality' columns showing the score for the cluster configuration with k clusters, whether k is selected,
        and the number of members in each cluster, respectively.
        """

        def _summary():
            for k in sorted(self.k2score.keys()):
                yield {'k': k,
                       'score': self.k2score[k],
                       'selected': k == self.k,
                       'cardinality': self.k2cardinality[k]}

        return pd.DataFrame(_summary()).set_index('k')


class TSGA(Clustering):
    def __init__(self,
                 k_min: 'int > 1' = 2,
                 k_max: Optional['int > 1'] = None,
                 extractor: Callable[[], FeatureExtractor] = lambda: PCA(),
                 km_method: Literal[None,
                                    'kmeans',
                                    'kmeans++',
                                    'bisect',
                                    'bisect++',
                                    'hartigan',
                                    'kshape'] = 'hartigan',
                 km_init=1,
                 km_iter=1,
                 hc_method: Literal['average',
                                    'centroid',
                                    'complete',
                                    'median',
                                    'single',
                                    'ward',
                                    'weighted'] = 'ward',
                 score: Union[str, Callable[[pd.DataFrame, pd.Series], float]] = 'CH',
                 best: Callable[[Sequence[float]], float] = lambda _: Math.knee(sorted(_, reverse=True)).y):
        """
        Parallel search for an optimal number of clusters based on the Time Series Grouping Algorithm (TSGA) [1]_.
        TSGA combines feature extraction, both k-means and Agglomerative Hierarchical Clustering (AHC) [2]_, and
        cluster validation (score), to find an optimal number of time series clusters for a short execution time
        based on simple parametric settings (minimum and maximum number of clusters).

        - First, a k-means algorithm is used to obtain the maximum number of clusters and respective cluster centers.
          The current implementation supports the following k-means algorithms: classic k-means [3]_, k-means++ [4]_,
          bisect k-means [5]_, bisect k-means combined with k-means++, Hartigan's k-means [6]_, and k-shape [7]_.
        - In the next step, the k-means centers are used instead of the original data to build an AHC tree, according to
          the KnA method [8]_. The KnA method helps to avoid the problem of having a too large distance matrix when
          building the tree directly from the original data. AHC can be used with any linkage method supported by the
          :func:`fastcluster.linkage` function.
        - In the next step, the constructed tree is cut at different places to obtain the rest of the clustering results
          for the range between the minimum and the maximum number of clusters.
        - The final clustering result is then chosen according to the specified score function. The current
          implementation supports Calinski-Harabasz score [9]_, Davies-Bouldin score [10]_, Silhouette score [11]_,
          as well as custom score functions.

        By default, TSGA uses the following default configuration:

        - Principal Component Analysis (PCA) is used for feature extraction.
        - Hartigan's k-means is used for initial clustering.
        - AHC is used with Ward's linkage to make clusters of more even size.
        - The obtained clusters are evaluated with the Calinski-Harabasz score [9]_.
        - The best score is selected by finding the knee of a curve defined by the cluster score
          to avoid cluster configurations with a too small number of clusters.

        References
        -----
        .. [1] Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić:
               Time series grouping algorithm for load pattern recognition,
               Computers in Industry 111: 140-147 (2019),
               https://doi.org/10.1016/j.compind.2019.07.009
        .. [2] Saeed Aghabozorgi, Ali Seyed Shirkhorshidi, Teh Ying Wah:
               Time-series clustering – A decade review,
               Information Systems 53: 16-38 (2015),
               https://doi.org/10.1016/j.is.2015.04.007
        .. [3] Stuart P. Lloyd:
               Least Squares Quantization in PCM,
               IEEE Transactions on Information Theory 28 (2): 129-137 (1982),
               https://doi.org/10.1109/TIT.1982.1056489
        .. [4] David Arthur, Sergei Vassilvitskii:
               k-means++: The advantages of careful seeding,
               Proceedings of the Eighteenth Annual ACM-SIAM symposium on Discrete algorithms: 1027-1035 (2007),
               https://dl.acm.org/doi/10.5555/1283383.1283494
        .. [5] Michael Steinbach, George Karypis and Vipin Kumar:
               A Comparison of Document Clustering Techniques,
               KDD Workshop on Text Mining 400: 535-536 (2000),
               https://tinyurl.com/333du6vv
        .. [6] John A. Hartigan:
               Clustering algorithms,
               Wiley series in probability and mathematical statistics: Applied probability and statistics (1975),
               https://tinyurl.com/yskvjcw4
        .. [7] John Paparrizos, Luis Gravano:
               Fast and Accurate Time-Series Clustering,
               ACM Transactions on Database SystemsVolume 42 (2): 1–49 (2017),
               https://doi.org/10.1145/3044711
        .. [8] Athman Bouguettaya, Qi Yu, Xumin Liu, Xiangmin Zhou, Andy Song:
               Efficient agglomerative hierarchical clustering,
               Expert Systems with Applications 42 (5): 2785-2797 (2015),
               https://doi.org/10.1016/j.eswa.2014.09.054
        .. [9] Calinski, T., Harabasz, J.,
               A dendrite method for cluster analysis,
               Communications in Statistics 3 (1): 1–27 (1974),
               https://dx.doi.org/10.1080/03610927408827101
        .. [10] David L. Davies, Donald W. Bouldin:
                A cluster separation measure,
                IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-1 (2): 224–227 (1979),
                https://dx.doi.org/10.1109/TPAMI.1979.4766909
        .. [11] Peter J. Rousseeuw:
                Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,
                Journal of Computational and Applied Mathematics 20: 53–65 (1987)
                https://dx.doi.org/10.1016/0377-0427(87)90125-7

        :param k_min: Minimum number of clusters.
        :param k_max: Maximum number of clusters (default: None => square root of the number of x rows).
        :param extractor: Returns feature extraction algorithm (default: PCA).
        :param km_method: k-means method: <br/>
            - None: skip k-means (not recommended for large datasets), <br/>
            - 'kmeans' : classic k-means, <br/>
            - 'kmeans++' : k-means++, <br/>
            - 'bisect' : bisect k-means, <br/>
            - 'bisect++' : bisect k-means++, <br/>
            - 'hartigan' : Hartigan's k-means, <br/>
            - 'kshape' : k-shape. <br/>
        :param km_init: Number of times the k-means algorithm will be run with different centroid seeds.
        :param km_iter: Maximum number of iterations of the k-means algorithm for a single run.
        :param hc_method: AHC linkage method supported by the fastcluster.linkage function: <br/>
            - 'average' : The distance between two clusters is
                          the average distance between each pair of elements from the two clusters. <br/>
            - 'centroid' : The distance between two clusters is
                           the distance between the cluster centers. <br/>
            - 'complete' : The distance between two clusters is
                           the maximum distance between any two elements from each cluster. <br/>
            - 'median' : The distance between two clusters is
                         the distance between the cluster midpoints. <br/>
            - 'single' : The distance between two clusters is
                         the minimum distance between any two elements from each cluster. <br/>
            - 'ward': The distance between two clusters is
                      the weighted squared distance between the cluster centers. <br/>
            - 'weighted' : The distance depends on the order of the merging steps.
        :param score: Function that takes dataset x and labels y as inputs and returns clustering score as output,
               or a case-insensitive abbreviation for one of the following score functions: <br/>
            - 'CH': Calinski-Harabasz score, <br/>
            - 'DB': Davies-Bouldin score, <br/>
            - 'S': Silhouette score. <br/>
        :param best: Returns the best score from the sequence (e.g., max for CH and S, min for DB).
        """
        super().__init__(k_min, k_max, extractor, score, best)
        self.km_method = km_method if km_method is None else km_method.lower()
        self.km_init = km_init
        self.km_iter = km_iter
        self.hc_method = hc_method.lower()

        self.__km_clusters = None
        self.__hc_tree = None

    def kmeans(self, k: int):
        """Returns new instance of k-means for the specified number of clusters."""

        if self.km_method == 'kmeans':
            return KMeans(n_clusters=k, n_init=self.km_init, max_iter=self.km_iter, init='random')
        if self.km_method == 'kmeans++':
            return KMeans(n_clusters=k, n_init=self.km_init, max_iter=self.km_iter, init='k-means++')
        if self.km_method == 'bisect':
            return BisectingKMeans(n_clusters=k, n_init=self.km_init, max_iter=self.km_iter, init='random')
        if self.km_method == 'bisect++':
            return BisectingKMeans(n_clusters=k, n_init=self.km_init, max_iter=self.km_iter, init='k-means++')
        if self.km_method == 'hartigan':
            return HKMeans(n_clusters=k, n_init=self.km_init, max_iter=self.km_iter, n_jobs=-1)
        if self.km_method == 'kshape':
            return KShape(n_clusters=k, n_init=self.km_init, max_iter=self.km_iter)
        else:
            return None

    def __cut_tree(self, k: int) -> np.array:
        return np.array(cut_tree(self.__hc_tree, n_clusters=k)).squeeze()

    def _fit(self, x: pd.DataFrame, k: int) -> pd.Series:
        if self.km_method is None:
            if self.__hc_tree is None:
                self.__hc_tree = linkage(x, method=self.hc_method)

            return pd.Series(self.__cut_tree(k), index=x.index)
        else:
            if self.__hc_tree is None:
                self.__km_clusters = pd.Series(self.kmeans(k).fit_predict(x.to_numpy()), index=x.index)
                self.__hc_tree = linkage(x.groupby(self.__km_clusters).mean(), method=self.hc_method)
                return self.__km_clusters

            return pd.Series(self.__cut_tree(k)[self.__km_clusters], index=x.index)

    def __str__(self):
        return f'{type(self).__name__}(km={self.km_method}, hc={self.hc_method})'

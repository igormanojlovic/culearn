from collections import OrderedDict

from culearn.clustering import *
from culearn.csv import *
from culearn.regression import *
from culearn.util import *
from plotly.graph_objs.scatter import Line
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer


class CumulantTransform(StrMixin):
    def __init__(self,
                 approx: Callable[[TimeSeriesID], Approximator] = lambda _: SeriesApproximator(),
                 clust: Callable[[], Clustering] = lambda: TSGA(),
                 imputer: Callable[[np.array], np.array] = lambda _: KNNImputer(copy=False).fit_transform(_),
                 encoder: TimeEncoder = TimeEncoders(MonthOfYear(), DayOfWeek(), TimeOfDay())):
        """
        Obtains time series clusters and transforms the input time series into time series of cluster-level cumulants:
        mean, standard deviation (instead of variance, to simplify scaling), standardized skewness, and standardized
        excess kurtosis.

        This method is a variation of time series approximation and clustering methods proposed in [1]_ and [2]_,
        extended to support probabilistic forecasting as follows:

        1. Each time series is approximated using the specified approximation method [1]_.
        2. Each approximated time series is normalized using the "division by mean" method [2]_.
        3. Each normalized approximated time series is aggregated by calculating the means of the temporal groups
           obtained using the specified time encoders.
        4. The aggregated time series with the same source are placed into one chain of values, representing one
           clustering object.
        5. The clustering objects are separated by time series categories and then clustered using an algorithm that
           finds an optimal cluster configuration (an optimal number of clusters in the specified range).
        6. The normalized values from step 2 are reduced to time series of cluster-level cumulants.
        7. These cumulants or the prediction of future cumulants obtained by a time series regressor can be
           inverse-transformed into quantiles based on Cornish–Fisher expansion.
        8. These quantiles can be denormalized (using the scaling factors from step 2) and combined into prediction
           intervals for individual time series.

        Steps 1-4 are performed in parallel per time series.
        Step 5 is performed in parallel per cluster configuration.
        Step 6 is performed in parallel per cluster.

        References
        -----
        .. [1] Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić:
               Time series grouping algorithm for load pattern recognition,
               Computers in Industry 111: 140-147 (2019),
               https://doi.org/10.1016/j.compind.2019.07.009
        .. [2] Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić, Darko Čapko:
               Hierarchical multiresolution representation of streaming time series,
               Big Data Research 26: 100256 (2021),
               https://doi.org/10.1016/j.bdr.2021.100256

        :param approx: Returns approximation method for specific time series ID.
        :param clust: Returns clustering method for obtaining the time series clusters.
        :param imputer: Returns imputation method for populating missing values across clustering objects.
        :param encoder: The encoder that defines the temporal groups for reducing time series before clustering.
        """
        self.approx = approx
        self.clust = clust
        self.imputer = imputer
        self.encoder = encoder

        self.series2approximator: MutableMapping[TimeSeriesID, Approximator] = {}
        self.category2clustering: MutableMapping[str, Clustering] = {}
        self.factors: MutableMapping[TimeSeriesID, float] = {}
        self.clusters = defaultdict(set)

    def __clustering_score(self, score: Callable[[Clustering], pd.DataFrame]) -> pd.DataFrame:
        """Returns clustering-specific scores additionally indexed by category."""

        def _scores():
            for category, clustering in self.category2clustering.items():
                original_score = score(clustering)
                extended_score = original_score.reset_index()
                extended_score['category'] = category
                yield extended_score.set_index(['category'] + list(original_score.index.names))

        return pd.concat(_scores(), axis=0)

    @property
    def clustering_score(self) -> pd.DataFrame:
        """
        Clustering scores indexed by the time series 'category' and the number of the obtained clusters 'k', with the
        'score', 'selected' and 'cardinality' columns showing the score for the cluster configuration with k clusters,
        whether k is selected, and the number of members in each cluster, respectively.
        """
        return self.__clustering_score(lambda _: _.summary())

    @property
    def extractor_score(self) -> pd.DataFrame:
        """
        Feature extraction scores indexed by the time series 'category' and 'feature', with the 'score' and 'selected'
        columns showing the score and whether the feature is selected, respectively.
        """
        return self.__clustering_score(lambda _: _.extractor.summary())

    def moments(self, ts_id: TimeSeriesID, interval: TimeInterval) -> pd.DataFrame:
        """Returns approximated central moments obtained by the fit method."""
        approx = self.series2approximator.get(ts_id)
        return pd.DataFrame() if approx is None else approx.transform(interval)

    def fit(self, x: Sequence[TimeSeries], resolution: TimeResolution, interval: TimeInterval):
        """
        Creates time series clusters.

        :param x: Input time series (future cluster members).
        :param resolution: Resolution of the time series values to cluster.
        :param interval: Time interval of the time series values to cluster.
        """
        x_steps = [_.start for _ in resolution.steps(interval)]
        x_time = pd.DataFrame([self.encoder(_) for _ in x_steps], index=x_steps)

        def aggregate(ts: TimeSeries):
            x_approx = self.approx(ts.ts_id)
            x_object = x_approx.fit_transform(ts, resolution, interval)
            if len(x_object) == 0:
                return ts.ts_id, x_approx, pd.Series(dtype=float), float('nan')

            x_object = x_object['mean']
            x_factor = float(x_object.mean())
            x_scaled = x_time.merge(x_object.to_frame(), left_index=True, right_index=True, how='left') \
                           .groupby(list(x_time.columns)).mean() / x_factor

            return ts.ts_id, x_approx, x_scaled, x_factor

        def create_objects() -> Mapping[TimeSeriesID, pd.Series]:
            id2obj = {}
            for x_id, x_approx, x_object, x_factor in parallel(aggregate, x):
                if len(x_object) == 0:
                    continue

                id2obj[x_id] = x_object
                self.series2approximator[x_id] = x_approx
                self.factors[x_id] = x_factor
            return id2obj

        def cluster_map(members: Iterable[TimeSeriesID], clusters: pd.Series):
            cluster2members = {}
            for member_id in members:
                cluster_index = clusters[clusters.index == member_id.source].values[0]
                cluster_id = TimeSeriesID(str(cluster_index), member_id.value_type, member_id.category)
                cluster2members.setdefault(cluster_id, set()).add(member_id)
            return cluster2members

        def create_clusters(id2obj: Mapping[TimeSeriesID, pd.Series]):
            for category in sorted({_.category for _ in id2obj.keys()}):
                x_category = {_id: ts for _id, ts in id2obj.items() if _id.category == category}
                x_merged = pd.concat(x_category.values(), axis=1)
                x_merged.columns = pd.Series(x_category.keys())

                x_id = pd.DataFrame({_: (_.source, _.value_type) for _ in x_category.keys()},
                                    index=['source', 'value_type']).T
                x_objects = pd.concat((x_id, x_merged.T), axis=1).pivot(index='source', columns='value_type')
                x_imputed = pd.DataFrame(self.imputer(x_objects.values), index=x_objects.index)

                self.category2clustering[category] = clustering = self.clust()
                x_clusters = clustering.fit_predict(x_imputed)
                self.clusters.update(cluster_map(x_id.index.values, x_clusters))

        create_clusters(create_objects())
        return self

    def transform(self, interval: TimeInterval) -> Sequence[TimeSeriesDataFrame]:
        """
        Transforms clustered time series into time series of cluster-level cumulants.

        :param interval: Time interval of the time series values to transform.
        :return: Cluster-level cumulants indexed by timestamp (row) and name (column).
        """

        def scale_moments(cluster_id: TimeSeriesID) -> Iterable[pd.DataFrame]:
            for member_id in self.clusters[cluster_id]:
                x_factor = self.factors.get(member_id)
                if x_factor is None:
                    continue

                x_moments = self.moments(member_id, interval)
                if len(x_moments) == 0:
                    continue

                x_moments['mean'] = x_moments['mean'] / x_factor
                x_moments['variance'] = x_moments['variance'] / x_factor ** 2
                x_moments['skewness'] = x_moments['skewness'] / x_factor ** 3
                x_moments['kurtosis'] = x_moments['kurtosis'] / x_factor ** 4
                yield x_moments

        def merge_moments(a: pd.DataFrame, b: pd.DataFrame):
            ab = a.merge(b, left_index=True, right_index=True, how='outer', suffixes=('_a', '_b'))
            ab.fillna(0, inplace=True)
            ab['count'] = ab['count_a'] + ab['count_b']
            ab['mean'] = (ab['count_a'] * ab['mean_a'] + ab['count_b'] * ab['mean_b']) / ab['count']

            def merge(moment: str, degree: int):
                sum_dev_a = ab['count_a'] * (ab[f'{moment}_a'] + (ab['mean_a'] - ab['mean']) ** degree)
                sum_dev_b = ab['count_b'] * (ab[f'{moment}_b'] + (ab['mean_b'] - ab['mean']) ** degree)
                return (sum_dev_a + sum_dev_b) / ab['count']

            ab['variance'] = merge('variance', 2)
            ab['skewness'] = merge('skewness', 3)
            ab['kurtosis'] = merge('kurtosis', 4)
            return ab[['count', 'mean', 'variance', 'skewness', 'kurtosis']]

        def reduce_moments(moments: Iterable[pd.DataFrame]) -> pd.DataFrame:
            """Workaround for calling the 'reduce' function with empty sequences."""
            reduced = pd.DataFrame()
            for m in moments:
                if len(m) == 0:
                    continue
                reduced = m if len(reduced) == 0 else merge_moments(reduced, m)
            return reduced

        def merge_cumulants(cluster_id: TimeSeriesID):
            cumulants = reduce_moments(scale_moments(cluster_id))
            if len(cumulants) == 0:
                return cluster_id, cumulants

            cumulants['skewness'] = cumulants['skewness'] / cumulants['variance'] ** 1.5
            cumulants['kurtosis'] = cumulants['kurtosis'] / cumulants['variance'] ** 2 - 3
            cumulants['variance'] = np.sqrt(cumulants['variance'])  # variance => stdev
            cumulants.rename(columns={'variance': 'stdev'}, inplace=True)
            cumulants.drop(columns=['count'], inplace=True)
            cumulants.interpolate('spline', order=2, inplace=True)
            cumulants.fillna(0, inplace=True)
            return cluster_id, cumulants

        id2df = dict(parallel(merge_cumulants, self.clusters.keys()))
        return [TimeSeriesDataFrame(_id, _df) for _id, _df in id2df.items() if len(_df) > 0]

    def fit_transform(self, x: Sequence[TimeSeries], resolution: TimeResolution, interval: TimeInterval):
        """
        Creates time series clusters and transforms the input time series into time series of cluster-level cumulants.

        :param x: Input time series (future cluster members).
        :param resolution: Resolution of the time series values to cluster and transform.
        :param interval: Time interval of the time series values to cluster and transform.
        :return: Cluster-level cumulants indexed by timestamp (row) and name (column).
        """
        return self.fit(x, resolution, interval).transform(interval)

    def inverse_transform(self,
                          cumulants: Sequence[TimeSeriesDataFrame],
                          p: Iterable[float],
                          clusters=True,
                          members=False) \
            -> Iterable[TimeSeriesPrediction]:
        """
        Transforms time series of cluster-level cumulants into cluster-level and/or member-level prediction intervals.
        If both cluster- and member-level intervals are requested then the cluster-level intervals are yielded before
        the member-level intervals for the same cluster (e.g., intervals for cluster 1, intervals for each member in
        cluster 1, intervals for cluster 2, intervals for each member in cluster 2, etc.).

        :param cumulants: Cluster-level cumulants indexed by timestamp (row) and name (column).
        :param p: Prediction interval probability value.
        :param clusters: Whether to return normalized cluster-level prediction intervals.
        :param members: Whether to return denormalized member-level prediction intervals.
        :return: Cluster/member prediction intervals indexed by timestamp.
        """

        def qcf(p_quantile: float, n: int, cdf: pd.DataFrame):
            return Math.qcf(p_quantile, n, cdf['mean'], cdf['stdev'], cdf['skewness'], cdf['kurtosis'])

        def pi(p_interval: float, n: int, cdf: pd.DataFrame):
            p_lower = (1 - p_interval) / 2
            p_upper = 1 - p_lower
            return PredictionInterval(p_interval, qcf(p_lower, n, cdf), qcf(p_upper, n, cdf))

        def pred(n: int, cdf: pd.DataFrame):
            return [pi(_, n, cdf) for _ in p]

        for c in cumulants:
            if len(c) == 0:
                continue

            member_ids = list(self.clusters[c.ts_id])
            c_pred = TimeSeriesPrediction(c.ts_id)
            c_pred.extend(pred(len(member_ids), c))

            if clusters:
                yield c_pred

            if members:
                for member_id in member_ids:
                    factor = self.factors.get(member_id)
                    if factor is None:
                        continue

                    m_pred = TimeSeriesPrediction(member_id)
                    for i in c_pred:
                        m_pred.append(PredictionInterval(i.p, i.lower * factor, i.upper * factor))

                    yield m_pred

    def figure(self,
               cumulants: Sequence[TimeSeriesDataFrame],
               p: Iterable[float],
               pred_fill='rgba(0, 115, 207, 0.1)',
               pred_line=Line(color='black'),
               true_line=Line(color='rgba(128, 128, 128, 0.1)'),
               show_lower=True,
               show_upper=True,
               show_actual=False,
               **kwargs):
        """
        Transforms time series of cluster-level cumulants into time series of cluster-level prediction intervals to
        create a plotly figure with one subplot for each cluster. Optionally, each subplot can be extended with the
        normalized member-level time series values (true values that match the scale of cluster-level values).

        :param cumulants: Cluster-level cumulants indexed by timestamp (row) and name (column).
        :param p: Prediction interval probability value.
        :param pred_fill: Plotly fill color for the prediction interval area.
        :param pred_line: Plotly line stype for cluster-level prediction intervals.
        :param true_line: Plotly line style for member-level time series values.
        :param show_lower: Whether to show lower boundaries of prediction intervals.
        :param show_upper: Whether to show upper boundaries of prediction intervals.
        :param show_actual: Whether to show the actual member-level time series values.
        :param kwargs: Arguments for the plotly update_layout method.
        """
        titles = [f'Cluster with {len(self.clusters[_.ts_id])} members: {_.ts_id}' for _ in cumulants]
        fig = make_subplots(rows=len(cumulants), cols=1, subplot_titles=titles)

        row = 1
        legendgroup = 'legend'
        member_legend_shown = False
        cluster_legend_shown = defaultdict(bool)
        for i in range(len(cumulants)):
            if show_actual:
                for mid in self.clusters[cumulants[i].ts_id]:
                    mdf = self.moments(mid, TimeInterval.from_index(cumulants[i].index))
                    factor = self.factors.get(mid)
                    if len(mdf) == 0 or factor is None:
                        continue

                    mdf_norm = mdf['mean'] / factor
                    fig.add_scatter(x=mdf.index,
                                    y=mdf_norm,
                                    name='actual',
                                    legendgroup=legendgroup,
                                    showlegend=not member_legend_shown,
                                    line=true_line,
                                    row=row,
                                    col=1)
                    member_legend_shown = True

            for prediction_intervals in self.inverse_transform([cumulants[i]], p, clusters=True, members=False):
                for pi in sorted(prediction_intervals, key=lambda _: _.p, reverse=True):
                    pi_name = f'{100 * pi.p}% PI'
                    if show_lower:
                        fig.add_scatter(x=pi.lower.index,
                                        y=pi.lower,
                                        name=pi_name,
                                        legendgroup=legendgroup,
                                        showlegend=not cluster_legend_shown.get(pi.p),
                                        line=pred_line,
                                        fillcolor=pred_fill,
                                        fill=None if show_upper else 'tozeroy',
                                        row=row,
                                        col=1)
                        cluster_legend_shown[pi.p] = True
                    if show_upper:
                        fig.add_scatter(x=pi.upper.index,
                                        y=pi.upper,
                                        name=pi_name,
                                        legendgroup=legendgroup,
                                        showlegend=not cluster_legend_shown.get(pi.p),
                                        line=pred_line,
                                        fillcolor=pred_fill,
                                        fill='tonexty' if show_lower else 'tozeroy',
                                        row=row,
                                        col=1)
                        cluster_legend_shown[pi.p] = True

            row += 1

        fig.update_layout(
            plot_bgcolor="white",
            font_family="Times New Roman",
            font_color="black",
            font_size=12,
            height=800,
        )

        fig.update_layout(**kwargs)

        return fig

    def evaluate(self, cumulants: Sequence[TimeSeriesDataFrame], p: Iterable[float]):
        """
        Transforms time series of cluster-level cumulants into time series of cluster-level prediction intervals to
        calculate pinball and winkler score compared to respective member-level time series values.

        :param cumulants: Cluster-level cumulants indexed by timestamp (row) and name (column).
        :param p: Prediction interval probability value.
        :return: A pair of pinball and winkler scores indexed by member ID (row) and the probability value (column).
        """

        def df_row(ts_id: TimeSeriesID, p2score: Mapping[float, float]):
            return pd.DataFrame(p2score.values(), columns=[ts_id], index=p2score.keys()).T.rename_axis('member')

        def ps_ws_row(member_intervals):
            member_id, norm_intervals = member_intervals

            factor = self.factors.get(member_id)
            if factor is None:
                return pd.DataFrame(), pd.DataFrame()

            time_interval = TimeInterval.from_index(norm_intervals[0].lower.index)
            x = self.moments(member_id, time_interval)
            if len(x) == 0:
                return pd.DataFrame(), pd.DataFrame()

            x = x['mean']
            ps = OrderedDict()
            ws = {}
            for norm_pi in norm_intervals:
                pi = PredictionInterval(norm_pi.p, norm_pi.lower * factor, norm_pi.upper * factor)
                p_lower = (1 - pi.p) / 2
                p_upper = 1 - p_lower
                merged = pd.concat((pi.lower, pi.upper), axis=1).merge(x, left_index=True, right_index=True).dropna()
                ps[p_lower] = Math.pinball(p_lower, merged.iloc[:, -1], merged.iloc[:, 0]).mean()
                ps[p_upper] = Math.pinball(p_upper, merged.iloc[:, -1], merged.iloc[:, 1]).mean()
                ws[pi.p] = Math.winkler(pi.p, merged.iloc[:, -1], merged.iloc[:, 0], merged.iloc[:, 1]).mean()

            return df_row(member_id, ps), df_row(member_id, ws)

        cluster_intervals = self.inverse_transform(cumulants, p, clusters=True, members=False)
        member2intervals = {member_id: c for c in cluster_intervals for member_id in self.clusters[c.ts_id]}
        ps_ws = parallel(ps_ws_row, member2intervals.items())
        all_ps = pd.concat([ps for ps, _ in ps_ws], axis=0)
        all_ws = pd.concat([ws for _, ws in ps_ws], axis=0)
        return all_ps.sort_index(axis=1), all_ws


class CumulantEvaluation(StrMixin):
    def __init__(self,
                 clustering_score: pd.DataFrame,
                 extractor_score: pd.DataFrame,
                 x_selector_score: pd.DataFrame,
                 y_selector_score: pd.DataFrame,
                 regressor_score: pd.DataFrame,
                 pinball_score: pd.DataFrame,
                 winkler_score: pd.DataFrame):
        """Wraps around :class:`CumulantLearner` evaluation results."""
        self.clustering_score = clustering_score
        self.extractor_score = extractor_score
        self.x_selector_score = x_selector_score
        self.y_selector_score = y_selector_score
        self.regressor_score = regressor_score
        self.pinball_score = pinball_score
        self.winkler_score = winkler_score

    def to_csv(self, *directory, sep=','):
        """Saves the results to separate CSV files."""
        DataFrameCSV(*directory, 'clustering_score.csv', sep=sep, index=True).write(self.clustering_score)
        DataFrameCSV(*directory, 'extractor_score.csv', sep=sep, index=True).write(self.extractor_score)
        DataFrameCSV(*directory, 'x_selector_score.csv', sep=sep, index=True).write(self.x_selector_score)
        DataFrameCSV(*directory, 'y_selector_score.csv', sep=sep, index=True).write(self.y_selector_score)
        DataFrameCSV(*directory, 'regressor_score.csv', sep=sep, index=True).write(self.regressor_score)
        DataFrameCSV(*directory, 'pinball_score.csv', sep=sep, index=True).write(self.pinball_score)
        DataFrameCSV(*directory, 'winkler_score.csv', sep=sep, index=True).write(self.winkler_score)


class CumulantLearner(StrMixin):
    def __init__(self,
                 dataset: PredictionDataset,
                 resolution: TimeResolution,
                 transformer: CumulantTransform,
                 regressor: Callable[[], TimeSeriesRegressor]):
        """
        Cumulant Learning is a pattern recognition method designed to support probabilistic time series forecasting
        with high forecast accuracy and short execution time when dealing with data streams coming from numerous data
        sources. The cumulant learner implements the Cumulant Learning method utilizing:

        - :class:`CumulantTransform` to cluster similar time series and obtain time series of cluster-level cumulants.
        - :class:`TimeSeriesRegressor` (:class:`DeepTimeSeriesRegressor` or :class:`ShallowTimeSeriesRegressor`) to
          capture local trends and time dependencies in the obtained cumulants based on lagged cumulants, encoded time
          (calendar) features, and other exogenous features. Note that one regressor is trained for each cluster.

        The learned patterns are used to predict future cumulants. After that, the predicted cumulants can be
        transformed into prediction intervals for individual time series based on Cornish–Fisher expansion.
        """
        self.dataset = dataset
        self.resolution = resolution
        self.transformer = transformer
        self.regressor = regressor

        self.interval2cumulants = {}
        self.cluster2regressor: MutableMapping[str, TimeSeriesGroupRegressor] = {}

    def __x(self, interval: TimeInterval):
        """Prepares x values using resampling and linear interpolation."""
        x = self.dataset.x.resample(self.resolution).interpolate(method='linear')
        return x[interval.contains(x.index)].dropna()

    @staticmethod
    def __clusters(y: Sequence[TimeSeriesDataFrame]) -> Mapping[str, Sequence[TimeSeriesDataFrame]]:
        clusters = {}
        for df in y:
            clusters.setdefault(df.ts_id.source, []).append(df)
        return clusters

    def __train(self, x: pd.DataFrame, y: Sequence[TimeSeriesDataFrame]):
        """Trains the underlying cluster-level regressors."""
        clusters = self.__clusters(y)
        for c, ydfs in clusters.items():
            regressor = self.cluster2regressor.get(c)
            if regressor is None:
                self.cluster2regressor[c] = regressor = TimeSeriesGroupRegressor(self.regressor())
            regressor.fit(x, ydfs)
        return self

    def __predict(self, x: pd.DataFrame, y: Sequence[TimeSeriesDataFrame]):
        """Predicts y future based on y past (in the lookback range) and x future (in the forecast horizon)."""
        y_pred = []
        clusters = self.__clusters(y)
        for c, ydfs in clusters.items():
            dfs_pred = self.cluster2regressor.get(c).predict(x, ydfs)
            ydfs_pred = [TimeSeriesDataFrame(ydfs[i].ts_id, dfs_pred[i]) for i in range(len(ydfs))]
            y_pred.extend(ydfs_pred)
        return y_pred

    def __regression_score(self, score: Callable[[TimeSeriesGroupRegressor], pd.DataFrame]) -> pd.DataFrame:
        """Returns regression-specific scores additionally indexed by cluster."""

        def _scores():
            for cluster, regressor in self.cluster2regressor.items():
                original_score = score(regressor)
                extended_score = original_score.reset_index()
                extended_score['cluster'] = cluster
                yield extended_score.set_index(['cluster'] + list(original_score.index.names))

        return pd.concat(_scores(), axis=0)

    @property
    def x_selector_score(self):
        """
        Feature selection scores indexed by 'cluster', 'x' and 'y' features, with the 'score' and 'selected' columns
        showing the score and whether the x feature is selected, respectively.
        """
        return self.__regression_score(lambda _: _.base.x_selector.summary())

    @property
    def y_selector_score(self):
        """
        Lag selection scores indexed by 'cluster', y 'feature' and 'lag', with the 'score', 'selected', and 'ci' columns
        showing the score, whether the feature is selected, and the confidence interval radius, respectively.
        """
        return self.__regression_score(lambda _: _.base.y_selector.summary())

    @property
    def regressor_score(self):
        """Regressor training scores indexed by 'cluster' and the calls to the 'fit' method."""
        return self.__regression_score(lambda _: _.base.summary())

    def horizon(self, t: datetime) -> TimeInterval:
        """Returns the maximum horizon range from the underlying regressors, starting at the specified timestamp."""
        horizon = max([_.base.horizon for _ in self.cluster2regressor.values()])
        return TimeInterval(t, t + timedelta(seconds=self.resolution.seconds * horizon))

    def lookback(self, t: datetime) -> TimeInterval:
        """Returns the maximum lookback range from the underlying regressors, ending at the specified timestamp."""
        lookback = max([_.base.lookback for _ in self.cluster2regressor.values()])
        return TimeInterval(t - timedelta(seconds=self.resolution.seconds * lookback), t)

    def fit(self, interval: TimeInterval):
        """Learns the time series of cluster-level cumulants from the specified interval."""
        self.transformer.fit(self.dataset.y, self.resolution, interval)
        self.interval2cumulants[interval] = y_cumulants = self.transformer.transform(interval)
        self.__train(self.__x(interval), y_cumulants)
        return self

    def update(self, interval: TimeInterval):
        """Learns more of the cluster-level cumulants from the specified interval but keeps the clusters."""
        fit_interval = TimeInterval(self.lookback(interval.start).start, interval.end)
        self.interval2cumulants[interval] = y_cumulants = self.transformer.transform(fit_interval)
        self.__train(self.__x(fit_interval), y_cumulants)
        return self

    def cumulants(self, interval: TimeInterval):
        """Returns time series of cluster-level cumulants in the specified interval."""
        for i, c in self.interval2cumulants.items():
            if i.contains(interval):
                return [_.select(interval) for _ in c]
        return self.transformer.transform(interval)

    def predict_cumulants(self, start: datetime):
        """
        Returns time series of cluster-level cumulants in the forecast horizon starting at the specified timestamp.

        :param start: Start of the forecast horizon.
        :return: Cluster-level cumulants indexed by timestamp (row) and name (column).
        """
        y_cumulants = self.cumulants(self.lookback(start))
        return self.__predict(self.__x(self.horizon(start)), y_cumulants)

    def predict(self, start: datetime, p: Iterable[float], clusters=True, members=False):
        """
        Returns time series of cluster-level and/or member-level prediction intervals in the forecast horizon.
        If both cluster- and member-level intervals are requested then the cluster-level intervals are yielded before
        the member-level intervals for the same cluster (e.g., intervals for cluster 1, intervals for each member in
        cluster 1, intervals for cluster 2, intervals for each member in cluster 2, etc.).

        :param start: Start of the forecast horizon.
        :param p: Prediction interval probability value.
        :param clusters: Whether to return normalized cluster-level prediction intervals.
        :param members: Whether to return denormalized member-level prediction intervals.
        :return: Cluster/member prediction intervals indexed by timestamp.
        """
        y_cumulants = self.predict_cumulants(start)
        return self.transformer.inverse_transform(y_cumulants, p, clusters, members)

    def figure(self,
               start: datetime,
               p: Iterable[float],
               pred_fill='rgba(0, 115, 207, 0.1)',
               pred_line=Line(color='black', dash='dash'),
               true_line=Line(color='rgba(128, 128, 128, 0.1)'),
               show_lower=True,
               show_upper=True,
               show_actual=False,
               **kwargs):
        """
        Returns plotly figure with cluster-level prediction intervals in the forecast horizon. One subplot is made for
        each cluster. Optionally, each subplot can be extended with the normalized member-level time series values
        (true values that match the scale of cluster-level values).

        :param start: Start of the forecast horizon.
        :param p: Prediction interval probability value.
        :param pred_fill: Plotly fill color for the prediction interval area.
        :param pred_line: Plotly line stype for cluster-level prediction intervals.
        :param true_line: Plotly line style for member-level time series values.
        :param show_lower: Whether to show lower boundaries of prediction intervals.
        :param show_upper: Whether to show upper boundaries of prediction intervals.
        :param show_actual: Whether to show the actual member-level time series values.
        :param kwargs: Arguments for the plotly update_layout method.
        """
        y_cumulants = self.predict_cumulants(start)
        return self.transformer.figure(cumulants=y_cumulants,
                                       p=p,
                                       pred_fill=pred_fill,
                                       pred_line=pred_line,
                                       true_line=true_line,
                                       show_lower=show_lower,
                                       show_upper=show_upper,
                                       show_actual=show_actual,
                                       **kwargs)

    def evaluate(self, fit: TimeInterval, predict: TimeInterval, update: int, p: Iterable[float], verbose=False):
        """
        Iteratively runs training (fit and update) and testing (prediction) over the specified time intervals, and then
        evaluates the obtained results compared to the actual time series values using the pinball and winkler scores.
        The evaluation results also contain other evaluation scores obtained during the training process.
        Note that initial training (fit) is performed only if the learner was not previously trained.

        :param fit: Initial training time interval.
        :param predict: Prediction time interval (for one or more forecast horizons).
        :param update: Number of predictions to perform before an incremental update.
        :param p: Prediction interval probability value.
        :param verbose: Whether to print execution details.
        """

        def vprint(s: str):
            if verbose:
                print(s)

        def select(y: Sequence[TimeSeriesDataFrame], interval: TimeInterval) -> Sequence[TimeSeriesDataFrame]:
            return [_.select(interval) for _ in y]

        def initialize() -> Sequence[TimeSeriesDataFrame]:
            step = 'CUMULANT LEARNING EVALUATION - Step 1/3: initial training'
            t = Time.now()

            if not self.cluster2regressor:
                vprint(step)
                vprint(f'Learning: {fit}')
                self.transformer.fit(self.dataset.y, self.resolution, fit)
            else:
                vprint(f'{step} skipped - the learner is already trained.')

            interval = TimeInterval(min(fit.start, predict.start), max(fit.end, predict.end))
            y = self.transformer.transform(interval)

            if self.cluster2regressor:
                vprint(f'Learned {len(y)} time series of cluster-level cumulants.')
            else:
                vprint(f'Learning {len(y)} time series of cluster-level cumulants.')
                self.__train(self.__x(fit), select(y, fit))
                vprint(f'{step} finished in {round(Time.now() - t)} sec.')

            return y

        def predict_and_update(y: Sequence[TimeSeriesDataFrame]) -> Sequence[TimeSeriesDataFrame]:
            step = f'CUMULANT LEARNING EVALUATION - Step 2/3: incremental prediction and updates'
            vprint(step)

            t = Time.now()
            p_interval = self.horizon(predict.start)
            u_interval_start = predict.start
            u_countdown = update
            u_duration = 0
            u_count = 0
            y_pred_dict: MutableMapping[TimeSeriesID, MutableSequence[TimeSeriesDataFrame]] = {}
            while p_interval.end <= predict.end:
                vprint(f'Predicting: {p_interval}')
                y_past = select(y, self.lookback(p_interval.start))
                y_pred_list = self.__predict(self.__x(p_interval), y_past)
                for ydf in y_pred_list:
                    y_pred_dict.setdefault(ydf.ts_id, []).append(ydf)

                u_countdown -= 1
                if u_countdown <= 0:
                    u_time = Time.now()
                    u_interval = TimeInterval(u_interval_start, p_interval.end)
                    vprint(f'Learning: {u_interval}')
                    self.update(u_interval)
                    u_interval_start = u_interval.end
                    u_duration += Time.now() - u_time
                    u_count += 1
                    u_countdown = update

                p_interval = TimeInterval(p_interval.end, p_interval.end + p_interval.delta)

            cumulants = [TimeSeriesDataFrame(yid, pd.concat(ydfs, axis=0)) for yid, ydfs in y_pred_dict.items()]
            elapsed = Time.now() - t

            vprint(f'{step} finished in {round(elapsed)} sec (~{round(u_duration / u_count)} sec/update).')
            return cumulants

        def create_evaluation(cumulants: Sequence[TimeSeriesDataFrame]) -> CumulantEvaluation:
            step = f'CUMULANT LEARNING EVALUATION - Step 3/3: evaluation of prediction'
            vprint(step)
            t = Time.now()
            pinball_score, winkler_score = self.transformer.evaluate(cumulants, p)
            vprint(f'{step} finished in {round(Time.now() - t)} sec.')
            return CumulantEvaluation(self.transformer.clustering_score,
                                      self.transformer.extractor_score,
                                      self.x_selector_score,
                                      self.y_selector_score,
                                      self.regressor_score,
                                      pinball_score,
                                      winkler_score)

        return create_evaluation(predict_and_update(initialize()))

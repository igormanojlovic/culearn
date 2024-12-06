import tensorflow as tf

from math import floor, ceil
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import SVR, NuSVR
from statsmodels.formula.api import quantreg
from tensorflow.keras import layers as tfl
from xgboost import XGBRegressor

from culearn.features import *
from culearn.util import Time, ignore_warnings, parallel


class MultiRegressor:
    def __init__(self, base: Callable, n=1, incremental=False):
        """
        Wrapper around multiple univariate regressors, imitating a multivariate regressor.

        :param base: Creates one underlying regressor that contains fit and predict methods.
        :param n: Number of y variables (one y variable will be predicted by each regressor).
        :param incremental: Whether to imitate incremental learning (true) or to retrain the model on each fit (false).
               The incremental learning is imitated by preserving the x and y values after each fit, and by joining the
               preserved values with the new ones next time the fit is called. Note that the true incremental learning
               can only be achieved with the right underlying regression model. On the other hand, the retraining is
               simply performed by replacing the old models with the new ones, trained only on the new data.
        """
        self.base = base
        self.n = n
        self.incremental = incremental

        self.__regressors = [base() for _ in range(n)]
        self.__x: Sequence[np.array] = []
        self.__y: Sequence[np.array] = []

    @ignore_warnings
    def fit(self, x: Sequence[np.array], y: Sequence[np.array]):
        """Fits underlying regressors with equal-size sequences of inputs and outputs."""
        if self.incremental:
            self.__x = x = [np.vstack((self.__x[i], x[i])) for i in range(self.n)] if self.__x else x
            self.__y = y = [np.vstack((self.__y[i], y[i])) for i in range(self.n)] if self.__y else y
        parallel(lambda i: self.__regressors[i].fit(x[i], y[i]), range(self.n))
        return self

    @ignore_warnings
    def predict(self, x: Sequence[np.array]) -> Sequence[np.array]:
        """Returns a sequence of predicted outputs for a sequence of inputs."""
        return [self.__regressors[i].predict(x[i]) for i in range(self.n)]

    def fit_predict(self, x: Sequence[np.array], y: Sequence[np.array]) -> Sequence[np.array]:
        return self.fit(x, y).predict(x)


class MultiQuantileRegressor(StrMixin):
    def __init__(self, p: Iterable[float]):
        """
        Wrapper around multiple univariate quantile regressors, imitating a multivariate quantile regressor.

        :param p: Quantile probability.
        """
        self.p = list(p)
        self.__regressors = {}

    @staticmethod
    def __rename(x: pd.DataFrame, y=pd.DataFrame()):
        """
        Assigns universal names to x and y so that they can be always used in formulas.

        :param x: Multivariate input values.
        :param y: Multivariate output values.
        :return: Data frame with joined x and y values, and a map from the original y column to y formula.
        """
        xy = pd.concat((x, y), axis=1)
        x_indexed = [f'X{_}' for _ in range(len(x.columns))]
        y_indexed = [f'Y{_}' for _ in range(len(y.columns))]
        xy.columns = x_indexed + y_indexed
        x_formula = ' + '.join(x_indexed)
        y2formula = {y.columns[i]: f'{y_indexed[i]} ~ {x_formula}' for i in range(len(y.columns))}
        return xy, y2formula

    @ignore_warnings
    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        xy, y2formula = self.__rename(x, y)

        def params() -> Sequence[Tuple]:
            return [(_y, _p, _f) for _y, _f in y2formula.items() for _p in self.p]

        def fit_one(y_p_f: Tuple):
            _y, _p, _f = y_p_f
            return f'{_y} (p={_p})', quantreg(_f, xy).fit(q=_p)

        self.__regressors = dict(parallel(fit_one, params()))

    @ignore_warnings
    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        x, _ = self.__rename(x)
        y_pred = pd.concat([r.predict(x) for r in self.__regressors.values()], axis=1)
        y_pred.columns = self.__regressors.keys()
        return y_pred

    def fit_predict(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        return self.fit(x, y).predict(x)


class DNN(tfl.Layer, StrMixin):
    def __init__(self, f: Callable[[], Iterable[tfl.Layer]], *args, **kwargs):
        """Deep Neural Network (DNN)."""

        super().__init__(*args, **kwargs)
        self._layers = list(f())

    def call(self, inputs, training=None, mask=None):
        for _layer in self._layers:
            inputs = _layer(inputs)
        return inputs


class CNN(DNN):
    def __init__(self,
                 depth: 'int >= 0',
                 n_filters: 'int > 0',
                 shape: Union['int > 1', Tuple],
                 avg: bool = True,
                 drop: '0 < float < 1' = 0.5,
                 *args, **kwargs):
        """
        A series of Convolution Neural Networks (CNNs), with alternating convolution, pooling and dropout layers.

        :param depth: Number of alternating convolution, pooling and dropout layers (number of layer triples).
        :param n_filters: Number of convolution filters in each convolution layer.
        :param shape: Shape of both filters in convolution layers and pools and pooling layers: <br/>
             - (x) for 1D layers, <br/>
             - (x, y) for 2D layers, <br/>
             - (x, y, z) for 3D layers.
        :param avg: If true, then average pooling is used; otherwise, maximum pooling is used.
        :param drop: Dropout rate for each dropout layer.
        :param args: Base class arguments.
        :param kwargs: Base class key-value arguments.
        """

        def create_layers() -> Iterable[tfl.Layer]:
            n_dim = 1 if isinstance(shape, int) else len(list(shape))
            conv, pool = self.__conv_and_pool(n_dim, avg)
            for _ in range(depth):
                yield conv(filters=n_filters, kernel_size=shape, activation='relu', padding='same')
                yield pool(pool_size=shape)
                yield tfl.Dropout(rate=drop)

        super().__init__(create_layers, *args, **kwargs)

    @staticmethod
    def __conv_and_pool(n_dim: 'int > 0', avg: bool):
        if n_dim == 1:
            return tfl.Conv1D, (tfl.AveragePooling1D if avg else tfl.MaxPooling1D)
        elif n_dim == 2:
            return tfl.Conv2D, (tfl.AveragePooling2D if avg else tfl.MaxPooling2D)
        else:
            return tfl.Conv3D, (tfl.AveragePooling3D if avg else tfl.MaxPooling3D)


class SAE(DNN):
    def __init__(self,
                 depth: 'int >= 0',
                 edge: 'int > 0',
                 middle: 'int > 0',
                 drop: '0 < float < 1' = 0.5,
                 *args, **kwargs):
        """
        A series of alternating fully-connected feed-forward encoder and decoder layers with interjected dropout layers,
        namely Stacked Autoencoder (SAE).

        :param depth: Number of alternating feed-forward layers and dropout layers (number of layer pairs).
        :param edge: Number of neurons on each edge of SAE (number of neurons in input and output layers).
        :param middle: Number of neurons in the middle of SAE (number of neurons in the code layer).
        :param drop: Dropout rate for each dropout layer.
        :param args: Base class arguments.
        :param kwargs: Base class key-value arguments.
        """

        def create_layers() -> Iterable[tfl.Layer]:
            for w in self.__width(depth, edge, middle):
                yield tfl.Dense(units=w, activation='tanh')
                yield tfl.Dropout(rate=drop)

        super().__init__(create_layers, *args, **kwargs)

    @staticmethod
    def __width(depth: int, edge: int, middle: int) -> Sequence[int]:
        if depth < 3:
            return [edge] * depth

        half = (depth - 1) / 2
        step = floor((edge - middle) / floor(half))

        width = [edge]
        for i in range(floor(half)):
            width.append(np.clip(width[i] - step, middle, edge))

        width.extend(reversed(width[:ceil(half)]))
        return width


class S2S_DNN(tfl.Layer, StrMixin):
    def __init__(self, *args, **kwargs):
        """Abstract Sequence-to-Sequence (S2S) Deep Neural Network (DNN)."""
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _s2s(self, y_past, x_future):
        """
        Creates the S2S-DNN structure.

        :param y_past: Tensor with endogenous input values (past).
        :param x_future: Tensor with exogenous input values (future).
        :return: Output layer of the S2S-DNN structure.
        """
        pass

    def call(self, inputs, training=None, mask=None):
        """
        Calls S2S-DNN on new inputs and returns the output as tensor.

        :param inputs: List with two tensors: past y values and future x values.
        :param training: Training or inference mode.
        :param mask: Bool tensor or None.
        :return: Output tensor.
        """
        return self._s2s(inputs[0], inputs[1])


class S2S_RNN(S2S_DNN):
    def __init__(self,
                 units: 'int > 0' = 64,
                 cnn_depth: 'int >= 0' = 5,
                 cnn_filters: 'int > 0' = 256,
                 cnn_shape: 'int > 1' = 2,
                 sae_depth: 'int >= 0' = 11,
                 sae_middle: 'int > 0' = 16,
                 drop: '0 < float < 1' = 0.5,
                 *args, **kwargs):
        """
        Abstract Sequence-to-Sequence (S2S) Recurrent Neural Network (RNN) with attention and bidirectional encoder.
        The total length of the S2S context vector in both directions will be 2*S*U, where S is the number of states
        obtained by one recurrent unit (which depends on the type of unit) and U is the number of units that will be
        used to encode the input sequence in one direction.

        Optionally, :class:`CNN` can be prepended to S2S-RNN to prevent long and noisy input sequences in destabilizing
        the network and :class:`SAE` can be applied to the S2S context to denoise the context vector and increase the
        representation capacity.

        :param units: Number of recurrent units that will be used to encode the input sequence in one direction.
        :param cnn_depth: Number of CNN layers (CNN will not be used if the number is 0).
        :param cnn_filters: Number of CNN filters.
        :param cnn_shape: Shape of 1D CNN filters and pools.
        :param sae_depth: Number of SAE layers (SAE will not be used if the number is 0).
        :param sae_middle: Number of neurons in the middle of SAE.
        :param drop: Dropout rate for CNN and SAE layers.
        :param args: Base class arguments.
        :param kwargs: Base class key-value arguments.
        """

        super().__init__(*args, **kwargs)

        bi_units = 2 * units
        rnn = self._cell_type()

        self.cnn = CNN(depth=cnn_depth, n_filters=cnn_filters, shape=cnn_shape, drop=drop)
        self.rnn_encoder = tfl.Bidirectional(rnn(units, return_state=True, return_sequences=False))
        self.rnn_decoder = rnn(bi_units, return_state=False, return_sequences=True)
        self.rnn_state_sae = [SAE(sae_depth, bi_units, sae_middle, drop) for _ in self._cell_state_index()]
        self.rnn_state_concat = [tfl.Concatenate() for _ in self._cell_state_index()]
        self.rnn_attention = tfl.Attention()
        self.s2s_concat = tfl.Concatenate()

    @abstractmethod
    def _cell_type(self) -> Type[tfl.RNN]:
        """Type of RNN cell."""
        pass

    @abstractmethod
    def _cell_state_index(self) -> Sequence[Tuple]:
        """Index of RNN encoded states in both directions (forward and backward)."""
        pass

    def rnn_states(self, encoded):
        states = []
        indexes = self._cell_state_index()
        for i in range(len(indexes)):
            i_forward, i_backward = indexes[i]
            cell_state = [encoded[i_forward], encoded[i_backward]]
            states.append(self.rnn_state_sae[i](self.rnn_state_concat[i](cell_state)))
        return states

    @staticmethod
    def rnn_output(encoded, decoded):
        output = tf.expand_dims(encoded[0], axis=1)
        return tf.tile(output, [1, tf.shape(decoded)[1], 1])

    def _s2s(self, y_past, x_future):
        cnn = self.cnn(y_past)
        rnn_encoded = self.rnn_encoder(cnn)
        rnn_decoded = self.rnn_decoder(x_future, initial_state=self.rnn_states(rnn_encoded))
        rnn_output = self.rnn_output(rnn_encoded, rnn_decoded)
        rnn_attention = self.rnn_attention([rnn_decoded, rnn_output])
        return self.s2s_concat([rnn_decoded, rnn_attention])


class S2S_LSTM(S2S_RNN):
    """:class:`S2S_RNN` based on Long Short-Term Memory (LSTM) cells, that have 2 states (hidden and cell states)."""

    def _cell_type(self) -> Type[tfl.RNN]:
        return tfl.LSTM

    def _cell_state_index(self) -> Sequence[Tuple]:
        return [(1, 3), [2, 4]]  # Hidden and cell states in both directions.


class S2S_GRU(S2S_RNN):
    """:class:`S2S_RNN` based on Gated Recurrent Unit (GRU) cells, that have only 1 state (hidden state)."""

    def _cell_type(self) -> Type[tfl.RNN]:
        return tfl.GRU

    def _cell_state_index(self) -> Sequence[Tuple]:
        return [(1, 2)]  # Hidden states in both directions.


class S2S(StrMixin):
    """Abstract wrapper around compile, fit and predict methods for Sequence-to-Sequence (S2S) models."""

    @staticmethod
    def transform(x: np.array, lookback: 'int > 0', horizon: 'int > 0'):
        """
        Transforms time series from 2D array with the shape (no. values, no. series) into 2x3D arrays with shapes
        (no. samples, no. lookback values, no. series) and (no. samples, no. values in horizon, no. series).

        :param x: 2D array where each column represents one time series.
        :param lookback: Length of the lookback range.
        :param horizon: Length of the forecast horizon.
        :return: 2x3D arrays with the values in lookback range and the values in the forecast horizon.
        """

        rows = range(x.shape[0] - lookback - horizon + 1)
        x_past = np.stack([x[i: i + lookback, :].T for i in rows], axis=-1).swapaxes(0, 2)
        x_future = np.stack([x[(i + lookback):(i + lookback + horizon), :].T for i in rows], axis=-1).swapaxes(0, 2)
        return x_past, x_future

    @abstractmethod
    def compile(self, x_count: int, y_count: int, lookback: int, horizon: int) -> any:
        """
        Constructs the regression model.

        :param x_count: Number of x series.
        :param y_count: Number of y series.
        :param lookback: Length of the lookback range.
        :param horizon: Length of the forecast horizon.
        :return: Compiled regression model.
        """
        pass

    @abstractmethod
    def fit(self, model, x_future: np.array, y_past: np.array, y_future: np.array) -> pd.DataFrame:
        """
        Trains the regression model.

        :param model: Compiled regression model.
        :param x_future: 3D array with the shape (no. samples, no. values in horizon, no. x series).
        :param y_past: 3D array with the shape (no. samples, no. lookback values, no. y series)
        :param y_future: 3D array with the shape (no. samples, no. values in horizon, no. y series).
        :return: Training metadata.
        """
        pass

    @abstractmethod
    def predict(self, model, x_future: np.array, y_past: np.array) -> np.array:
        """
        Predicts y values in the forecast horizon based on x values in the horizon and y values in the lookback range.

        :param model: Trained regression model.
        :param x_future: 3D array with the shape (no. samples, no. values in horizon, no. x series).
        :param y_past: 3D array with the shape (no. samples, no. lookback values, no. y series)
        :return: 3D array with the shape (no. samples, no. values in horizon, no. y series).
        """
        pass


class DeepS2S(S2S):
    def __init__(self,
                 hidden: Callable[[], S2S_DNN] = lambda: S2S_LSTM(),
                 output: Callable[[int], tfl.Layer] = lambda y_count: tfl.Dense(units=y_count),
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
                 loss: Union[str, Callable[[any, any], any]] = 'mse',
                 metric: Union[str, Callable[[any, any], any]] = 'mean_absolute_percentage_error',
                 epochs: 'int > 0' = 200,
                 batch: 'int > 0' = 128,
                 patience: 'int > 0' = 20,
                 validation: '0 <= int < 1' = 0,
                 verbose=False):
        """
        Deep learning approach to Sequence-to-Sequence (S2S) time series regression.

        :param hidden: Returns hidden S2S-DNN model.
        :param output: Returns output layer given the number of y variables.
        :param optimizer: Optimizer that will be used to train the model. See `tf.keras.optimizers`.
        :param loss: The loss function that will be used to train the model. See `tf.keras.losses`.
        :param metric: Metric that will be used to evaluate the model (does not affect training). See `tf.keras.metrics`.
        :param epochs: Number of training iterations over the complete training dataset.
        :param batch: Batch (sample) size for training iterations in each epoch.
        :param patience: Number of epochs to wait before early stopping.
        :param validation: Ratio of the training set to use for validation.
        :param verbose: Whether to print the training progress or not.
        """
        self.hidden = hidden
        self.output = output
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.epochs = epochs
        self.batch = batch
        self.patience = patience
        self.validation = validation
        self.verbose = verbose

    @ignore_warnings
    def compile(self, x_count: int, y_count: int, lookback: int, horizon: int) -> tf.keras.Model:
        y_past = tfl.Input(shape=(lookback, y_count))
        x_future = tfl.Input(shape=(horizon, x_count))
        hidden = self.hidden()([y_past, x_future])
        y_future = self.output(y_count)(hidden)
        model = tf.keras.Model(inputs=[y_past, x_future], outputs=y_future)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])
        if self.verbose:
            model.summary()
        return model

    @ignore_warnings
    def fit(self, model: tf.keras.Model, x_future: np.array, y_past: np.array, y_future: np.array) -> pd.DataFrame:
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=self.patience)
        h = model.fit(x=[y_past, x_future],
                      y=y_future,
                      epochs=self.epochs,
                      batch_size=self.batch,
                      callbacks=[es],
                      validation_split=self.validation,
                      verbose=int(self.verbose),
                      shuffle=False)
        return pd.DataFrame(h.history).rename_axis('epoch')

    @ignore_warnings
    def predict(self, model: tf.keras.Model, x_future: np.array, y_past: np.array) -> np.array:
        return model.predict([y_past, x_future], verbose=0)


class ShallowS2S(S2S):
    __regressor_types = {
        'esvm': SVR,
        'nsvm': NuSVR,
        'xgb': XGBRegressor,
        'rf': RandomForestRegressor,
        'dt': DecisionTreeRegressor
    }

    def __init__(self,
                 base: Union[str, Callable[[], any]] = 'eSVM',
                 incremental=False,
                 metric: Callable[[any, any], float] = mean_absolute_percentage_error):
        """
        Shallow learning approach that imitates Sequence-to-Sequence (S2S) time series regression in the recursive
        (iterative) manner, where the predicted value in one time step becomes an input for the value in the next step.

        :param base: Either a function that returns underlying regression model with 'fit(x, y)' and 'predict(x) -> y'
               methods, or a case-insensitive abbreviation for one of the following models: <br/>
            - 'eSVM': epsilon Support Vector Machines, <br/>
            - 'nSVM': nu Support Vector Machines, <br/>
            - 'XGB': eXtreme Gradient Boosting, <br/>
            - 'RF': Random Forest, <br/>
            - 'DT': Decision Tree.
        :param incremental: Whether to imitate incremental learning (true) or to retrain the model on each fit (false).
               The incremental learning is imitated by preserving the x and y values after each fit, and by joining the
               preserved values with the new ones next time the fit is called. Note that the true incremental learning
               can only be achieved with the right underlying regression model. On the other hand, the retraining is
               simply performed by replacing the old models with the new ones, trained only on the new data.
        :param metric: Metric that will be used to evaluate the model (does not affect the training).
        """
        self.base = base
        self.incremental = incremental
        self.metric = metric

    def compile(self, x_count: int, y_count: int, lookback: int, horizon: int) -> MultiRegressor:
        regressor_type = self.__regressor_types.get(self.base.lower()) if isinstance(self.base, str) else self.base
        return MultiRegressor(regressor_type, y_count, self.incremental)

    def fit(self, model: MultiRegressor, x_future: np.array, y_past: np.array, y_future: np.array) -> pd.DataFrame:
        x = [np.column_stack((x_future[:, 0, :], y_past[:, :, i])) for i in range(model.n)]
        y = [y_future[:, 0, i].reshape(-1, 1) for i in range(model.n)]
        y_fit = model.fit_predict(x, y)
        metrics = [self.metric(y[i], y_fit[i]) for i in range(model.n)]
        return pd.DataFrame({'metric': metrics}).rename_axis('model')

    def predict(self, model: MultiRegressor, x_future: np.array, y_past: np.array) -> np.array:
        y_past = y_past.copy()  # The past will iteratively change.
        y_pred = np.empty((np.shape(x_future)[0], np.shape(x_future)[1], model.n))
        i_samples = range(np.shape(y_pred)[0])
        i_timestamps = range(np.shape(y_pred)[1])
        for s in i_samples:
            for t in i_timestamps:
                x = [np.column_stack((x_future[s, t, :].reshape(1, -1), y_past[s, :, i].reshape(1, -1)))
                     for i in range(model.n)]
                y_pred[s, t, :] = np.column_stack(model.predict(x))
                y_past[s, :-1, :] = y_past[s, 1:, :]
                y_past[s, -1, :] = y_pred[s, t, :]
        return y_pred

    def __str__(self):
        b = self.base if isinstance(self.base, str) else self.base.__name__
        return f'{type(self).__name__}({b})'


class TimeSeriesRegressor(StrMixin):
    _scalers = {
        'minmax': MinMaxScaler,
        'absmax': MaxAbsScaler,
        'mean': StandardScaler,
        'median': RobustScaler
    }

    def __init__(self,
                 horizon: 'int > 0',
                 base: S2S = DeepS2S(),
                 t_encoder: TimeEncoder = TimeEncoders(MonthOfYear(), DayOfWeek(), TimeOfDay()),
                 x_selector: Callable[[], FeatureSelector] = lambda: JMIM(),
                 y_selector: Callable[[], LagSelector] = lambda: PACF(),
                 x_scaler: Literal['minmax', 'absmax', 'mean', 'median', None] = 'minmax',
                 y_scaler: Literal['minmax', 'absmax', 'mean', 'median', None] = 'mean'):
        """
        Sequence-to-Sequence (S2S) regression combined with time encoding, feature selection, and feature scaling.

        :param horizon: Length of the forecast horizon.
        :param base: Underlying S2S regression approach.
        :param t_encoder: Method for encoding input timestamps.
        :param x_selector: Returns a method for selecting exogenous input features.
        :param y_selector: Returns a method for selecting endogenous input features.
        :param x_scaler: Method for scaling input values: <br/>
            - 'minmax': Scales the values to [0, 1] range. <br/>
            - 'absmax': Scales the values to [-1, 1] range. <br/>
            - 'mean': Scales the values around mean so that mean = 0 and standard deviation = 1. <br/>
            - 'median': Scales the values around median so that Q1 = -1, Q2 = 0, and Q3 = 1. <br/>
            - None: no scaling will be applied if anything other than the above described scalers is specified.
        :param y_scaler: Method for scaling output values (see x_scaler).
        """

        def create_scaler(scaler_name: str):
            scaler_type = self._scalers.get(scaler_name)
            return FunctionTransformer(lambda _: _) if scaler_type is None else scaler_type()

        self.horizon = horizon
        self.base = base
        self.t_encoder = t_encoder
        self.x_selector = x_selector()
        self.y_selector = y_selector()
        self.x_scaler = create_scaler(x_scaler)
        self.y_scaler = create_scaler(y_scaler)

        self.model = None
        self.history: MutableSequence[pd.DataFrame] = []

    def __encode(self, t: pd.Index) -> pd.DataFrame:
        return pd.DataFrame([self.t_encoder(Time.py(_)) for _ in t], index=t)

    @property
    def lookback(self) -> int:
        """Length of the lookback range according to the specified y lag selector."""
        return self.y_selector.lag

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Trains the regression model on input (x) and output (y) time series.
        To properly encode time, x and y must have assigned DateTimeIndex.

        :param x: Input time series values indexed by timestamp.
        :param y: Output time series values indexed by timestamp.
        """
        xy = x.merge(y, left_index=True, right_index=True, sort=True)
        x = xy.iloc[:, :len(x.columns)]
        y = xy.iloc[:, len(x.columns):]

        t_encoded = self.__encode(x.index)
        if self.history:
            x_selected = pd.concat((t_encoded, self.x_selector.transform(x)), axis=1)
        else:
            self.y_selector.fit(y)
            x_selected = pd.concat((t_encoded, self.x_selector.fit_transform(x, y)), axis=1)
            self.model = self.base.compile(len(x_selected.columns), len(y.columns), self.lookback, self.horizon)

        _, x_future = self.base.transform(self.x_scaler.fit_transform(x_selected.values), self.lookback, self.horizon)
        y_past, y_future = self.base.transform(self.y_scaler.fit_transform(y.values), self.lookback, self.horizon)
        self.history.append(self.base.fit(self.model, x_future, y_past, y_future))
        return self

    def predict(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts y values in the forecast horizon based on x values in the horizon and y values in the lookback range.
        To properly encode time, x and y must have assigned DateTimeIndex.

        :param x: Input time series values in the forecast horizon indexed by timestamp.
        :param y: Output time series values in the lookback range indexed by timestamp.
        :return: Output time series values in the forecast horizon indexed by timestamp, or an empty data frame
                 if there are not enough x values in the horizon or y values in the lookback range.
        :raise Exception: Regression model cannot be used for prediction before training.
        """
        if not self.history:
            raise Exception('Regression model cannot be used for prediction before training.')
        if len(x) < self.horizon or len(y) < self.lookback:
            return pd.DataFrame()

        x = x.iloc[:self.horizon, :]
        y = y.iloc[-self.lookback:, :]
        x_selected = pd.concat((self.__encode(x.index), self.x_selector.transform(x)), axis=1)
        x_lookback = np.empty((self.lookback, len(x_selected.columns)))  # dummy values
        y_lookback = self.y_scaler.transform(y.values)  # true values
        x_horizon = self.x_scaler.transform(x_selected.values)  # true values
        y_horizon = np.empty((self.horizon, len(y.columns)))  # dummy values
        _, x_future = self.base.transform(np.vstack((x_lookback, x_horizon)), self.lookback, self.horizon)
        y_past, _ = self.base.transform(np.vstack((y_lookback, y_horizon)), self.lookback, self.horizon)
        y_pred_3d = self.base.predict(self.model, x_future, y_past)
        y_pred_2d = y_pred_3d.reshape(-1, np.shape(y_pred_3d)[2])
        y_pred_data = self.y_scaler.inverse_transform(y_pred_2d)
        y_pred_index = x.iloc[range(len(y_pred_2d)), :].index
        return pd.DataFrame(y_pred_data, index=y_pred_index, columns=y.columns)

    def summary(self) -> pd.DataFrame:
        """Returns regressor training scores indexed by the calls to the 'fit' method."""

        def _summary():
            for i in range(len(self.history)):
                h = self.history[i]
                s = h.reset_index()
                s['fit'] = i
                yield s.set_index(['fit'] + ['index' if i is None else i for i in h.index.names])

        return pd.concat(_summary(), axis=0)

    def __str__(self):
        return f'{type(self).__name__}({self.base})'


class MultiSeriesRegressor:
    def __init__(self, base: TimeSeriesRegressor):
        """
        Applies one time series regressor to multiple output time series data frames
        (e.g., time series of cumulants for multiple value types in the same cluster).

        :param base: Underlying time series regressor.
        """
        self.base = base

    def fit(self, x: pd.DataFrame, y: Sequence[pd.DataFrame]):
        """
        Trains the underlying regressor on multiple input and output time series (x and y).

        :param x: Input time series values indexed by timestamp.
        :param y: Output time series values indexed by timestamp.
        """
        self.base.fit(x, pd.concat(y, axis=1))
        return self

    def predict(self, x: pd.DataFrame, y: Sequence[pd.DataFrame]) -> Sequence[pd.DataFrame]:
        """
        Utilizes the underlying regressor on multiple input and output time series (x and y).

        :param x: Input time series values in the forecast horizon indexed by timestamp.
        :param y: Output time series values in the lookback range indexed by timestamp.
        :return: Output time series values in the forecast horizon indexed by timestamp.
        """
        pred_merged = self.base.predict(x, pd.concat(y, axis=1))
        pred_split = []
        col_prev = 0
        for i in range(len(y)):
            col_next = col_prev + len(y[i].columns)
            pred_split.append(pred_merged.iloc[:, col_prev:col_next])
            col_prev = col_next
        return pred_split

    def __str__(self):
        return str(self.base)

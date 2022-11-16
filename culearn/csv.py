import csv
import os

from culearn.base import *
from culearn.util import ignore_warnings, File, Time


class DataFrameCSV(StrMixin):
    def __init__(self, *path: str, overwrite=False, header=True, index=False, chunk: int = 0, sep=','):
        """
        Data frame wrapper around CSV file.

        :param path: Path to CSV file.
        :param overwrite: Whether to overwrite existing data.
        :param header: Whether to read/write data frame headers.
        :param index: Whether to write data frame index.
        :param chunk: Number of rows to read in one chunk (default: chunk < 1 => all rows).
        :param sep: CSV separator.
        """
        self.path = str(os.path.join(*path))
        self.overwrite = overwrite
        self.header = header
        self.index = index
        self.chunk = chunk
        self.sep = sep

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def read(self) -> Iterable[pd.DataFrame]:
        if os.path.exists(self.path):
            h = "infer" if self.header else None
            if self.chunk < 1:
                yield pd.read_csv(self.path, sep=self.sep, header=h)
            else:
                for x in pd.read_csv(self.path, sep=self.sep, header=h, chunksize=self.chunk):
                    yield x

    def write(self, x: pd.DataFrame):
        File.make_dir(os.path.dirname(self.path))
        if not os.path.exists(self.path):
            x.to_csv(self.path, header=self.header, index=self.index, sep=self.sep)
        else:
            mode = 'w' if self.overwrite else 'a'
            save_header = (self.header and self.overwrite)
            x.to_csv(self.path, header=save_header, index=self.index, sep=self.sep, mode=mode)


class TimeSeriesCSV(PandasTimeSeries, StreamingTimeSeries):
    def __init__(self,
                 ts_id: TimeSeriesID,
                 ts_time: Union[int, str],
                 ts_value: Union[int, str],
                 *df: DataFrameCSV):
        """
        Wraps around one time series from one or more CSV files, primarily for testing purposes.
        Note that custom :class:`TimeSeries` implementation can be used if the data should be stored to database.

        :param ts_time: Timestamp column index or column name.
        :param ts_value: Value column index or column name.
        :param df: Data frame wrapper around CSV file.
        """
        super().__init__(ts_id)
        self.ts_time = ts_time
        self.ts_value = ts_value
        self.df = df

    def series(self) -> pd.Series:
        ts = pd.DataFrame()
        for df_csv in self.df:
            for df in df_csv.read():
                ts_time = df[self.ts_time] if self.ts_time is str else df.iloc[:, self.ts_time]
                ts_value = df[self.ts_value] if self.ts_value is str else df.iloc[:, self.ts_value]
                ts = pd.concat((ts, pd.concat([ts_time, ts_value], axis=1)), axis=0)

        ts.columns = ['timestamp', 'value']
        ts.set_index('timestamp', inplace=True)
        ts.index = pd.DatetimeIndex(ts.index)
        return ts['value']

    def stream(self) -> Iterable[TimeSeriesTuple]:
        for df_csv in self.df:
            for df in df_csv.read():
                for _, row in df.iterrows():
                    ts_time = Time.py(row[self.ts_time])
                    ts_value = float(row[self.ts_value])
                    yield TimeSeriesTuple(ts_time, ts_value)


class MultiSeriesCSV(MultiSeries):
    def __init__(self, *directory: str):
        """
        :class:`MultiSeries` with in-memory buffer and CSV-based disc, primarily for testing purposes.
        Note that custom :class:`MultiSeries` implementation can be used if the data should be stored to database.
        """
        self.__last: Optional[TimeSeriesTuple] = None
        self.__buffer: MutableMapping[TimeResolution, TimeSeriesSegment] = {}
        self.directory = directory
        File.make_dir(*directory)

    def __path(self, resolution: TimeResolution):
        return os.path.join(*self.directory, f'{int(resolution.length)}.csv')

    def last(self) -> Optional[TimeSeriesTuple]:
        return self.__last

    def buffer(self, resolution: TimeResolution) -> Optional[TimeSeriesSegment]:
        return self.__buffer.get(resolution)

    def disc(self, resolution: TimeResolution) -> Sequence[TimeSeriesSegment]:
        csv_path = self.__path(resolution)
        if not os.path.exists(csv_path):
            return []

        segments = []
        with open(csv_path, 'r') as csv_file:
            for row in csv.reader(csv_file):
                segments.append(TimeSeriesSegment.from_csv(row))
        return segments

    def update_last(self, last: Optional[TimeSeriesTuple]):
        self.__last = last

    def update_buffer(self, resolution: TimeResolution, segment: Optional[TimeSeriesSegment]):
        self.__buffer[resolution] = segment

    def extend_disc(self, resolution: TimeResolution, *segment: TimeSeriesSegment):
        csv_path = self.__path(resolution)
        with open(csv_path, mode='a', newline='') as csv_file:
            csv.writer(csv_file).writerows([_.to_csv() for _ in segment])


class ConcatSeriesCSV(StrMixin):
    def __init__(self,
                 ts_id: Callable[[str, pd.Series], TimeSeriesID],
                 ts_time: Union[int, str],
                 ts_value: Union[int, str],
                 *df: DataFrameCSV):
        """
        Wraps around multiple concatenated time series from one or more CSV files.

        :param ts_id: Takes CSV file path and CSV row as inputs and returns time series ID as output.
        :param ts_time: Time series timestamp column index or column name.
        :param ts_value: Time series value column index or column name.
        :param df: Data frame wrapper around CSV file.
        """
        super().__init__()
        self.ts_id = ts_id
        self.ts_time = ts_time
        self.ts_value = ts_value
        self.df = df

    @ignore_warnings
    def read(self):
        """Returns time series IDs paired with pandas time series."""
        for df_csv in self.df:
            for df in df_csv.read():
                ts_id = df.apply(lambda _: self.ts_id(df_csv.path, _), axis=1)
                ts_time = df[self.ts_time] if self.ts_time is str else df.iloc[:, self.ts_time]
                ts_value = df[self.ts_value] if self.ts_value is str else df.iloc[:, self.ts_value]
                ts = pd.concat([ts_id, ts_time, ts_value], axis=1)
                ts.columns = ['ID', 'timestamp', 'value']
                ts = ts.pivot_table(columns='ID', index='timestamp', values='value')
                ts.index = pd.DatetimeIndex(ts.index)
                for series_id, series in ts.items():
                    yield series_id, series.dropna()

    @ignore_warnings
    def split(self, file_path: Callable[[TimeSeriesID], Sequence[str]], overwrite=True) -> Sequence[TimeSeries]:
        """
        Creates one CSV file for each time series.

        :param file_path: Takes time series ID as input and returns new CSV file path as output.
        :param overwrite: Whether to overwrite the existing files.
        :return: Wrappers around individual time series from created CSV files.
        """
        id2path: MutableMapping[TimeSeriesID, Sequence[str]] = {}
        for ts_id, ts_part in self.read():
            path = id2path.get(ts_id)
            if path is None:
                id2path[ts_id] = path = file_path(ts_id)
                df_csv = DataFrameCSV(*path, index=True, overwrite=overwrite)
            else:
                df_csv = DataFrameCSV(*path, index=True)

            df_csv.write(ts_part.to_frame())

        return [TimeSeriesCSV(_id, 0, 1, DataFrameCSV(*_path)) for _id, _path in id2path.items()]

import gzip
import tarfile
import urllib.parse
from random import randrange
from time import sleep

import meteostat as meteo
import numpy as np
import requests
import wget
import xmltodict
import zipfile_deflate64 as zipfile
from culearn.csv import *
from culearn.util import File, Time
from meteostat.series.fetch import fetch as meteo_fetch
from py7zr import py7zr
from sklearn.neighbors import NearestNeighbors


class DataSource(StrMixin):
    """Abstract data source providing the access to :class:`PredictionDataset`."""

    @abstractmethod
    def dataset(self) -> PredictionDataset:
        pass


class _Download(StrMixin):
    """Commonly used download methods."""

    @classmethod
    def get(cls, url: str, path: str) -> str:
        try:
            return wget.download(url=url, out=path)
        except:
            response = requests.get(url)
            with open(path, 'wb') as f:
                f.write(response.content)
            return path

    @classmethod
    def zip(cls, url: str, zip_dir: str, zip_name: str = 'download'):
        File.make_dir(zip_dir)
        zip_path = os.path.join(zip_dir, f'{zip_name}.zip')
        if not os.path.exists(zip_path):
            zip_path = cls.get(url, zip_path)

        with zipfile.ZipFile(zip_path, mode='r') as extractor:
            extractor.extractall(zip_dir)

    @classmethod
    def seven_zip(cls, url: str, zip_dir: str, zip_name: str = 'download'):
        File.make_dir(zip_dir)
        zip_path = os.path.join(zip_dir, f'{zip_name}.7z')
        if not os.path.exists(zip_path):
            zip_path = cls.get(url, zip_path)

        with py7zr.SevenZipFile(zip_path, mode='r') as extractor:
            extractor.extractall(zip_dir)

    @classmethod
    def umass(cls, url: str, zip_dir: str, zip_name: str):
        url_param = {'name': 'x', 'affiliation': 'x', 'country': 'x', 'email': 'x', 'submit': 'x'}

        File.make_dir(zip_dir)
        zip_path = os.path.join(zip_dir, f'{zip_name}.tar.gz')
        if not os.path.exists(zip_path):
            response = requests.post(url, data=url_param, stream=True)
            with open(zip_path, 'wb') as zip_file:
                zip_file.write(gzip.decompress(response.content))

        with tarfile.open(zip_path) as extractor:
            extractor.extractall(zip_dir)


class Weather(StrMixin):
    def __init__(self, time_column='Timestamp'):
        """Abstract weather data provider."""
        self.time_column = time_column

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass


class OnlineWeather(Weather):
    def __init__(self, interval: TimeInterval, location: str):
        """Abstract online weather data provider."""
        super().__init__()
        self.location = location
        self.interval = interval

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass


class WorldWeather(OnlineWeather):
    def __init__(self, interval: TimeInterval, location: str, api_key: str):
        """Weather data from https://worldweatheronline.com."""
        super().__init__(interval, location)
        self.api_key = api_key

    def read(self) -> pd.DataFrame:
        def history(i: TimeInterval) -> bytes:
            _start_date = i.start.date()
            _end_date = (i.end - timedelta(hours=1)).date()
            url = 'https://api.worldweatheronline.com/premium/v1/past-weather.ashx'
            url_param = {'key': self.api_key, 'q': self.location, 'date': str(_start_date), 'enddate': str(_end_date),
                         'format': 'xml', 'tp': '1'}
            return requests.post(url, data=url_param).content

        def forecast() -> bytes:
            url = 'https://api.worldweatheronline.com/premium/v1/weather.ashx'
            url_param = {'key': self.api_key, 'q': self.location, 'num_of_days': 14, 'format': 'xml', 'tp': '1'}
            return requests.post(url, data=url_param).content

        def convert(response: bytes, i: TimeInterval, print_query: bool) -> Iterable[dict]:
            x = xmltodict.parse(response)
            if print_query:
                query = x['data']['request']['query']
                print(f'Downloading weather data from worldweatheronline.com for {query}.')
            for dates in x['data']['weather']:
                date = Time.py(dates['date'])
                for values in dates['hourly']:
                    hhmm = int(values['time'])
                    timestamp = date + timedelta(hours=hhmm / 100, minutes=hhmm % 100)
                    if i.contains(timestamp):
                        yield {
                            self.time_column: timestamp,
                            'Temperature [°C]': float(values['tempC']),
                            'Feels Like [°C]': float(values['FeelsLikeC']),
                            'Dew Point [°C]': float(values['DewPointC']),
                            'Heat Index [°C]': float(values['HeatIndexC']),
                            'Cloud Cover [%]': float(values['cloudcover']),
                            'Humidity [%]': float(values['humidity']),
                            'Precipitation [mm]': float(values['precipMM']),
                            'Pressure [mbar]': float(values['pressure']),
                            'UV Index': float(values['uvIndex']),
                            'Visibility [km]': float(values['visibility']),
                            'Wind Chill [°C]': float(values['WindChillC']),
                            'Wind Direction [°]': float(values['winddirDegree']),
                            'Wind Gust [km/h]': float(values['WindGustKmph']),
                            'Wind Speed [km/h]': float(values['windspeedKmph'])
                        }

        past = []
        future = []
        today = datetime.utcnow().date()
        today = datetime(today.year, today.month, today.day)
        if self.interval.end > today:
            step = TimeInterval(today, self.interval.end)
            future.extend(convert(forecast(), step, True))
        if self.interval.start < today:
            month = timedelta(days=30)  # Workaround for the server limit for weather history.
            step = TimeInterval(self.interval.start, min(self.interval.start + month, self.interval.end, today))
            while step.start < step.end:
                if past or future:
                    sleep(10)  # Workaround for the server limit between requests.
                past.extend(convert(history(step), step, not past and not future))
                new_start = step.start + month
                new_end = min(new_start + month, self.interval.end, today)
                step = TimeInterval(new_start, new_end)

        return pd.DataFrame(past + future)


class MeteoWeather(OnlineWeather):
    def __init__(self, interval: TimeInterval, location: str):
        """Weather data from https://meteostat.net."""
        super().__init__(interval, location)

    @staticmethod
    def __location(location: str):
        """Returns full location info, latitude and longitude."""
        url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(location) + '?format=json'
        response = requests.get(url).json()[0]
        return str(response['display_name']), float(response['lat']), float(response['lon'])

    def read(self) -> pd.DataFrame:
        x = self.__location(self.location)
        print(f'Downloading weather data from meteostat.net for {x[0]}.')

        point = meteo.Point(x[1], x[2])
        data = meteo_fetch(meteo.Hourly(point, self.interval.start, self.interval.end - timedelta(hours=1)))
        data.reset_index(inplace=True)
        take_columns = {
            'time': self.time_column,
            'temp': 'Temperature [°C]',
            'dwpt': 'Dew Point [°C]',
            'rhum': 'Humidity [%]',
            'prcp': 'Precipitation [mm]',
            'snow': 'Snow Depth [mm]',
            'wdir': 'Wind Direction [°]',
            'wspd': 'Wind Speed [km/h]',
            'wpgt': 'Wind Gust [km/h]',
            'pres': 'Pressure [mbar]',
            'tsun': 'Sunshine [min]'
        }

        drop_columns = set(data.columns).difference(take_columns.keys())
        data.drop(drop_columns, axis=1, inplace=True)
        data.rename(columns=take_columns, inplace=True)
        return data


class UMassWeather(Weather):
    def __init__(self, directory: str):
        """Weather data from UMass repository: https://traces.cs.umass.edu/index.php/Smart/Smart."""
        super().__init__()
        self.directory = directory

    def read(self) -> pd.DataFrame:
        def fahrenheit_to_celsius(fahrenheit):
            return (fahrenheit - 32) / 1.8

        def inches_to_mm(inches):
            return inches * 2.54

        def miles_to_km(miles):
            return miles * 1.609344

        def ratio_to_percentage(ratio):
            return ratio * 100

        df_csv = [DataFrameCSV(self.directory, 'apartment-weather', f'apartment{_}.csv') for _ in range(2014, 2017)]
        if [_ for _ in df_csv if not _.exists()]:
            _Download.umass('https://tinyurl.com/4h4usr2z', self.directory, 'weather')
            missing = [_.path for _ in df_csv if not _.exists()]
            if missing:
                m = '\n'.join(missing)
                raise Exception(f'Missing:\n{m}')

        source = pd.concat([_df for _ in df_csv for _df in _.read()])
        target = pd.DataFrame()
        target[self.time_column] = pd.to_datetime(source['time'], unit='s')
        target['Temperature [°C]'] = fahrenheit_to_celsius(source['temperature'])
        target['Feels Like [°C]'] = fahrenheit_to_celsius(source['apparentTemperature'])
        target['Dew Point [°C]'] = fahrenheit_to_celsius(source['dewPoint'])
        target['Cloud Cover [%]'] = ratio_to_percentage(source['cloudCover'])
        target['Humidity [%]'] = ratio_to_percentage(source['humidity'])
        target['Precipitation [mm]'] = inches_to_mm(source['precipIntensity'])
        target['Pressure [mbar]'] = source['pressure']
        target['Visibility Index'] = source['visibility']
        target['Wind Direction [°]'] = source['windBearing']
        target['Wind Speed [km/h]'] = miles_to_km(source['windSpeed'])
        return target


class SmartMeterDataSource(DataSource):
    def __init__(self, directory: str, interval: TimeInterval, calendar_iso_code: str, weather: Weather):
        """Abstract data source with smart meter and weather data."""
        super().__init__()
        self.directory = directory
        self.interval = interval
        self.calendar = Time.calendar(calendar_iso_code)
        self.weather = weather

    def __raw_weather_data(self):
        x_csv = DataFrameCSV(self.directory, f'{str(self.weather)}.csv')
        if x_csv.exists():
            for x in x_csv.read():
                return x

        x = self.weather.read()
        x_csv.write(x)
        return x

    def _weather_data(self) -> pd.DataFrame:
        x = self.__raw_weather_data()
        x.set_index(self.weather.time_column, inplace=True)
        x.index = pd.DatetimeIndex(x.index)
        x = x.groupby(x.index).mean()
        x.fillna(0, inplace=True)
        return x

    @abstractmethod
    def _smart_meter_data(self) -> Sequence[TimeSeries]:
        pass

    def dataset(self) -> PredictionDataset:
        """
        Prepares time series forecasting dataset (e.g., downloads and unzips data files, splits large files with
        multiple time series into smaller ones with individual time series to support parallel processing, etc.).
        """
        x = self._weather_data()
        y = self._smart_meter_data()
        return PredictionDataset(x, y)


class LCL(SmartMeterDataSource):
    def __init__(self, directory: str, weather_type: Type = MeteoWeather, *args, **kwargs):
        """
        :class:`SmartMeterDataSource` with the data from the UK Power Networks led Low Carbon London (LCL) project.

        - Source: https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households
        - No. consumers: 5560 (households)
        - Type of values: power consumption [kWh]
        - Sampling interval: 30 minutes
        - Temporal coverage: November 2011 - February 2014
        - Geographical coverage: London, UK
        - Weather: :class:`OnlineWeather` subclass (:class:`MeteoWeather`, :class:`WorldWeather`, or a custom subclass)

        :param directory: Download directory.
        :param weather_type: Type of weather provider.
        :param args: Weather provider arguments other than interval and location.
        :param kwargs: Weather provider key-value arguments other than interval and location.
        """
        i = TimeInterval(datetime(2011, 11, 1), datetime(2014, 3, 1))
        w = weather_type(interval=i, location='London, UK', *args, **kwargs)
        super().__init__(directory, i, 'GB', w)

    def _smart_meter_data(self) -> Sequence[TimeSeries]:
        id2csv = {TimeSeriesID(f.split('house_')[-1].split('.csv')[0], 'P', 'consumer'): DataFrameCSV(f)
                  for f in File.paths(self.directory, 'houses')
                  if f.split('household_')[-1].endswith('.csv')}
        if len(id2csv) >= 5560:
            return [TimeSeriesCSV(_id, 0, 1, _csv) for _id, _csv in id2csv.items()]

        df_csv = DataFrameCSV(self.directory, 'CC_LCL-FullData.csv', chunk=1000)
        if not df_csv.exists():
            _Download.zip('https://tinyurl.com/wdew5kuz', self.directory)
            if not df_csv.exists():
                raise Exception(f'Missing: {df_csv.path}')

        concat_csv = ConcatSeriesCSV(lambda _, row: TimeSeriesID(str(row[0]), 'P', 'consumer'), 2, 3, df_csv)
        return concat_csv.split(lambda _: [self.directory, 'houses', f'house_{_.source}.csv'])


class REFIT(SmartMeterDataSource):
    def __init__(self, directory: str, weather_type: Type = MeteoWeather, *args, **kwargs):
        """
        :class:`SmartMeterDataSource` with the REFIT Electrical Load Measurements data.

        - Source: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned
        - No. consumers: 20 (households)
        - Type of values: power consumption [W]
        - Sampling interval: 8 seconds
        - Temporal coverage: October 2013 - June 2015
        - Geographical coverage: Loughborough, UK
        - Weather: :class:`OnlineWeather` subclass (:class:`MeteoWeather`, :class:`WorldWeather`, or a custom subclass)

        :param directory: Download directory.
        :param weather_type: Type of weather provider.
        :param args: Weather provider arguments other than interval and location.
        :param kwargs: Weather provider key-value arguments other than interval and location.
        """
        i = TimeInterval(datetime(2013, 10, 1), datetime(2015, 7, 1))
        w = weather_type(interval=i, location='Loughborough, UK', *args, **kwargs)
        super().__init__(directory, i, 'GB', w)

    def _smart_meter_data(self) -> Sequence[TimeSeries]:
        id2csv = {TimeSeriesID(str(i), 'P', 'consumer'): DataFrameCSV(self.directory, f'House_{i}.csv')
                  for i in range(1, 22) if i != 14}

        if [_ for _ in id2csv.values() if not _.exists()]:
            _Download.seven_zip('https://tinyurl.com/msc39r85', self.directory)
            missing = [_.path for _ in id2csv.values() if not _.exists()]
            if missing:
                m = '\n'.join(missing)
                raise Exception(f'Missing:\n{m}')

        return [TimeSeriesCSV(_id, 0, 2, _csv) for _id, _csv in id2csv.items()]


class SGSC(SmartMeterDataSource):
    def __init__(self, directory: str, weather_type: Type = MeteoWeather, *args, **kwargs):
        """
        :class:`SmartMeterDataSource` with the Smart Grid Smart City (SGSC) project data.

        - Source: https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data
        - No. consumers: 13.735 (households)
        - Type of values: power consumption [kW]
        - Sampling interval: 30 minutes
        - Temporal coverage: October 2011 - February 2014
        - Geographical coverage: Newcastle, Australia
        - Weather: :class:`OnlineWeather` subclass (:class:`MeteoWeather`, :class:`WorldWeather`, or a custom subclass)

        :param directory: Download directory.
        :param weather_type: Type of weather provider.
        :param args: Weather provider arguments other than interval and location.
        :param kwargs: Weather provider key-value arguments other than interval and location.
        """
        i = TimeInterval(datetime(2011, 10, 1), datetime(2014, 3, 1))
        w = weather_type(interval=i, location='Newcastle, Australia', *args, **kwargs)
        super().__init__(directory, i, 'AU-NSW', w)

    def _smart_meter_data(self) -> Sequence[TimeSeries]:
        id2csv = {TimeSeriesID(f.split('house_')[-1].split('.csv')[0], 'P', 'consumer'): DataFrameCSV(f)
                  for f in File.paths(self.directory, 'houses')
                  if f.split('household_')[-1].endswith('.csv')}
        if len(id2csv) >= 13735:
            return [TimeSeriesCSV(_id, 0, 1, _csv) for _id, _csv in id2csv.items()]

        df_csv = DataFrameCSV(self.directory, 'CD_INTERVAL_READING_ALL_NO_QUOTES.csv', chunk=1000)
        if not df_csv.exists():
            _Download.seven_zip('https://tinyurl.com/y6brtc9d', self.directory)
            if not df_csv.exists():
                raise Exception(f'Missing: {df_csv.path}')

        concat_csv = ConcatSeriesCSV(lambda _, row: TimeSeriesID(str(row[0]), 'P', 'consumer'), 1, 4, df_csv)
        return concat_csv.split(lambda _: [self.directory, 'houses', f'house_{_.source}.csv'])


class UMass(SmartMeterDataSource):
    def __init__(self, directory: str):
        """
        :class:`SmartMeterDataSource` with the UMass apartments data.

        - Source: https://traces.cs.umass.edu/index.php/Smart/Smart
        - No. consumers: 113 (single-family apartments)
        - Type of values: power consumption [kW]
        - Sampling interval: 1-15 minutes
        - Temporal coverage: August 2014 - December 2016
        - Geographical coverage: New England, USA
        - Weather: :class:`UMassWeather`

        :param directory: Download directory.
        """
        i = TimeInterval(datetime(2014, 8, 1), datetime(2016, 12, 15))
        super().__init__(directory, i, 'US-NH', UMassWeather(directory))

    def _smart_meter_data(self) -> Sequence[TimeSeries]:
        id_csv = [(TimeSeriesID(str(a), 'P', 'consumer'),
                   DataFrameCSV(self.directory, 'apartment', str(y), f'Apt{a}_{y}.csv', header=False))
                  for y in range(2014, 2017) for a in range(1, 114)
                  if not (y == 2014 and a in [3, 6, 21, 65, 94, 112])]

        if [_csv for _, _csv in id_csv if not _csv.exists()]:
            _Download.umass('https://tinyurl.com/2p9cvzcp', self.directory, 'electrical')
            missing = [_csv.path for _, _csv in id_csv if not _csv.exists()]
            if missing:
                m = '\n'.join(missing)
                raise Exception(f'Missing:\n{m}')

        id2csv = {}
        for _id, _csv in id_csv:
            id2csv.setdefault(_id, []).append(_csv)

        return [TimeSeriesCSV(_id, 0, 1, *_csv) for _id, _csv in id2csv.items()]


class GeneratedTimeSeries(PandasTimeSeries, StreamingTimeSeries):
    def __init__(self, ts_id=TimeSeriesID(), ts=pd.Series(dtype=float)):
        """Generated time series for testing purposes."""
        super().__init__(ts_id)
        self.ts = ts

    def series(self) -> pd.Series:
        return self.ts

    def stream(self) -> Iterable[TimeSeriesTuple]:
        for t, v in self.series().items():
            yield TimeSeriesTuple(Time.py(t), float(v))


class GeneratedDataSource(DataSource):
    def __init__(self,
                 x_count=10,
                 y_count=100,
                 interval=TimeInterval(datetime(2021, 1, 1), datetime(2022, 1, 1)),
                 resolution=TimeResolution(hours=1),
                 calendar_iso_code='US',
                 degree=3):
        """Generator of polynomial sinusoidal time series data with random noise coefficients, for testing purposes."""
        self.x_count = x_count
        self.y_count = y_count
        self.interval = interval
        self.resolution = resolution
        self.calendar = Time.calendar(calendar_iso_code)
        self.degree = degree

    def dataset(self) -> PredictionDataset:
        y_id = [TimeSeriesID(str(self.y_count + i // 2), str(i % 2), str(i % 2)) for i in range(self.y_count)]
        y = [GeneratedTimeSeries(_, self(1).iloc[:, 0]) for _ in y_id]
        return PredictionDataset(self(self.x_count), y)

    def __call__(self, count: int, prefix='X') -> pd.DataFrame:
        """Generates specified number of time series."""

        def fake_one(length: int):
            x = np.sin(np.linspace(0, length, length))
            y = np.zeros(length)
            for p in range(self.degree):
                y += np.power(x, p) * np.random.rand(len(x))
            return y

        timestamps = [_.start for _ in self.resolution.steps(self.interval)]
        df = pd.DataFrame([fake_one(len(timestamps)) for _ in range(count)], columns=timestamps).T
        df.columns = [f'{prefix}{i}' for i in range(count)]
        return df


class SyntheticSample(StrMixin):
    def __init__(self, k: int = 5, noise: Callable[[], float] = np.random.normal):
        """
        Generates synthetic samples using the Synthetic Minority Over-sampling TEchnique (SMOTE).

        :param k: The number of nearest neighbors to use for generating synthetic samples.
        :param noise: The number that will be multiplied by the difference between a sample and one of its randomly
         selected nearest neighbors, to generate new sample.
        """
        self.k = k
        self.noise = noise

    def __call__(self, x: pd.DataFrame) -> pd.DataFrame:
        synthetic = np.zeros_like(x)
        # Using k+1 to find k nearest neighbors because sklearn returns the point itself along with the neighbors:
        knn = NearestNeighbors(n_neighbors=self.k + 1).fit(x.values)
        for i in range(len(x)):
            # Selecting one of the k nearest neighbors randomly:
            n = knn.kneighbors(x.iloc[i].values.reshape(1, -1), return_distance=False)[0][randrange(1, self.k + 1)]
            # Multiplying the difference between the point and the neighbor with the specified noise:
            for j in range(len(x.columns)):
                diff = x.iloc[n, j] - x.iloc[i, j]
                synthetic[i, j] = x.iloc[i, j] + diff * self.noise()

        return pd.DataFrame(synthetic, index=x.index, columns=x.columns)

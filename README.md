# culearn: Cumulant Learning in Python

[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![build](https://github.com/igormanojlovic/culearn/actions/workflows/build.yml/badge.svg)](https://github.com/igormanojlovic/culearn/actions/workflows/build.yml)
[![cover](https://coveralls.io/repos/github/igormanojlovic/culearn/badge.svg)](https://coveralls.io/github/igormanojlovic/culearn)
[![linkedin](https://img.shields.io/badge/LinkedIn-informational?style=flat&logo=linkedin&logoColor=white&color=0D76A8)](https://www.linkedin.com/in/igormanojlovic/)

Cumulant Learning is a pattern recognition method designed to support probabilistic time series forecasting with high and stable forecast accuracy and short execution time when dealing with high-resolution time series coming from numerous data sources. This method has been publised in IEEE Transactions on Smart Grid, so if you are using this software for writing your own paper please cite the reference [^1] (preferably) or one of the references [^2][^3][^4][^5] (if you are using a specific subpart of the solution published in one of those papers).

## How does it work?

Cumulant Learning is designed to recognize time series patterns as follows:

1. Input time series are approximated and clustered to obtain time series of normalized cluster-level cumulants.
2. Regression models are used to capture local trends and time dependencies in the obtained cumulants based on lagged cumulant values, encoded time (calendar) features, and other exogenous features (such as weather in load forecasting).
3. Trained regressors are used to predict the future cluster-level cumulants. Then, these cumulants are transformed into time series of normalized cluster-level quantiles based on Cornish–Fisher expansion. Finally, these quantiles are denormalized and combined into prediction intervals for individual time series.

## How to use the package?

Install the package using pip:

```python
pip install git+https://github.com/igormanojlovic/culearn#egg=culearn
```

The following code snippet shows a simple approach to Cumulant Learning, with a step-by-step explanation below (for more examples, please look at the Jupyter [notebooks](https://github.com/igormanojlovic/culearn/tree/main/examples)).

```python
from culearn.data import *
from culearn.learn import *

# Step 1: Prepare the dataset
source = GeneratedDataSource(resolution=TimeResolution(minutes=30))
dataset = source.dataset()

# Step 2: Prepare the transformer
transform_encoders = TimeEncoders(MonthOfYear(), DayType(source.calendar), TimeOfDay())
transformer = CumulantTransform(encoder=transform_encoders)

# Step 3: Prepare the regressors
one_day = timedelta(1)
horizon = int(one_day / source.resolution)
regressor_encoders = TimeEncoders(MonthOfYear(), DayOfWeek(), TimeOfDay(), Holiday(source.calendar))
regressor = lambda: TimeSeriesRegressor(horizon, t_encoder=regressor_encoders, base=DeepS2S(epochs=1))

# Step 4: Prepare the learner
learner = CumulantLearner(dataset, source.resolution, transformer, regressor)

# Step 5: Train the learner
fit_interval = TimeInterval(source.interval.start, source.interval.end - one_day)
learner.fit(fit_interval, verbose=True)

# Step 6: Test the learner
pred_intervals = learner.predict(fit_interval.end, p=[0.75, 0.95, 0.99], clusters=True, members=True)
for pi in pred_intervals: 
  print(pi.ts_id)
  display(pi.to_frame())
  break
pred_cumulants = learner.predict_cumulants(fit_interval.end)
for pc in pred_cumulants:  
  print(pc.ts_id)
  display(pc)
  break
pred_figure = learner.figure(fit_interval.end, p=[0.75, 0.95, 0.99])
pred_figure.show()
```

### Step 1: Prepare the dataset

The first step is to create an instance of a data source wrapper and to retrieve a dataset for Cumulant Learning. The code snippet above uses synthetic dataset from `GeneratedDataSource`, but you can implement your own subclass of `DataSource` or use one of the existing implementations: `LCL`[^6], `REFIT`[^7], `SGSC`[^8], and `UMass`[^9]. The existing implementations will download smart meter and weather data for load forecasting from external data sources, unzip the data files, split the larger CSV files into smaller ones to support parallel processing, etc. This might take a while the first time, but makes the rest of the process much faster.

### Step 2: Prepare the transformer

The next step is to create the `transformer` that will be used to approximate and cluster input time series to obtain the time series of normalized cluster-level cumulants. The `transformer` will reduce the amount of time series data before clustering without losing significant information along the way, utilizing: 1) time series approximation at forecast time resolution; 2) aggregation of approximated values over encoded time features, and 3) feature extraction based on principal component analysis. All these methods can be configured to fit the specific needs of your application. 

### Step 3: Prepare the regressors

The next step is to prepare the `regressor` function that will be used to create one regressor for each cluster produced by the `transformer`. Each regressor is a wrapper around: 1) time encoding (used to extend exogenous inputs with time features); 2) feature selection (used to choose significant exogenous inputs); 3) lag selection (used to choose the lag for endogenous inputs); 4) feature scaling used to normalize the input values), and finally 5) regression model training and prediction (used learn the cumulants and predict the future values). All these methods can be configured as well, to fit the specific needs of your application. For example, in the code snippet above, we train a Sequence-to-Sequence (S2S) deep learning regression model in only one epoch simply for demonstration purposes.  

### Step 4: Prepare the learner

The final preparation step is to create the `learner` utilizing `transformer` and `regressor`. In the code snippet above, the `learner` is configured to learn half-hour time series patterns and to predict the patterns for 48 time steps ahead (one day ahead).

### Step 5: Train the learner

Once created, the `learner` can be trained using the data from an arbitrary time interval (from the specified dataset). For example, in the code snippet above, one day of history is left for testing while all the other days are used for training. 

Note that the `fit` method can be called multiple times if you wish to retrain the `learner`. However, if you wish only to update the underlying regressors with new data there is the `update` method, which enables incremental learning.

### Step 6: Test the learner

Once trained, the `learner` can be used to obtain cluster- and/or member-level prediction intervals in the specified forecast horizon at the specified time resolution using the `predict` method. This method will yield the `TimeSeriesPrediction` objects containing the series of lower and upper bounds of the requested intervals for specified probability values. In the code snippet above, the `predict` method will yield both cluster- and member-level 75%, 95%, and 99% prediction intervals at the half-hour resolution for the next day (48 steps ahead).

Additionally, you can use the `predict_cumulants` method to predict only the cluster-level cumulants. Furthermore, you can use the `figure` method to create a [plotly](https://plotly.com) figure that shows the cluster-level prediction intervals against the actual (but normalized) time series values. Alternatively, you can use the `evaluate` method to run initial training as well as incremental testing and updates, to obtain and evaluate the prediction results over longer time intervals (more in the [notebooks](https://github.com/igormanojlovic/culearn/tree/main/examples)).

## Background story

Different parts of the [culearn](https://github.com/igormanojlovic/culearn) package were originally implemented in [MTSR](https://github.com/igormanojlovic/MTSR) and [TimeSeriesR](https://github.com/igormanojlovic/TimeSeriesR) packages (in C#, T-SQL, and R). These two packages require many manual steps for preparing the input datasets, configuring the provided tools for data processing, and combining the tools into a single data processing pipeline that ultimately delivers the forecasting results. To minimize the number of manual steps, the complete data processing pipeline is now available in the [culearn](https://github.com/igormanojlovic/culearn) package, written entirely in Python.

## References

[^1]: Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić, Darko Čapko, *Cumulant Learning: Highly Accurate and Computationally Efficient Load Pattern Recognition Method for Probabilistic STLF at the LV Level*, IEEE Transactions on Smart Grid, 2024, DOI: [10.1109/TSG.2024.3481894](https://doi.org/10.1109/TSG.2024.3481894)

[^2]: Igor Manojlović, *Kratkoročna probabilistička prognoza opterećenja na niskom naponu u elektrodistributivnim mrežama* | *Probabilistic short-term load forecasting at low voltage in distribution networks*, PhD Thesis, Faculty of Technical Sciences, University of Novi Sad, 2023, [link](https://nardus.mpn.gov.rs/handle/123456789/21279)

[^3]: Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, *Load Pattern Recognition Method for Probabilistic Short-Term Load Forecasting at Low Voltage Level*, 2022 IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT-Europe), Novi Sad, Serbia, 2022, DOI: [10.1109/ISGT-Europe54678.2022.9960310](https://doi.org/10.1109/ISGT-Europe54678.2022.9960310)

[^4]: Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić, Darko Čapko, *Hierarchical Multiresolution Representation of Streaming Time Series*, Big Data Research 26: 100256, 2021, DOI: [10.1016/j.bdr.2021.100256](https://doi.org/10.1016/j.bdr.2021.100256)

[^5]: Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić, *Time Series Grouping Algorithm for Load Pattern Recognition*, Computers in Industry 111: 140-147, 2019, DOI: [10.1016/j.compind.2019.07.009](https://doi.org/10.1016/j.compind.2019.07.009)

[^6]: LCL dataset, https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households

[^7]: REFIT dataset, https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned

[^8]: SGSC dataset, https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data

[^9]: UMass dataset, https://traces.cs.umass.edu/index.php/Smart/Smart

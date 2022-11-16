# culearn: Cumulant Learning in Python

[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/igormanojlovic/culearn/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![build](https://github.com/igormanojlovic/culearn/actions/workflows/build.yml/badge.svg)](https://github.com/igormanojlovic/culearn/actions/workflows/build.yml)
[![cover](https://coveralls.io/repos/github/igormanojlovic/culearn/badge.svg)](https://coveralls.io/github/igormanojlovic/culearn)
[![linkedin](https://img.shields.io/badge/LinkedIn-informational?style=flat&logo=linkedin&logoColor=white&color=0D76A8)](https://www.linkedin.com/in/igormanojlovic/)

Cumulant Learning is a pattern recognition method designed to support probabilistic time series forecasting with high and stable forecast accuracy and short execution time when dealing with high-resolution time series coming from numerous data sources. The research paper describing the proposed solution is currently in the writing/publishing process. However, if you are using this software before the paper is published, please cite one of the references [^1][^2][^3]. 

## How does it work?

Cumulant Learning is designed to recognize time series patterns as follows:

1. Input time series are approximated and clustered to obtain the time series of normalized cluster-level cumulants, utilizing modified versions of existing representation methods[^1][^2][^3].
2. A set of regressors is used to capture local trends and time dependencies in the obtained cumulants based on lagged cumulant values, encoded time (calendar) features, and other exogenous features (such as weather in load forecasting).
3. Trained regressors are used to predict the future cluster-level cumulants. Then, these cumulants are transformed into time series of normalized cluster-level quantiles based on Cornish–Fisher expansion. Finally, these quantiles are denormalized and combined into prediction intervals for individual time series.

## What's inside the package?

The [culearn](https://github.com/igormanojlovic/culearn) package contains the following modules:

- [base](https://github.com/igormanojlovic/culearn/blob/main/culearn/base.py): basic classes used to support time series data wrangling throughout the package;
- [learn](https://github.com/igormanojlovic/culearn/blob/main/culearn/learn.py): implementation of the Cumulant Learning method, based on feature wrangling, clustering, and regression methods;
- [features](https://github.com/igormanojlovic/culearn/blob/main/culearn/features.py): various feature wrangling methods for time encoding, time series approximation, feature extraction, feature selection, and lag selection;
- [clustering](https://github.com/igormanojlovic/culearn/blob/main/culearn/clustering.py): k-means and hierarchical clustering methods combined with feature extraction to support high-dimensional time series clustering;
- [regression](https://github.com/igormanojlovic/culearn/blob/main/culearn/regression.py) time series regression methods based on deep learning (sequence-to-sequence recurrent neural networks, convolutional neural networks, stacked autoencoders, etc.) and "shallow" learning (support vector machines, extreme gradient boosting, etc.), combined with time encoding, feature selection, lag selection, and feature scaling;
- [data](https://github.com/igormanojlovic/culearn/blob/main/culearn/data.py): wrappers around open time series data sources;
- [csv](https://github.com/igormanojlovic/culearn/blob/main/culearn/csv.py): wrappers around CSV files with time series data;
- [util](https://github.com/igormanojlovic/culearn/blob/main/culearn/util.py): various helper functions used throughout the package;

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
source = GeneratedDataSource()
# You can also try:
# source = LCL('../data/LCL')
# source = REFIT('../data/REFIT')
# source = SGSC('../data/SGSC')
# source = UMass('../data/UMass')
ds = source.dataset()

# Step 2: Prepare the transformer
transform_encoders = TimeEncoders(MonthOfYear(), DayType(source.calendar), TimeOfDay())
transformer = CumulantTransform(encoder=transform_encoders)

# Step 3: Prepare the regressors
regressor_encoders = TimeEncoders(MonthOfYear(), DayOfWeek(), TimeOfDay(), Holiday(source.calendar))
regressor = lambda: TimeSeriesRegressor(48, t_encoder=regressor_encoders)

# Step 4: Prepare the learner
learner = CumulantLearner(ds, TimeResolution(minutes=30), transformer, regressor)

# Step 5: Train the learner
fit_interval = TimeInterval(source.interval.start, source.interval.end - timedelta(1))
learner.fit(fit_interval)

# Step 6: Test the learner
pred_intervals = learner.predict(fit_interval.end, p=[0.75, 0.95, 0.99], clusters=True, members=True)
# You can also try:
# pred_cumulants = learner.predict_cumulants(fit_interval.end)
# learner.figure(fit_interval.end, [0.75, 0.95, 0.99]).show()
```

### Step 1: Prepare the dataset

The first step is to create an instance of a data source wrapper and to retrieve a dataset for Cumulant Learning. You can implement a custom data source or use one of the existing implementations: LCL[^4], REFIT[^5], SGSC[^6], and UMass[^7]. The existing implementations will download smart meter and weather data for load forecasting from external data sources, unzip the data files, split the larger CSV files into smaller ones to support parallel processing, etc. This might take a while the first time, but makes the rest of the process much faster.

### Step 2: Prepare the transformer

The next step is to create a transformer that will be used to approximate and cluster input time series to obtain the time series of normalized cluster-level cumulants. This transformer will reduce the amount of time series data before clustering without losing significant information along the way, utilizing time series approximation at forecast time resolution, aggregation of approximated values over encoded time features, and feature extraction based on principal component analysis. All these methods can be configured to fit the specific needs of your application. 

### Step 3: Prepare the regressors

The next step is to prepare the regressors that will learn the time series patterns for each cluster obtained by the transformer. The regressors are specified in the form of a function that returns an instance of the *TimeSeriesRegressor* class. This class is used to apply generic regression models (from [keras](https://keras.io/api/), [sklearn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning), etc.) to time series data, together with time encoding (to extend inputs with time features), feature selection (to choose significant exogenous inputs), lag selection (to choose the lag for endogenous inputs), and feature scaling (to normalize the input values). All these methods can be configured to fit the specific needs of your application.

### Step 4: Prepare the learner

The final preparation step is to create the learner utilizing the transformer and regressors. In the code snippet above, the learner is configured to learn half-hour time series patterns and to predict the patterns for 48 time steps ahead (one day ahead).

### Step 5: Train the learner

Once created, the learner can be trained using the data from an arbitrary time interval (from the specified dataset). For example, in the code snippet above, one day of history is left for testing while all the other days are used for training. 

Note that the *fit* method can be called multiple times if you wish to retrain the learner. However, if you wish only to update the underlying regressors with new data, there is also the *update* method, which enables incremental learning.

### Step 6: Test the learner

Once trained, the learner can be used to obtain cluster- and/or member-level prediction intervals in the specified forecast horizon at the specified time resolution using the *predict* method. This method will yield  *TimeSeriesPrediction* objects containing the series of lower and upper bounds of the requested intervals for specified probability values. In the code snippet above, the *predict* method will yield both cluster- and member-level 75%, 95%, and 99% prediction intervals at the half-hour resolution for the next day (48 steps ahead).

Alternatively, you can use the *predict_cumulants* method to predict only the cluster-level cumulants, or the *figure* method to create a [plotly](https://plotly.com) figure that shows the cluster-level prediction intervals against the actual (but normalized) time series values. Note that the learner also contains the *evaluate* method that can be used to run initial training as well as incremental testing and updates, to obtain and evaluate the prediction results over longer time intervals (more in the [notebooks](https://github.com/igormanojlovic/culearn/tree/main/examples)).

## Background story

Different parts of the [culearn](https://github.com/igormanojlovic/culearn) package were originally implemented in [MTSR](https://github.com/igormanojlovic/MTSR) and [TimeSeriesR](https://github.com/igormanojlovic/TimeSeriesR) packages (in C#, T-SQL, and R). These two packages require many manual steps for preparing the input datasets, configuring the provided tools for data processing, and combining the tools into a single data processing pipeline that ultimately delivers the forecasting results. To minimize the number of manual steps, the complete data processing pipeline is now available in the [culearn](https://github.com/igormanojlovic/culearn) package, written entirely in Python.

## References

[^1]: Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić, Darko Čapko: *Hierarchical multiresolution representation of streaming time series*, Big Data Research 26: 100256 (2021), DOI: [10.1016/j.bdr.2021.100256](https://doi.org/10.1016/j.bdr.2021.100256)

[^2]: Igor Manojlović, Goran Švenda, Aleksandar Erdeljan, Milan Gavrić: *Time series grouping algorithm for load pattern recognition*, Computers in Industry 111: 140-147 (2019), DOI: [10.1016/j.compind.2019.07.009](https://doi.org/10.1016/j.compind.2019.07.009)

[^3]: Igor Manojlović, Goran Švenda, Aleksandar Erdeljan: *Load pattern recognition method for probabilistic short-term load forecasting at low voltage level*, 2022 IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT-Europe): 1-5 (2022), DOI: [10.1109/ISGT-Europe54678.2022.9960310](https://doi.org/10.1109/ISGT-Europe54678.2022.9960310) 

[^4]: LCL dataset, https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households

[^5]: REFIT dataset, https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned

[^6]: SGSC dataset, https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data

[^7]: UMass dataset, https://traces.cs.umass.edu/index.php/Smart/Smart

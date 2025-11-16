# Algerian Forest Fires — EDA & Modeling

## Overview

This repository contains EDA and regression modeling notebooks for the Algerian Forest Fires dataset. The primary objective is to analyze drivers of burned area and build a regression model (Ridge regression artifact is included) to predict the burned area.

## Files in this repo

- `Algerian_forest_fires_dataset.csv`
 — CSV, shape (247, 14)


- `Algerian_forest_fires_update_dataset.csv`
 — CSV, shape (243, 15)


- `EDA Notebook.ipynb`


- `Model Training.ipynb`


- `ridge.pkl`
 — artifact, ~624 bytes


- `scaler.pkl`
 — artifact, ~1215 bytes


## Dataset summary

- `Algerian_forest_fires_dataset.csv`: shape **(247, 14)**. Columns: `day, month, year, Temperature,  RH,  Ws, Rain , FFMC, DMC, DC, ISI, BUI, FWI, Classes  `.


Sample rows:

|   day |   month |   year |   Temperature |    RH |    Ws |   Rain  |   FFMC |   DMC |   DC |   ISI |   BUI |   FWI | Classes     |
|------:|--------:|-------:|--------------:|------:|------:|--------:|-------:|------:|-----:|------:|------:|------:|:------------|
|     1 |       6 |   2012 |            29 |    57 |    18 |     0   |   65.7 |   3.4 |  7.6 |   1.3 |   3.4 |   0.5 | not fire    |
|     2 |       6 |   2012 |            29 |    61 |    13 |     1.3 |   64.4 |   4.1 |  7.6 |   1   |   3.9 |   0.4 | not fire    |
|     3 |       6 |   2012 |            26 |    82 |    22 |    13.1 |   47.1 |   2.5 |  7.1 |   0.3 |   2.7 |   0.1 | not fire    |
|     4 |       6 |   2012 |            25 |    89 |    13 |     2.5 |   28.6 |   1.3 |  6.9 |   0   |   1.7 |   0   | not fire    |
|     5 |       6 |   2012 |            27 |    77 |    16 |     0   |   64.8 |   3   | 14.2 |   1.2 |   3.9 |   0.5 | not fire    |


- `Algerian_forest_fires_update_dataset.csv`: shape **(243, 15)**. Columns: `day, month, year, Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI, Classes, Region`.


Sample rows:

|   day |   month |   year |   Temperature |   RH |   Ws |   Rain |   FFMC |   DMC |   DC |   ISI |   BUI |   FWI | Classes   |   Region |
|------:|--------:|-------:|--------------:|-----:|-----:|-------:|-------:|------:|-----:|------:|------:|------:|:----------|---------:|
|     1 |       6 |   2012 |            29 |   57 |   18 |    0   |   65.7 |   3.4 |  7.6 |   1.3 |   3.4 |   0.5 | not fire  |        0 |
|     2 |       6 |   2012 |            29 |   61 |   13 |    1.3 |   64.4 |   4.1 |  7.6 |   1   |   3.9 |   0.4 | not fire  |        0 |
|     3 |       6 |   2012 |            26 |   82 |   22 |   13.1 |   47.1 |   2.5 |  7.1 |   0.3 |   2.7 |   0.1 | not fire  |        0 |
|     4 |       6 |   2012 |            25 |   89 |   13 |    2.5 |   28.6 |   1.3 |  6.9 |   0   |   1.7 |   0   | not fire  |        0 |
|     5 |       6 |   2012 |            27 |   77 |   16 |    0   |   64.8 |   3   | 14.2 |   1.2 |   3.9 |   0.5 | not fire  |        0 |


## Notebooks

- `EDA Notebook.ipynb` — headings extracted:

  - ## About Dataset

  - ## Attribute Informations

  - ## Data Cleaning

  - ## Exploratory Data Analysis(EDA)



- `Model Training.ipynb` — headings extracted:

  - ## Drop day, month and year column

  - ## Feature Scaling or Standarization

  - ## Box PLot to understand the effect of standard scaling

  - ## Linear Regression

  - ## Lasso

  - ## Cross Validation Lasso

  - ## Ridge Regression model

  - ## Elasticnet Regression

  - ## Pickle the machine learning models, preprocessing model standardscaler



## Artifacts (models & scalers)

- `ridge.pkl` — size: 624 bytes

  - successfully loaded with: joblib

  - pickle load error: invalid load key, '\x07'.

  - type: `Ridge`

  - estimator class: `Ridge`

  - `n_features_in_`: 9

- `scaler.pkl` — size: 1215 bytes

  - successfully loaded with: pickle

  - type: `ndarray`

## Detected imports and suggested dependencies

Imports detected in notebooks:

```text

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import ElasticNetCV

from sklearn.linear_model import Lasso

from sklearn.linear_model import LassoCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import RidgeCV

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import joblib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

```

Suggested `requirements.txt`:

```
python>=3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
jupyter
```

## Preprocessing (found code snippets)

- `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)`

- `data = data.dropna().reset_index(drop=True)`

- `data.dropna(subset=['day'], inplace=True)`

- `data[['Region']] = data[['Region']].astype(int)`

- `data[['month', 'year', 'day', 'Temperature', 'RH', 'Ws']] = data[['month', 'year', 'day', 'Temperature', 'RH', 'Ws']].astype(int)`

- `data[i] = data[i].astype(float)`

- `from sklearn.model_selection import train_test_split`

- `from sklearn.preprocessing import StandardScaler`

- `standard = StandardScaler()`

## Model training (found snippets)

- `e.fit(X_train_scaled, y_train)`

- `el.fit(X_train_scaled, y_train)`

- `la.fit(X_train_scaled, y_train)`

- `lasso.fit(X_train_scaled, y_train)`

- `linear = LinearRegression()`

- `linear.fit(X_train_scaled, y_train)`

- `r = Ridge()`

- `r.fit(X_train_scaled, y_train)`

- `rc = RidgeCV(cv=5)`

- `rc.fit(X_train_scaled, y_train)`

- `score = r2_score(y_pred, y_test)`

- `score = r2_score(y_test, y_pred)`

- `y_pred = e.predict(X_test_scaled)`

- `y_pred = el.predict(X_test_scaled)`

- `y_pred = la.predict(X_test_scaled)`

- `y_pred = lasso.predict(X_test_scaled)`

- `y_pred = linear.predict(X_test_scaled)`

- `y_pred = r.predict(X_test_scaled)`

- `y_pred = rc.predict(X_test_scaled)`

Found metric calculations (search inside notebooks for these lines):

- `from sklearn.metrics import r2_score`

- `from sklearn.metrics import mean_absolute_error`

- `mae = mean_absolute_error(y_test, y_pred)`

- `score = r2_score(y_test, y_pred)`

- `from sklearn.metrics import r2_score`

- `from sklearn.metrics import mean_absolute_error`

- `mae = mean_absolute_error(y_test, y_pred)`

- `score = r2_score(y_test, y_pred)`

- `mae = mean_absolute_error(y_test, y_pred)`

- `score = r2_score(y_test, y_pred)`

- `from sklearn.metrics import mean_absolute_error`

- `from sklearn.metrics import r2_score`

- `mae = mean_absolute_error(y_pred, y_test)`

- `score = r2_score(y_pred, y_test)`

- `from sklearn.metrics import mean_absolute_error`

- `from sklearn.metrics import r2_score`

- `mae = mean_absolute_error(y_pred, y_test)`

- `score = r2_score(y_pred, y_test)`

- `mae = mean_absolute_error(y_pred, y_test)`

- `score = r2_score(y_pred, y_test)`

## How to reproduce / run

1. Create a virtual environment and install requirements.

2. Place CSV files in repository root or `data/` folder.

3. Run `EDA Notebook.ipynb` to explore and preprocess data.

4. Run `Model Training.ipynb` to train and evaluate models. This notebook should save `ridge.pkl` and `scaler.pkl`.

5. Use the example script below to load artifacts and predict on new data.

### Example: load artifacts and predict

```python
import pickle
import pandas as pd
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)
with open('ridge.pkl','rb') as f:
    model = pickle.load(f)
# X_new must have the same columns/order used during training
# X_scaled = scaler.transform(X_new)
# y_pred = model.predict(X_scaled)
```

## Notes & next steps

- Confirm the exact feature order expected by the scaler/model. If `scaler` has `feature_names_in_`, use it to reorder input columns.

- Add `requirements.txt` with exact versions used for training to ensure reproducibility.

- Add `predict.py` and/or a small API to demonstrate real-time predictions.

- Include final evaluation metrics and sample visualizations in the repository root README.

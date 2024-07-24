### Stock Market Prediction Program Documentation

This documentation provides an overview and step-by-step explanation of the stock market prediction program contained within the provided Jupyter Notebook.

#### Table of Contents

1. [Introduction](#Introduction)
2. [Data Preparation](#Data-Preparation)
3. [Feature Engineering](#Feature-Engineering)
4. [Model Training](#Model-Training)
5. [Model Evaluation](#Model-Evaluation)
6. [Predictions](#Predictions)
7. [Conclusion](#Conclusion)

---

### Introduction

This notebook demonstrates a stock market prediction model using historical stock price data. The model aims to predict the direction of stock price movement (whether the closing price will be higher than the opening price).

---

### Data Preparation

#### Importing Libraries

The necessary libraries for data manipulation, visualization, and machine learning are imported at the beginning of the notebook:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
```

#### Loading the Data
Dataset - https://drive.google.com/file/d/15FYJXeIVCedBhpqafJHN3a3euuxcLL4V/view?usp=drive_link

The historical stock price data is loaded into a Pandas DataFrame. The data includes columns such as `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.

```python
data = pd.read_csv('path_to_your_data.csv')
```

---

### Feature Engineering

Feature engineering involves creating new features that can help improve the model's performance. In this notebook, several new features are created based on the existing stock price data:

1. **Percentage Change**: The percentage change between the opening and closing prices.
2. **High-Low Percentage**: The percentage difference between the high and low prices.
3. **Daily Change**: The difference between the closing price and the opening price.

```python
data['Price_Change'] = ((data['Close'] - data['Open']) / data['Open']) * 100
data['High_Low_Percentage'] = ((data['High'] - data['Low']) / data['Low']) * 100
data['Daily_Change'] = data['Close'] - data['Open']
```

#### Creating the Target Variable

The target variable for the model is a binary indicator of whether the closing price is higher than the opening price:

```python
data['Target'] = data['Close'] > data['Open']
```

---

### Model Training

The model used in this notebook is a RandomForestClassifier. The training data is split into training and testing sets, and the RandomForestClassifier is trained on the training set.

#### Splitting the Data

```python
train = data[data['Date'] < '2022-01-01']
test = data[data['Date'] >= '2022-01-01']
```

#### Defining the Predictor Variables

The predictor variables include all the features created during feature engineering:

```python
predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'High_Low_Percentage', 'Daily_Change']
```

#### Training the Model

```python
model = RandomForestClassifier(n_estimators=200, min_samples_split=100, random_state=1)
model.fit(train[predictors], train['Target'])
```

---

### Model Evaluation

The model's performance is evaluated using precision score, which is the ratio of correctly predicted positive observations to the total predicted positives.

#### Making Predictions

```python
predictions = model.predict(test[predictors])
```

#### Calculating Precision Score

```python
precision = precision_score(test['Target'], predictions)
print(f'Precision Score: {precision}')
```

---

### Predictions

The model is used to make predictions on the test set. The predictions indicate whether the closing price will be higher than the opening price for each day in the test set.

```python
pred = model.predict(test[predictors])
pred
```

---

### Conclusion

This notebook demonstrates the process of building a stock market prediction model using a RandomForestClassifier. The model leverages historical stock price data to predict the direction of stock price movement. By following the steps outlined in this documentation, you can replicate the analysis and adapt the model to other datasets or improve it further.


# Linear Regression Implementation from Scratch

This repository contains a custom Python implementation of Simple Linear Regression. Unlike standard machine learning workflows that rely on pre-built libraries like Scikit-Learn for the model logic, this project builds the regression algorithm from the ground up using fundamental mathematical formulas.

## 📄 Project Overview

The goal of this project is to understand the mathematics behind Linear Regression by implementing the **Ordinary Least Squares (OLS)** method manually.

The project demonstrates:
1.  Building a custom class `MyLR` that mimics the behavior of Scikit-Learn's estimator.
2.  Calculating the slope ($m$) and intercept ($b$) using closed-form mathematical equations.
3.  Training the custom model on a student placement dataset.
4.  Predicting salary packages based on CGPA.

## 📊 Dataset

The model uses a dataset named `placementML.csv` with the following features:
* **Independent Variable ($X$):** `cgpa` (Cumulative Grade Point Average)
* **Dependent Variable ($y$):** `package` (Salary Package)

## 🧮 Mathematical Approach

The model determines the best-fit line $y = mx + b$ by minimizing the sum of squared errors. The coefficients are calculated using the following formulas:

### Slope ($m$)
$$m = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

### Intercept ($b$)
$$b = \bar{y} - m\bar{x}$$

Where:
* $\bar{x}$ and $\bar{y}$ are the means of the input feature and target variable, respectively.

## 🛠️ Implementation Details

The core logic is encapsulated in the `MyLR` class:

```python
class MyLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        num = 0
        den = 0
        
        # Iterating through training data to calculate numerator and denominator
        for i in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))
            
        self.m = num / den
        self.b = y_train.mean() - (self.m * X_train.mean())

    def predict(self, X_test):
        return self.m * X_test + self.b

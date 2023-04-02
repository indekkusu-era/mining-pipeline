from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

def include(*col):
    def f(df: pd.DataFrame):
        return df[list(col)]
    return f

def exclude(*col):
    def f(df: pd.DataFrame):
        return df[df.columns[~df.columns.isin(col)]]
    return f

class Model(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        ...
    
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        ...
    
    def fit_predict(self, data: pd.DataFrame):
        self.fit(data)
        return self.predict(data)
    
    def __rshift__(self, x):
        if isinstance(x, Model):
            return Pipeline([self, x])
        return self.fit(x)
    
class Pipeline(Model):
    def __init__(self, models):
        self._models = models
    
    def fit(self, data: pd.DataFrame):
        x = data.copy()
        for i in range(len(self._models)):
            x = self._models[i].fit_predict(x)
        return self
    
    def predict(self, data: pd.DataFrame):
        x = data.copy()
        for i in range(len(self._models)):
            x = self._models[i].predict(x)
        return x

    def __rshift__(self, x):
        if isinstance(x, Model):
            self._models.append(model)
        if isinstance(x, pd.DataFrame):
            self.fit(x)
        return self

class LinearRegression(Model):
    def __init__(self, X_col, y_col):
        self.X_col = X_col
        self.y_col = y_col
    
    def pad_ones(self, x):
        return np.hstack((x, np.ones((len(x), 1))))

    def fit(self, data: pd.DataFrame):
        self._data = data.copy()
        X = self.X_col(data).values
        X = self.pad_ones(X)
        y = self.y_col(data).values
        b = np.linalg.inv(X.T @ X) @ X.T @ y
        b = b.flatten()
        self.coef_ = b[:-1]
        self.intercept_ = b[-1]
        return self
    
    def predict(self, X: pd.DataFrame):
        x_predict = self.X_col(X)
        return x_predict @ self.coef_ + self.intercept_

class Standardize(Model):
    def __init__(self, X_col):
        self.X_col = X_col
    
    def fit(self, data: pd.DataFrame):
        self._data = data.copy()
        X = self.X_col(data).values
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0, ddof=1)
        return self
    
    def predict(self, data: pd.DataFrame):
        selected_cols_data = self.X_col(data)
        selected_cols = selected_cols_data.columns
        standardized_values = (selected_cols_data.values - self._mean) / self._std
        new_data = data
        new_data[selected_cols] = pd.DataFrame(standardized_values, columns=selected_cols)
        return new_data

if __name__ == "__main__":
    dataset = pd.read_csv('multiple_linear_regression_dataset.csv')
    model = (Standardize(exclude('income')) >> LinearRegression(exclude('income'), include('income'))) >> dataset
    print(model.predict(dataset))

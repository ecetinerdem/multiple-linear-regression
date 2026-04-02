import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from housing_analysis.data_processing import ModelData
from housing_analysis.exceptions import ModelOperation
from housing_analysis.logging_utils import logger


@dataclass
class ModelResults:
    model: LinearRegression
    scaler: StandardScaler
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float


def train_model(data: ModelData) -> Tuple[LinearRegression, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.X_train)

    # Train the model
    model = LinearRegression()
    model.fit(X_scaled, data.y_train)

    return model, scaler



def evaluate_model(data: ModelData, model: LinearRegression, scaler: StandardScaler) -> ModelResults:
    # Evaluate our training data
    X_train_scaled = scaler.transform(data.X_train)
    train_predictions = model.predict(X_train_scaled)
    train_r2 = r2_score(data.y_train, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(data.y_train, train_predictions))

    #Evaluate our test data
    X_test_scaled = scaler.transform(data.X_test)
    test_predictions = model.predict(X_test_scaled)
    test_r2 = r2_score(data.y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(data.y_test, test_predictions))


    logger.info(f"Model evaluated. R-squared (train): {train_r2:.4f}, R-squared (test): {test_r2:.4f}")
    return ModelResults(
        model=model,
        scaler=scaler,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        train_r2=train_r2,
        test_r2=test_r2,
        train_rmse=train_rmse,
        test_rmse=test_rmse
    )
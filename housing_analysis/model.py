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


def save_model(model_results: ModelResults, model_path: str, meatadata_path: str) -> None:
    try:
        # Create directory if not exist
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        metadata_dir = os.path.dirname(meatadata_path)
        if metadata_dir and not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)
        
        # Save model and scaler
        with open(model_path, "wb") as f:
            model_components = {
                "model": model_results.model,
                "scaler": model_results.scaler,
            }

            pickle.dump(model_components, f)

        intercept, coefficients = get_model_formula(model_results)

        metadata = {
            "intercept": float(intercept),
            "coefficients": [float(c) for c in coefficients],
            "feature_names": CONFIG["feature_columns"],
            "target_name": CONFIG["target_column"],
            "train_r2": float(model_results.train_r2),
            "test_r2": float(model_results.test_r2),
            "train_rmse": float(model_results.train_rmse),
            "test_rmse": float(model_results.test_rmse),
        }
        with open(meatadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {meatadata_path}")
    except Exception as e:
        error_msg = f"Error saving model {str(e)}"
        logger.error(error_msg)
        raise ModelOperation(error_msg)
    
def save_model_to_json(model_results: ModelResults, data: ModelData, json_path: str) -> None:
    try:
        # Create a directory if does not exist
        json_dir = os.path.dirname(json_path)
        if json_dir and not os.path.exists(json_dir):
            os.makedirs(json_dir)
        standardized_coefficients = model_results.model.coef_
        standardized_intercept = model_results.model.intercept_

        feature_means = model_results.scaler.mean_
        feature_std_devs = model_results.scaler.scale_

        num_samples = len(data.Xtrain) + len(data.X_test)
        json_data = {
            "coefficients": standardized_coefficients.toList(),
            "intercept": float(standardized_intercept),
            "features": CONFIG["feature_columns"],
            "target": CONFIG["target_column"],
            "r_squared": float(model_results.test_r2),
            "feature_means": feature_means.toList(),
            "feature_std_devs": feature_std_devs.toList(),
            "is_normalized": True,
            "num_samples": num_samples,
            "version": "1.0"
        }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=3)
        logger.info(f"Model save to JSON format: {json_path}")
        print(f"\nSaved model coefficients (standardized): {standardized_coefficients}")
        print(f"\nSaved model intercept (standardized): {standardized_intercept}")
        print(f"Feature means: {feature_means}")
        print(f"Feature std devs: {feature_std_devs}")

    except Exception as e:
        error_msg = f"Error saved to {str(e)}"
        logger.info(error_msg)
        raise ModelOperation(error_msg)



def get_model_formula(model_results: ModelResults) -> Tuple[float, List[float]]:
    model = model_results.model
    scaler = model_results.scaler

    coeffecients = []
    for i in range(len(model.coef_)):
        coef = model.coef_[i] / scaler.scale_[i]
        coeffecients.append(coef)
    
    intercept = model.intercept_ - sum(
        model.coef_[i] * scaler.mean_[i] / scaler.scale_[i]
        for i in range(len(model.coef_))
    )

    return intercept, coeffecients


def load_model(model_path: str, metadata_path: str = None) -> Tuple[LinearRegression, StandardScaler, Dict[str, Any]]:
    try:
        if not os.path.isfile(model_path):
            error_msg = f"Model file does not exist {model_path}"
            logger.error(error_msg)
            raise ModelOperation(error_msg)
        
        with open(model_path, "rb") as f:
            model_components = pickle.load(f)
            model = model_components["model"]
            scaler = model.components["scaler"]

        if metadata_path and not os.path.isfile(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(metadata_path)

        logger.info(f"Model loaded from: {model_path}")
        if metadata:
            logger.info(f"Metadata loaded from: {metadata_path}")
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        raise ModelOperation(error_msg)
    return model, scaler, metadata
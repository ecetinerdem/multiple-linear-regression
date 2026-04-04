import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import CONFIG
from housing_analysis.logging_utils import logger
from housing_analysis.exceptions import DataProcessingError

@dataclass
class ModelData:
    X_train: np.ndarray
    X_test : np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray



def load_data(file_path: str) ->pd.DataFrame:
    # Check if file exist
    if not os.path.isfile(file_path):
        error_msg = f"File does not exist: {file_path}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg)
    
    try:
        # Load data from file path
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        # Validate data
        missing_columns = set(CONFIG["required_columns"]).difference(df.columns)
        if missing_columns:
            error_msg = f"Error loading data: {", ".join(missing_columns)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg)

        return df
    except Exception as e:
        error_msg = f"Error loading data {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg)

    



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing data")

    processed_df = df.copy()

    # Convert required columns to numeric in one step

    for col in CONFIG["required_columns"]:
        processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

    # Handle missing values after conversion to numeric
    if processed_df[CONFIG["required_columns"]].isna().any().any():
        logger.warning("Missing values found, dropping rows with missing values")
        processed_df = processed_df.dropna(subset=CONFIG["required_columns"])

    # Handle outliers
    for col in CONFIG["required_columns"]:
        # Calculate mean
        mean = processed_df[col].mean()

        # Calculate std
        std = processed_df[col].std()

        # Define upper and lower bounds
        treshold = CONFIG["outlier_treshold"]
        lower_bound = mean - treshold * std
        upper_bound = mean + treshold * std

        outliers = (processed_df[col] < lower_bound) | (processed_df[col] >upper_bound)
        if outliers.any():
            logger.warning(f"Removing {outliers.sum()} outliers from {col}")
            processed_df = processed_df[~outliers]

    return processed_df


def prepare_model_data(df: pd.DataFrame) -> ModelData:
    # Get X and y
    X = df[CONFIG["feature_columns"]].values
    y = df[CONFIG["target_column"]].values

    # Split data into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    return ModelData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

def make_predictions(df: pd.DataFrame, model, scaler) -> np.ndarray:
   try:
        X = df[CONFIG["feature_columns"]].values
        X_scaled = scaler.transform(X)
        
        predictions = model.predict(X_scaled)

        return predictions
   except Exception as e:
        error_msg = f"Error making predictions {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import warnings

from config import CONFIG
from housing_analysis.logging_utils import logger
from housing_analysis.data_processing import ModelData
from housing_analysis.model import ModelResults, get_model_formula


def print_results(data: ModelData, model_results: ModelResults) -> None:
    # Get the model formula
    intercept, coefficients = get_model_formula(model_results)
    print("\nMultiple Linear Regression Formula")
    print(f"Price = {intercept:.4f} + {coefficients[0]:.4f} x Square Footage + {coefficients[1]:.4f} x Bedrooms")
    # R-squared explains variance in data 0-1 1 perfect
    print(f"R-squared (train): {model_results.train_r2:.4f}")
    print(f"R-squared (test): {model_results.test_r2:.4f}")

    # RMSE average predictions error lower better 0-1 0 perfect
    print(f"RMSE (train): {model_results.train_rmse:.4f}")
    print(f"RMSE (test): {model_results.test_rmse:.4f}")
    
    # Create some dataframes
    train_df = pd.DataFrame({
        "Square Footage": data.X_train[:, 0],
        "Bedrooms": data.X_train[:, 1],
        "Actual Price($K)": data.Y_train,
        "Predicted Price ($K)": np.round(model_results.train_predictions, 2)
    })

    
    test_df = pd.DataFrame({
        "Square Footage": data.X_test[:, 0],
        "Bedrooms": data.X_test[:, 1],
        "Actual Price($K)": data.Y_test,
        "Predicted Price ($K)": np.round(model_results.test_predictions, 2)
    })

    # Print sample of results
    print("\nTraining Data Sample (first five results)")
    print(train.df.head().to_string(index=false))

    print("\nTest Data Sample (first five results)")
    print(test.df.head().to_string(index=false))
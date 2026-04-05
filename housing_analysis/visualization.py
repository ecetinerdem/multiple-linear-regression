import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import warnings

from config import CONFIG
from housing_analysis.logging_utils import logger
from housing_analysis.data_processing import ModelData
from housing_analysis.model import ModelResults, get_model_formula

def create_visualization(
        data: ModelData,
        model_results: ModelResults
) -> Dict[str, Any]:
    # Combine training and test data to get the full range of values
    X_combined = np.vstack((data.X_train, data.X_test))

    # Get the feature range for plotting
    x_min, x_max = X_combined[:, 0].min(), X_combined[:, 0].max() # Square footage
    y_min, y_max = X_combined[:, 1].min(), X_combined[:, 1].max() # Bedroom

    feature_ranges = [
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    ]

    # Calculate mean of features for regression line/plane
    feature_means = X_combined.mean(axis=1)

    # Get formula for display
    intercept, coefficients = get_model_formula(model_results)
    formula_text = f"Price = {intercept:.4f} + {coefficients[0]:.4f} x Square Footage + {coefficients[1]:.4f} x Bedrooms"

    # Create mesh grid for 3d visualization
    x_range = np.linspace(x_min, x_max, CONFIG["mesh_grid_size"])
    y_range = np.linspace(y_min, y_max, CONFIG["mesh_grid_size"])
    xx, yy = np.meshgrid(x_range, y_range)

    # Prepare grid points for predictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Scale grid points using the same scalar for training
    grid_points_scaled = model_results.scaler.transform(grid_points)

    # Make predictions
    z_pred = model_results.model.predict(grid_points_scaled)

    # Reshape the predictions
    zz = z_pred.reshape(xx.shape)

    return {
        "feature_ranges": feature_ranges,
        "feature_means": feature_means,
        "formula_text": formula_text,
        "xx": xx,
        "yy": yy,
        "zz": zz
    }

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
    print(train_df.head().to_string(index=False))

    print("\nTest Data Sample (first five results)")
    print(test_df.head().to_string(index=False))


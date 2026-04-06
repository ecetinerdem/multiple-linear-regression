import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import warnings

from config import CONFIG
from housing_analysis.logging_utils import logger
from housing_analysis.data_processing import ModelData
from housing_analysis.model import ModelResults, get_model_formula

def create_3d_visualization(
        data: ModelData,
        model_results: ModelResults,
        viz_data: Dict[str, Any],
        output_file: str,
        show_plot: bool = True
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Create a figure and add a 3d subplot
        fig = plt.figure(figsize=CONFIG["figure_size"])
        ax = fig.add_subplot(111, projection="3d")

        # Set initial view angle
        ax.view_init(elev=30, azim=45)

        # Plot training data points
        ax.scatter(
            data.X_train[:, 0],
            data.X_train[:, 1],
            data.y_train,
            color= CONFIG["point_color"],
            alpha=CONFIG["point_alpha"],
            label="Training data"
        )

        # Plot test data points
        ax.scatter(
            data.X_test[:, 0],
            data.X_test[:, 1],
            data.y_test,
            color= CONFIG["point_color"],
            alpha=CONFIG["point_alpha"],
            label="Test data"
        )

        # Plot the regression plain
        surf = ax.plot_surface(
            viz_data["xx"],
            viz_data["yy"],
            viz_data["zz"],
            alpha=CONFIG["plain_alpha"],
            color=CONFIG["plain_color"],
            rstride=2,
            cstride=2
        )

        # Add labels and title
        ax.set_xlabel("Square Footage")
        ax.set_ylabel("Bedrooms")
        ax.set_zlabel("Price ($thousands $)")
        ax.set_title("Multiple Linear Regression: 3D visualization with regression plain")

        # Add formula text
        plt.figtext(0.1, 0.01, viz_data["formula_text"], fontsize=12)

        # Save the figure to a file
        plt.savefig(output_file, bbox_inches="tight")
        logger.info(f"3d plot save as {output_file}")

        if show_plot:
            plt.show()
        
        plt.close()


def create_2d_visualization(
        data: ModelData,
        model_results: ModelResults,
        viz_data: Dict[str, Any],
        output_file: str,
        show_plot: bool = True
) -> None:
    # Create a figure with two side-by-side plots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=CONFIG["figure_size"])
    feature_names = CONFIG["feature_columns"]
    feature_ranges = viz_data["feature_ranges"]
    feature_means = viz_data["feature_means"]

    # Createa a plot for each feature

    for i, feature in enumerate(feature_names):
        ax = axes[i]

        # Extract feature values for this specific feature
        X_train_feature = data.X_train[:, i]
        X_test_feature = data.X_test[:, i]

        # Plot training data point
        ax.scatter(
            X_train_feature, # X coords
            data.y_train, # y coords
            color=CONFIG["point_color"],
            alpha=CONFIG["point_alpha"],
            label="Training data"
        )

        
        # Plot test data point
        ax.scatter(
            X_test_feature, # X coords
            data.y_test, # y coords
            color=CONFIG["test_point_color"],
            alpha=CONFIG["point_alpha"],
            label="Test data"
        )

        # Add regression line
        if i == 0:
            # Square footage line
            line_X = np.c_[feature_ranges[0], np.full(
                feature_ranges[0].shape,
                feature_means[1]
            )]
        else:
            # Bedrroms line
            line_X = np.c_[np.full(feature_ranges[1].shape, feature_means[0]), feature_ranges[1]]

        # Scale the line points and predict prices
        line_X_scaled = model_results.scaler.transform(line_X)
        line_y = model_results.model.predict(line_X_scaled)

        # Plot the regression line
        ax.plot(
            feature_ranges[i],
            line_y,
            color=CONFIG["line_color"],
            linewidth=CONFIG["line_width"],
            label="Regression line"
        )

        # Add labels and title
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel("Price (thousands $)")
        ax.set_title(
           f"Price vs {feature.replace('_', ' ').title()} with Regression Line"
        )
        ax.legend()
        ax.grid(True, alpha=CONFIG["grid_alpha"])

    # Add overall title
    plt.suptitle("Multiple Linear Regression: Housing Price vs Features")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_file)
    logger.info(f"2d plot saved as {output_file}")

    if show_plot:
        plt.show()
    plt.close()

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

    # Calculate mean of features for regression line/plain
    feature_means = X_combined.mean(axis=0)

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
        "Actual Price($K)": data.y_train,
        "Predicted Price ($K)": np.round(model_results.train_predictions, 2)
    })

    
    test_df = pd.DataFrame({
        "Square Footage": data.X_test[:, 0],
        "Bedrooms": data.X_test[:, 1],
        "Actual Price($K)": data.y_test,
        "Predicted Price ($K)": np.round(model_results.test_predictions, 2)
    })

    # Print sample of results
    print("\nTraining Data Sample (first five results)")
    print(train_df.head().to_string(index=False))

    print("\nTest Data Sample (first five results)")
    print(test_df.head().to_string(index=False))


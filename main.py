import argparse
import sys
from config import CONFIG
from housing_analysis import (
    DataProcessingError,
    ModelOperation,
    evaluate_model,
    load_data,
    load_model,
    make_predictions,
    prepare_model_data,
    preprocess_data,
    save_model,
    train_model
    )

from housing_analysis.logging_utils import logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiple Linear Regression Analysis on housing data")
    
    parser.add_argument(
        "-f","--file",
        type=str,
        default=CONFIG["default_csv"],
        help=f"Path to csv file (default: {CONFIG["default_csv"]})"
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display the plot (still save to file)",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the train model to file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=CONFIG["default_model_path"],
        help=f"Path to load/save model (default: {CONFIG["default_model_path"]})"
    )

    parser.add_argument(
        "--load-model",
        action="store_true",
        help= "Load a previously trained model instead of training new one"
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=CONFIG["default_metadata_path"],
        help=f"Path to load/save model (default: {CONFIG["default_metadata_path"]})"
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only make predictions using a loaded model, no training or evaluation"
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Unknown arguments are ignored: {unknown}")
    
    if args.predict_only and not args.load_model:
        parser.error("--predivt-only requires --load-model")
    return args

def main() -> int:
    try:
        # Parse command line arguments
        args = parse_arguments()
        # Choose operation mode based on arguments
        if args.load_model:
            # Load model and metadata from files
            model, scaler, metadata = load_model(args.model_path, args.metadata_path)

            if args.predict_only:
                df = load_data(args.file)
                processed_df = preprocess_data(df)
                predictions = make_predictions(processed_df, model, scaler)
                print(f"Length of predictions {len(predictions)}")
            
            else:
                df = load_data(args.file)
                processed_df = preprocess_data(df)
                model_data = prepare_model_data(processed_df)
                model_results = evaluate_model(model_data, model, scaler)
                print(f"Test R2: {model_results.test_r2: .4f}")
        else:
            df = load_data(args.file)
            processed_df = preprocess_data(df)
            model_data = prepare_model_data(processed_df)
            model, scaler = train_model(model_data)

            model_results = evaluate_model(model_data, model, scaler)
            print(f"Test R2: {model_results.test_r2: .4f}")

            if args.save_model:
                save_model(model_results, args.model_path, args.metadata_path)
    except DataProcessingError as e:
        logger.error(f"Data Processing error: {str(e)}")
        return 1
    except ModelOperation as e:
        logger.error(f"Model Operation error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected  error: {str(e)}")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())
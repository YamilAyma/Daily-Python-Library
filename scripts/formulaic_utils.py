#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
formulaic_utils.py: Utilities for the Formulaic library.

This script provides helper functions to streamline the process of creating
design matrices from pandas DataFrames using Wilkinson formulas, ready for
statistical modeling.
"""
from __future__ import annotations

import sys
import os
import argparse
from typing import Tuple, TYPE_CHECKING

# This block is only processed by type checkers, not by the Python interpreter.
if TYPE_CHECKING:
    import pandas as pd
    import formulaic
    from sklearn.linear_model import LinearRegression

# Allow running as a script or as a module
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .scripttypes import ScriptMetadata, function_metadata, FUNCTIONS_METADATA
FUNCTIONS_METADATA.clear()

# --- Script Metadata ---
metadata = ScriptMetadata(
    title="Formulaic Utilities",
    description="Utilities to simplify data preprocessing using Wilkinson formulas with Formulaic, creating design matrices for modeling.",
    version="0.2.0", # Version bump for lazy loading refactor
    author="AI",
    email="Without email",
    license="MIT",
    status="development",
    dependencies=["formulaic", "pandas", "scikit-learn"],
    tags=["formula-parser", "data-preprocessing", "statistics", "modeling", "scikit-learn"],
    cli=True,
    links=[
        {"Formulaic GitHub Repository": "https://github.com/matthewwardrop/formulaic"}
    ],
    note="SETUP: pip install formulaic pandas scikit-learn"
)

##########################
#    SCRIPT FUNCTIONS
##########################

@function_metadata(status="development", note="Transforms a single DataFrame into a model matrix based on a formula.", category="matrix_creation", tags=["formula", "dataframe", "design-matrix"])
def create_model_matrix(data: pd.DataFrame, formula: str) -> Tuple[pd.Series | None, pd.DataFrame]:
    """
    Creates a model matrix (and outcome vector, if specified) from a DataFrame.

    Args:
        data (pd.DataFrame): The input data.
        formula (str): The Wilkinson formula (e.g., 'y ~ x1 + C(x2)').

    Returns:
        A tuple containing the outcome vector (y) and the design matrix (X).
        If no outcome is in the formula, y will be None.
    """
    try:
        import formulaic
    except ImportError as e:
        raise ImportError("Formulaic is not installed. Please run: 'pip install formulaic'") from e

    print(f"Creating model matrix with formula: '{formula}'")
    y, X = formulaic.model_matrix(formula, data)
    print("Model matrix created successfully.")
    return y, X


@function_metadata(status="development", note="Ensures consistent data encoding between training and testing sets.", category="data_transformation", tags=["train-test", "encoding", "consistency", "pipeline"])
def transform_train_test_split(
    train_data: pd.DataFrame, test_data: pd.DataFrame, formula: str
) -> Tuple[formulaic.ModelMatrix, formulaic.ModelMatrix]:
    """
    Applies a formula to a train/test split, ensuring consistent encoding.

    Args:
        train_data (pd.DataFrame): The training dataset.
        test_data (pd.DataFrame): The testing dataset.
        formula (str): The Wilkinson formula to apply.

    Returns:
        A tuple containing the transformed training and testing matrices.
    """
    try:
        import formulaic
    except ImportError as e:
        raise ImportError("Formulaic is not installed. Please run: 'pip install formulaic'") from e

    print("Creating a materializer from the training data...")
    materializer = formulaic.Formula(formula).get_materializer()
    
    print("Transforming training data...")
    train_matrix = materializer.get_model_matrix(train_data)
    
    print("Transforming testing data using the same encoding...")
    test_matrix = materializer.get_model_matrix(test_data)
    
    print("Train and test sets transformed consistently.")
    return train_matrix, test_matrix


@function_metadata(status="development", note="An end-to-end example of creating a design matrix and training a scikit-learn model.", category="modeling", tags=["scikit-learn", "linear-regression", "end-to-end"])
def build_and_train_model(data: pd.DataFrame, formula: str) -> LinearRegression:
    """
    Creates a design matrix from a formula and trains a simple Linear Regression model.

    Args:
        data (pd.DataFrame): The dataset to use for training.
        formula (str): The Wilkinson formula specifying the model.

    Returns:
        A trained scikit-learn LinearRegression model object.
    """
    try:
        import formulaic
        from sklearn.linear_model import LinearRegression
    except ImportError as e:
        raise ImportError("Required libraries are not installed. Please run: 'pip install formulaic scikit-learn'") from e

    print("Preparing data for modeling...")
    y, X = formulaic.model_matrix(formula, data)
    
    if y is None:
        raise ValueError("The formula must include an outcome variable (e.g., 'y ~ x') for model training.")

    print("Training scikit-learn LinearRegression model...")
    model = LinearRegression()
    model.fit(X, y)
    
    r_squared = model.score(X, y)
    print(f"Model training complete. R-squared: {r_squared:.4f}")
    
    return model

##########################
#    CLI FUNCTIONS
##########################

def _load_dataframe_from_csv(filepath: str) -> pd.DataFrame:
    """Helper to load a DataFrame from a CSV file."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("Pandas is not installed. Please run: 'pip install pandas'") from e
        
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded DataFrame from '{filepath}'.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)

def _handle_cli():
    """Manages the Command-Line Interface for this script."""
    parser = argparse.ArgumentParser(description=metadata.description)
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)
    
    # --- Argument definitions remain the same ---
    p_create = subparsers.add_parser("create-matrix", help="Transform a single CSV into a design matrix.")
    p_create.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    p_create.add_argument("--formula", type=str, required=True, help="Wilkinson formula (e.g., 'y ~ x1 + C(x2)').")
    p_create.add_argument("--output-x", type=str, required=True, help="Path to save the output design matrix (X) CSV.")
    p_create.add_argument("--output-y", type=str, help="Path to save the output outcome vector (y) CSV (if any).")
    p_traintest = subparsers.add_parser("train-test-transform", help="Consistently transform a train/test CSV split.")
    p_traintest.add_argument("train_csv", type=str, help="Path to the training data CSV.")
    p_traintest.add_argument("test_csv", type=str, help="Path to the testing data CSV.")
    p_traintest.add_argument("--formula", type=str, required=True, help="Wilkinson formula.")
    p_traintest.add_argument("--output-train-x", type=str, required=True, help="Path to save the transformed training matrix (X).")
    p_traintest.add_argument("--output-test-x", type=str, required=True, help="Path to save the transformed testing matrix (X).")
    p_train = subparsers.add_parser("train-model", help="Create a matrix and train a simple linear model.")
    p_train.add_argument("input_csv", type=str, help="Path to the input CSV file for training.")
    p_train.add_argument("--formula", type=str, required=True, help="Wilkinson formula with an outcome (e.g., 'y ~ x').")
    
    args = parser.parse_args()

    try:
        # --- Command execution ---
        if args.command == "create-matrix":
            df = _load_dataframe_from_csv(args.input_csv)
            y, X = create_model_matrix(df, args.formula)
            X.to_csv(args.output_x, index=False)
            print(f"Design matrix saved to '{args.output_x}'.")
            if y is not None and args.output_y:
                y.to_csv(args.output_y, index=False)
                print(f"Outcome vector saved to '{args.output_y}'.")

        elif args.command == "train-test-transform":
            train_df = _load_dataframe_from_csv(args.train_csv)
            test_df = _load_dataframe_from_csv(args.test_csv)
            train_matrix, test_matrix = transform_train_test_split(train_df, test_df, args.formula)
            
            train_matrix.to_pandas().to_csv(args.output_train_x, index=False)
            print(f"Transformed training matrix saved to '{args.output_train_x}'.")
            test_matrix.to_pandas().to_csv(args.output_test_x, index=False)
            print(f"Transformed testing matrix saved to '{args.output_test_x}'.")

        elif args.command == "train-model":
            import pandas as pd
            import formulaic
            df = _load_dataframe_from_csv(args.input_csv)
            model = build_and_train_model(df, args.formula)
            print("\n--- Model Results ---")
            print(f"Intercept: {model.intercept_}")
            
            y, X = formulaic.model_matrix(args.formula, df)
            coeffs = pd.Series(model.coef_, index=X.model_spec.feature_names)
            print("Coefficients:")
            print(coeffs)
            
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

##########################
#    EXECUTE
##########################

if __name__ == "__main__":
    _handle_cli()
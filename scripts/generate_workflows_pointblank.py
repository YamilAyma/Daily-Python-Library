#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pointblank_utils.py: Workflow accelerator utilities for the Pointblank framework.

This script focuses on meta-tasks like generating validation templates and
integrating YAML-based validations into Python workflows.

Its recommendations include:
1. Use on Jupyter Notebook for a better response.
"""
from __future__ import annotations

import sys
import os
import argparse
from typing import TYPE_CHECKING

# This block is only processed by type checkers.
if TYPE_CHECKING:
    import pandas as pd
    from pointblank import Validate

# Allow running as a script or as a module
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .scripttypes import ScriptMetadata, function_metadata, FUNCTIONS_METADATA
FUNCTIONS_METADATA.clear()

# --- Script Metadata ---
metadata = ScriptMetadata(
    title="Pointblank Workflow Utilities",
    description="Utilities to accelerate data validation workflows, primarily by generating baseline validation YAML files from existing data and create actions",
    version="0.1.0",
    author="with AI",
    email="No Email",
    license="MIT",
    status="development",
    dependencies=["pointblank", "pandas", "pyyaml"],
    tags=["data-validation", "data-quality", "yaml", "automation", "ci-cd", "jupyter"],
    cli=True,
    links=[
        {"Pointblank GitHub Repository": "https://github.com/posit-dev/pointblank"},
        {"Pointblank Documentation": "https://posit-dev.github.io/pointblank/user-guide/"},
        {"with YAML": "https://posit-dev.github.io/pointblank/user-guide/yaml-validation-workflows.html"},
        {"about actions": "https://posit-dev.github.io/pointblank/user-guide/actions.html"}
    ],
    note="SETUP: pip install pointblank pandas pyyaml"
)

##########################
#    SCRIPT FUNCTIONS
##########################
    

@function_metadata(status="development", note="Generates a baseline validation YAML file from a DataFrame.",category="generation", tags=["yaml", "scaffolding", "profiling"])
def generate_baseline_yaml(df: pd.DataFrame, output_yaml_path: str, table_name: str, source_data_path: str) -> None:
    """
    Generates a baseline validation YAML file from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to profile.
        output_yaml_path (str): The path to the output YAML file.
        table_name (str): The name of the table being validated.
        source_data_path (str): The path to the source data file.
        
    Examples:
        >>> generate_baseline_yaml(df, "output.yaml", "my_table", "data.csv")
        
    Raises:
        ImportError: If required libraries are not installed.
    """
    try:
        import yaml
        import pandas as pd
    except ImportError as e:
        raise ImportError("Required libraries not installed. Please run: 'pip install pyyaml pandas'") from e

    print(f"Generating intelligent baseline YAML for table '{table_name}'...")
    
    all_columns = df.columns.tolist()
    
    yaml_structure = {
        "tbl": source_data_path,
        "tbl_name": table_name,
        "label": f"Baseline data quality checks for {table_name}",
        "df_library": "pandas",
        "steps": []
    }
    
    yaml_structure["steps"].append({"rows_distinct": {}})
    yaml_structure["steps"].append({"col_exists": {"columns": all_columns}})
    
    mostly_not_null_cols = [col for col in all_columns if df[col].notna().mean() > 0.9]
    if mostly_not_null_cols:
        yaml_structure["steps"].append({"col_vals_not_null": {"columns": mostly_not_null_cols}})

    for col in all_columns:
        # Detect if a column can be parsed as a date.
        try:
            # Use a strict format to be sure it's a date column.
            pd.to_datetime(df[col], errors='raise', format='%Y-%m-%d')
            is_date_col = True
        except (ValueError, TypeError):
            is_date_col = False

        if is_date_col:
            # If it's a date, use 'col_vals_regex' as 'col_is_date' is not supported in YAML.
            yaml_structure["steps"].append({
                "col_vals_regex": {
                    "columns": [col],
                    "pattern": r"^\d{4}-\d{2}-\d{2}$" # Regex for YYYY-MM-DD
                }
            })
        
        elif pd.api.types.is_numeric_dtype(df[col]):
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            yaml_structure["steps"].append({
                "col_vals_between": {"columns": [col], "left": min_val, "right": max_val}
            })
        
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            if 1 < len(unique_vals) <= 10:
                yaml_structure["steps"].append({
                    "col_vals_in_set": {"columns": [col], "set": unique_vals.tolist()}
                })
                
    with open(output_yaml_path, 'w') as f:
        yaml.dump(yaml_structure, f, sort_keys=False, indent=2, default_flow_style=False)


@function_metadata(status="development", note="Executes a Pointblank validation defined in a YAML file.",category="execution", tags=["yaml", "workflow", "integration"])
def run_validation_from_yaml(yaml_path: str) -> Validate:
    """
    Executes a Pointblank validation defined in a YAML file.

    Args:
        yaml_path (str): The path to the validation.yaml file.
        
    Examples:
    >>> run_validation_from_yaml("output.yaml")

    Returns:
        Validate: The interrogated Pointblank Validate object.
    """
    try:
        import pointblank as pb
    except ImportError as e:
        raise ImportError("Pointblank is not installed. Please run: 'pip install pointblank'") from e
        
    print(f"Running validation from '{yaml_path}'...")
    validation_Validate = pb.yaml_interrogate(yaml=yaml_path)
    return validation_Validate

##########################
#    CLI FUNCTIONS
##########################

def _load_dataframe_from_csv(filepath: str) -> "pd.DataFrame":
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        print(f"Successfully loaded DataFrame from '{filepath}'.")
        return df
    except ImportError:
        raise ImportError("Pandas is not installed. Please run: 'pip install pandas'")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)

def _handle_cli():
    """Manages the Command-Line Interface for this script."""
    parser = argparse.ArgumentParser(description=metadata.description)
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    p_gen = subparsers.add_parser("generate-yaml", help="Generate a baseline validation YAML from a CSV file.")
    p_gen.add_argument("input_csv", type=str, help="Path to the input CSV to profile.")
    p_gen.add_argument("--output-yaml", type=str, required=True, help="Path to save the generated validation.yaml.")
    p_gen.add_argument("--table-name", type=str, default="source_table", help="Name to use for the table in the YAML file.")

    p_run = subparsers.add_parser("run-yaml", help="Run a validation from a YAML file and show a report.")
    p_run.add_argument("yaml_path", type=str, help="Path to the validation.yaml file.")
    p_run.add_argument("--report", action="store_true", help="Display a tabular report in the console after running.")

    args = parser.parse_args()

    try:
        if args.command == "generate-yaml":
            df = _load_dataframe_from_csv(args.input_csv)
            # Pass the original CSV path to be included in the YAML file
            generate_baseline_yaml(df, args.output_yaml, args.table_name, args.input_csv)
            print(f"Baseline validation YAML saved to '{args.output_yaml}'.")

        elif args.command == "run-yaml":
            validation_Validate = run_validation_from_yaml(args.yaml_path)
            if validation_Validate and args.report:
                print("\n--- Validation Report ---")
                validation_Validate.get_tabular_report().show()

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
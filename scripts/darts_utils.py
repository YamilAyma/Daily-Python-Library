#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
darts_utils.py: Utilities for the Darts library for time series forecasting and anomaly detection.

This script provides high-level wrappers to simplify common tasks like forecasting,
model evaluation, and anomaly detection using the Darts library.
"""
from __future__ import annotations

import sys
import os
import argparse
from typing import Callable, Dict, TYPE_CHECKING

# For type checking at create documentation, type hints
if TYPE_CHECKING:
    from darts import TimeSeries

# Allow running as a script or as a module
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .scripttypes import ScriptMetadata, function_metadata, FUNCTIONS_METADATA
FUNCTIONS_METADATA.clear() # Don't remove, it's for documentation

# --- Script Metadata (description is HTML)---
metadata = ScriptMetadata(
    title="Darts Time Series Utilities",
    description="""A collection of utility functions to simplify forecasting, backtesting, and anomaly detection with Darts.    
                    Darts is a Python library designed for user-friendly time series forecasting and anomaly detection. It provides a unified API for various models, from classical statistical methods to deep learning architectures.
                    """,
    version="0.2.1", # Version bump for lazy loading refactor
    author="AI",
    email="No Email",
    license="MIT",
    status="development",
    dependencies=["darts", "pandas", "matplotlib"],
    tags=["time-series", "forecasting", "anomaly-detection", "data-science"],
    cli=True,
    links=[
        {"Darts GitHub Repository": "https://github.com/unit8co/darts"},
        {"Demand Forecasting with Darts: A Tutorial": "https://towardsdatascience.com/demand-forecasting-with-darts-a-tutorial-480ba5c24377/"},
        {"Darts Documentation": "https://unit8co.github.io/darts/"},
        {"Resources": "https://unit8.com/resources/darts-time-series-made-easy-in-python/"}
    ],
    note="SETUP: pip install darts pandas matplotlib"
)

##########################
#    SCRIPT FUNCTIONS
##########################

@function_metadata(status="development", note="A high-level function to quickly train a model, generate a forecast, and save a plot.", category="forecasting", tags=["forecast", "visualization", "ARIMA", "ExponentialSmoothing"])
def quick_forecast_and_plot(
    series: TimeSeries,
    model_name: str = "auto_arima",
    forecast_horizon: int = 12,
    output_plot_path: str = "forecast_plot.png"
) -> TimeSeries:
    """
    Trains a simple forecasting model, makes a prediction, and saves a visualization.

    Args:
        series (TimeSeries): The Darts TimeSeries object to forecast.
        model_name (str, optional): The model to use. Options: "auto_arima", "arima", "exponential". Defaults to "auto_arima".
        forecast_horizon (int, optional): The number of steps to forecast into the future. Defaults to 12.
        output_plot_path (str, optional): Path to save the output plot image. Defaults to "forecast_plot.png".

    Returns:
        TimeSeries: A Darts TimeSeries object containing the forecast.
        
    Raises:
        ValueError: If an unsupported model_name is provided.
        ImportError: If required libraries are not installed.

    Examples:
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load()
        >>> forecast = quick_forecast_and_plot(series, forecast_horizon=24)
        >>> print(forecast)
    """
    try:
        import matplotlib.pyplot as plt
        from darts.models import ExponentialSmoothing, ARIMA, AutoARIMA
    except ImportError as e:
        raise ImportError("Required libraries are not installed. Please run: pip install darts matplotlib") from e

    models = {
        "exponential": ExponentialSmoothing(),
        "arima": ARIMA(),
        "auto_arima": AutoARIMA()
    }
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Unsupported model: {model_name}. Available options are: {list(models.keys())}")

    print(f"Training {model_name} model...")
    model.fit(series)
    
    print(f"Generating forecast for {forecast_horizon} steps...")
    prediction = model.predict(n=forecast_horizon)

    print(f"Saving plot to {output_plot_path}...")
    plt.figure(figsize=(12, 6))
    series.plot(label="Historical Data")
    prediction.plot(label="Forecast")
    plt.title(f"{model_name.capitalize()} Forecast")
    plt.legend()
    plt.savefig(output_plot_path)
    plt.close()
    
    return prediction


@function_metadata(status="development", note="Evaluates a forecasting model using historical backtesting and returns error metrics.", category="evaluation", tags=["backtesting", "metrics", "MAPE", "RMSE"])
def evaluate_model_backtesting(
    series: TimeSeries,
    model,
    start_point: float = 0.7,
    metric: Callable | None = None
) -> Dict[str, float]:
    """
    Performs historical backtesting to evaluate a model's performance.

    It trains the model on an expanding window of the series and evaluates the forecast
    on the immediately following points.

    Args:
        series (TimeSeries): The full time series for backtesting.
        model: An instantiated Darts forecasting model (e.g., ARIMA()).
        start_point (float, optional): The proportion of the series to use as the initial training set. Defaults to 0.7 (70%).
        metric (Callable, optional): The metric function to use for evaluation. Defaults to mape.

    Returns:
        Dict[str, float]: A dictionary containing the average metric score.

    Examples:
        >>> from darts.models import ExponentialSmoothing
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.metrics import rmse
        >>> series = AirPassengersDataset().load()
        >>> model = ExponentialSmoothing()
        >>> results = evaluate_model_backtesting(series, model, metric=rmse)
        >>> print(f"Average RMSE: {results['avg_metric']:.2f}")
    """
    try:
        from darts.metrics import mape
    except ImportError as e:
        raise ImportError("Darts library is not installed. Please run: pip install darts") from e
    
    if metric is None:
        metric = mape
        
    print("Starting historical backtesting...")
    historical_forecasts = model.historical_forecasts(
        series,
        start=start_point,
        forecast_horizon=1,
        verbose=True
    )
    
    metric_score = metric(series, historical_forecasts)
    print(f"Backtesting complete. Average {metric.__name__}: {metric_score:.2f}")
    
    return {"avg_metric": metric_score}


@function_metadata(status="development", note="Identifies anomalies based on historical quantiles. No forecasting model is used.", category="anomaly_detection", tags=["outliers", "detection", "quantile", "statistics"])
def detect_anomalies_by_quantile(
    series: TimeSeries,
    low_quantile: float = 0.05,
    high_quantile: float = 0.95
) :
    """
    Detects anomalies using a QuantileDetector based on the series' own history.

    It calculates fixed quantile thresholds from the entire series. Any data points
    falling outside these thresholds are flagged as anomalies.

    Args:
        series (TimeSeries): The time series to analyze for anomalies.
        low_quantile (float, optional): The lower bound quantile. Defaults to 0.05.
        high_quantile (float, optional): The upper bound quantile. Defaults to 0.95.

    Returns:
        TimeSeries: A binary series where 1 indicates an anomaly and 0 is normal.

    Examples:
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load()
        >>> anomalies = detect_anomalies_by_quantile(series)
        >>> print(f"Detected {int(anomalies.sum().values()[0][0])} anomalies.")
    """
    try:
        from darts.ad.detectors import QuantileDetector
    except ImportError as e:
        raise ImportError("Darts library is not installed. Please run: pip install darts") from e
        
    print("Detecting anomalies using historical quantiles...")
    
    detector = QuantileDetector(low_quantile=low_quantile, high_quantile=high_quantile)
    detector.fit(series)
    anomalies = detector.detect(series)
    
    num_anomalies = int(anomalies.sum().values()[0][0])
    print(f"Detection complete. Found {num_anomalies} potential anomalies.")
    
    return anomalies


##########################
#    CLI FUNCTIONS
##########################

def _load_series_from_csv(filepath: str, time_col: str, value_col: str) -> "TimeSeries":
    """Helper to load a TimeSeries from a CSV file."""
    try:
        import pandas as pd
        from darts import TimeSeries
    except ImportError as e:
        raise ImportError("Required libraries are not installed. Please run: pip install darts pandas") from e

    try:
        df = pd.read_csv(filepath, parse_dates=[time_col])
        series = TimeSeries.from_dataframe(df, time_col, value_col)
        print(f"Successfully loaded series from '{filepath}'.")
        return series
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

def _handle_cli():
    """Manages the Command-Line Interface for this script."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=metadata.description)
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Forecast command ---
    p_forecast = subparsers.add_parser("forecast", help="Train a model and generate a forecast plot.")
    p_forecast.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    p_forecast.add_argument("--time-col", type=str, required=True, help="Name of the timestamp column.")
    p_forecast.add_argument("--value-col", type=str, required=True, help="Name of the value column to forecast.")
    p_forecast.add_argument("--model", type=str, default="auto_arima", choices=["exponential", "arima", "auto_arima"], help="Model to use.")
    p_forecast.add_argument("--horizon", type=int, default=12, help="Number of steps to forecast.")
    p_forecast.add_argument("--output-plot", type=str, default="forecast_plot.png", help="Path to save the forecast plot.")

    # --- Anomaly detection command ---
    p_anomaly = subparsers.add_parser("anomalies", help="Detect anomalies in a time series.")
    p_anomaly.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    p_anomaly.add_argument("--time-col", type=str, required=True, help="Name of the timestamp column.")
    p_anomaly.add_argument("--value-col", type=str, required=True, help="Name of the value column to analyze.")
    p_anomaly.add_argument("--low-q", type=float, default=0.05, help="Lower quantile for anomaly detection.")
    p_anomaly.add_argument("--high-q", type=float, default=0.95, help="Higher quantile for anomaly detection.")
    p_anomaly.add_argument("--output-csv", type=str, help="Path to save a CSV with detected anomalies.")

    args = parser.parse_args()

    try:
        # --- Command execution ---
        series = _load_series_from_csv(args.input_csv, args.time_col, args.value_col)
        
        if args.command == "forecast":
            quick_forecast_and_plot(
                series=series,
                model_name=args.model,
                forecast_horizon=args.horizon,
                output_plot_path=args.output_plot
            )
            print("Forecast command executed successfully.")

        elif args.command == "anomalies":
            anomalies = detect_anomalies_by_quantile(
                series=series,
                low_quantile=args.low_q,
                high_quantile=args.high_q
            )
            if args.output_csv:
                anomalies.pd_dataframe().to_csv(args.output_csv)
                print(f"Anomalies saved to {args.output_csv}")
            print("Anomalies command executed successfully.")
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


##########################
#    EXECUTE
##########################

if __name__ == "__main__":
    _handle_cli()
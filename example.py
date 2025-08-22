from scripts import darts_utils as du

# Use: Put your python scripts in the currently directory and execute -- with darts_utils.py
from darts.datasets import AirPassengersDataset
series = AirPassengersDataset().load()
forecast = du.quick_forecast_and_plot(series, forecast_horizon=24)
print(forecast)

# from darts.models import ExponentialSmoothing
# from darts.datasets import AirPassengersDataset
# series = AirPassengersDataset().load()
# model = ExponentialSmoothing()
# results = du.evaluate_model_backtesting(series, model, metric=du.rmse)
# print(f"Average RMSE: {results['avg_metric']:.2f}")


# from darts.datasets import AirPassengersDataset
# series = AirPassengersDataset().load()
# anomalies = du.detect_anomalies_by_quantile(series)
# print(f"Detected {anomalies.sum().values()[0][0]} anomalies.")
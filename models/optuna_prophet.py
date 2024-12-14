import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.integration import LightGBMPruningCallback
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['datetime'] = pd.to_datetime(data['Date'])
    data.rename(columns={'Day Ahead Auction Price EUR/MWh': 'price'}, inplace=True)
    data = data[['datetime', 'price']]
    data.set_index('datetime', inplace=True)
    data = data.asfreq('h')
    return data

# Split data
def split_data(data, split_date):
    train = data.loc[:split_date]
    test = data.loc[split_date:]
    return train, test

# Objective function for Optuna
def objective(trial, y_train):
    # Suggest hyperparameters
    seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
    changepoint_prior_scale = trial.suggest_loguniform("changepoint_prior_scale", 0.01, 1.0)
    seasonality_prior_scale = trial.suggest_loguniform("seasonality_prior_scale", 0.01, 10.0)
    holidays_prior_scale = trial.suggest_loguniform("holidays_prior_scale", 0.01, 10.0)

    # Train/validation split
    train_cutoff = int(len(y_train) * 0.8)
    train_data = y_train.iloc[:train_cutoff]
    valid_data = y_train.iloc[train_cutoff:]

    # Train Prophet model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    model.fit(train_data.reset_index().rename(columns={"datetime": "ds", "price": "y"}))

    # Forecast on validation set
    future = model.make_future_dataframe(periods=len(valid_data), freq="H")
    forecast = model.predict(future)

    # Calculate MAE
    valid_forecast = forecast.iloc[-len(valid_data):]
    mae = mean_absolute_error(valid_data["price"], valid_forecast["yhat"])
    return mae

# Optimize hyperparameters with Optuna
def optimize_hyperparameters(y_train):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, y_train), n_trials=50)
    return study.best_params

# Train final model with best hyperparameters
def train_final_model(y_train, best_params):
    model = Prophet(
        seasonality_mode=best_params["seasonality_mode"],
        changepoint_prior_scale=best_params["changepoint_prior_scale"],
        seasonality_prior_scale=best_params["seasonality_prior_scale"],
        holidays_prior_scale=best_params["holidays_prior_scale"],
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    model.fit(y_train.reset_index().rename(columns={"datetime": "ds", "price": "y"}))
    return model

# Visualize results
def visualize_results(model, y_train, forecast_path):
    # Forecast
    future = model.make_future_dataframe(periods=8760 * 5, freq="H")
    forecast = model.predict(future)

    # Save forecast
    forecast.to_csv(forecast_path)
    print(f"Forecast saved to '{forecast_path}'")

    # Plot
    fig = go.Figure()

    # Training data
    fig.add_trace(go.Scatter(
        x=y_train.index, y=y_train["price"],
        mode="lines", name="Training Data"
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines", name="Forecast"
    ))

    fig.update_layout(
        title="Training Data and Forecast",
        xaxis_title="Time", yaxis_title="Price (EUR/MWh)",
        template="plotly_white"
    )
    fig.show()

if __name__ == "__main__":
    # Load and preprocess
    data_path = "data/Day Ahead Auction Prices.csv"
    forecast_path = "data/prophet_5yr_forecast_optuna.csv"
    data = load_and_preprocess_data(data_path)

    # Split data
    split_date = "2023-04-30"
    y_train, y_test = split_data(data, split_date)

    # Optimize hyperparameters
    best_params = optimize_hyperparameters(y_train)
    print("Best parameters:", best_params)

    # Train final model
    best_model = train_final_model(y_train, best_params)

    # Visualize results
    visualize_results(best_model, y_train, forecast_path)

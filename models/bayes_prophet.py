import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.forecasting.compose import ForecastingPipeline
from sktime.forecasting.fbprophet import Prophet
from skopt import BayesSearchCV
from sktime.forecasting.base import ForecastingHorizon
from joblib import parallel_backend

def load_and_preprocess_data(filepath):
    """Load and preprocess the day-ahead auction price data."""
    data = pd.read_csv(filepath)
    data['datetime'] = pd.to_datetime(data['Date'])
    data.rename(columns={'Day Ahead Auction Price EUR/MWh': 'price'}, inplace=True)
    data = data[['datetime', 'price']]
    data.set_index('datetime', inplace=True)
    data = data.asfreq('h')
    return data

def split_data(data, split_date):
    """Split the data into training and testing sets."""
    y_train = data.loc[:split_date]
    y_test = data.loc[split_date:]
    return y_train, y_test

def setup_forecasting_pipeline():
    """Set up the forecasting pipeline with Prophet."""
    return ForecastingPipeline(
        steps=[("forecaster", Prophet(add_country_holidays={"country_name": "Germany"}))]
    )

def bayesian_hyperparameter_search(pipeline, y_train, fh):
    """Perform Bayesian hyperparameter search with cross-validation."""
    param_space = {
        'forecaster__seasonality_mode': ['additive', 'multiplicative'],
        'forecaster__yearly_seasonality': [True],
        'forecaster__weekly_seasonality': [True],
        'forecaster__daily_seasonality': [True],
        'forecaster__seasonality_prior_scale': (0.01, 10.0, 'log-uniform'),
        'forecaster__changepoint_prior_scale': (0.01, 1.0, 'log-uniform'),
        'forecaster__holidays_prior_scale': (0.01, 10.0, 'log-uniform')
    }

    cv = SlidingWindowSplitter(
        fh=[24 * 365 * 5],
        window_length=24 * 365 * 5,
        step_length=24 * 365
    )

    bayes_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=param_space,
        cv=cv,
        scoring=MeanAbsoluteError(),
        n_jobs=-1,
        n_iter=50,
        verbose=5,
        random_state=42
    )

    with parallel_backend("loky"):
        bayes_search.fit(y_train[['price']], fh=fh)

    return bayes_search

def evaluate_and_visualize_results(best_model, y_train, fh, forecast_path):
    """Evaluate the best model and visualize the training fit and forecast."""
    forecast = best_model.predict(fh)

    # Save forecast
    forecast.to_csv(forecast_path)
    print(f"Forecast saved to '{forecast_path}'")

    # Plot interactive visualization
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_train.index,
        y=y_train['price'],
        mode='lines',
        name='Training Data'
    ))

    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast,
        mode='lines',
        name='Forecast'
    ))

    fig.update_layout(
        title='Training Data and 5-Year Forecast',
        xaxis_title='Time',
        yaxis_title='Price (EUR/MWh)',
        template='plotly_white'
    )

    fig.show()

if __name__ == "__main__":
    # Load and preprocess data
    data_path = 'data/Day Ahead Auction Prices.csv'
    forecast_path = "data/prophet_5yr_forecast.csv"
    day_ahead = load_and_preprocess_data(data_path)

    # Split data
    split_date = '2023-04-30'
    y_train, y_test = split_data(day_ahead, split_date)

    # Define forecasting horizon
    fh = ForecastingHorizon([i for i in range(1, 8761 * 5)], is_relative=True)

    # Set up pipeline
    pipeline = setup_forecasting_pipeline()

    # Perform Bayesian search
    bayes_search = bayesian_hyperparameter_search(pipeline, y_train, fh)

    # Get best parameters and model
    best_params = bayes_search.best_params_
    best_model = bayes_search.best_estimator_

    print("Best parameters:", best_params)
    print("Best model:", best_model)

    # Evaluate and visualize results
    evaluate_and_visualize_results(best_model, y_train, fh, forecast_path)

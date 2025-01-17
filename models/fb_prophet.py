import itertools
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def load_day_ahead_prices(file_path):
    """
    Load and preprocess the day-ahead auction price data.

    Parameters:
        file_path (str): Path to the CSV file containing day-ahead auction price data.

    Returns:
        pd.DataFrame: Preprocessed time series data with hourly frequency.
    """
    day_ahead = pd.read_csv(file_path)
    day_ahead['datetime'] = pd.to_datetime(day_ahead['Date'])
    day_ahead.rename(columns={'Day Ahead Auction Price EUR/MWh': 'price'}, inplace=True)
    day_ahead = day_ahead[['datetime', 'price']]
    day_ahead.set_index('datetime', inplace=True)

    # Set hourly frequency for the index to ensure consistent time series data
    day_ahead = day_ahead.asfreq('h')
    return day_ahead


def tune_hyperparameters(df, param_grid, horizon):
    """
    Perform grid search for hyperparameter tuning using cross-validation with 5 equal parts.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'ds' (datetime) and 'y' (values to forecast).
        param_grid (dict): Dictionary of hyperparameter lists for grid search.
        horizon (str): Forecast horizon (e.g., '30 days').

    Returns:
        pd.DataFrame: DataFrame containing hyperparameters and their corresponding RMSE.
    """
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store RMSEs for each parameter set

    # Calculate the total training period duration
    train_start_date = df['ds'].min()
    train_end_date = df['ds'].max()
    train_duration = train_end_date - train_start_date

    # Calculate the interval size (approximately equal 5 parts)
    interval_size = train_duration / 5

    # Generate the cutoffs based on the interval size
    cutoffs = [train_start_date + interval_size * i for i in range(1, 5)]

    for params in all_params:
        m = Prophet(**params)
        m.add_country_holidays(country_name='DE')
        m.fit(df)
        df_cv = cross_validation(m, cutoffs=cutoffs, horizon=horizon, parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Store tuning results in a DataFrame
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    return tuning_results

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model's performance using RMSE, MAE, and MAPE.

    Parameters:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.

    Returns:
        dict: Dictionary containing RMSE, MAE, and MAPE.
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Filter out zero values in y_true to avoid division by zero
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def plot_residuals(y_true, y_pred):
    """
    Plot the residuals of the model.

    Parameters:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
    """
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=residuals, mode='markers', name='Residuals'))
    fig.update_layout(title='Residuals Plot', xaxis_title='Index', yaxis_title='Residuals')
    fig.show()

def plot_cv_results(df_cv):
    """
    Visualize the cross-validation scores for each validation period.

    Parameters:
        df_cv (pd.DataFrame): Cross-validation results containing columns 'cutoff', 'yhat', 'y', etc.
    """
    fig = go.Figure()
    for cutoff_date in df_cv['cutoff'].unique():
        cutoff_df = df_cv[df_cv['cutoff'] == cutoff_date]
        fig.add_trace(go.Scatter(x=cutoff_df['ds'], y=cutoff_df['y'], mode='lines', name=f'True - {cutoff_date}'))
        fig.add_trace(go.Scatter(x=cutoff_df['ds'], y=cutoff_df['yhat'], mode='lines', name=f'Predicted - {cutoff_date}'))
    fig.update_layout(title='Cross-Validation Results', xaxis_title='Date', yaxis_title='Values')
    fig.show()

def plot_training_and_forecast(df, forecast, forecast_full):
    """
    Plot the training data, 1-year forecast, and 5-year forecast together with prediction intervals using Plotly.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'ds' (datetime) and 'y' (values to forecast).
        forecast (pd.DataFrame): DataFrame containing 1-year forecasted values and intervals.
        forecast_full (pd.DataFrame): DataFrame containing 5-year forecasted values and intervals.
    """
    fig = go.Figure()

    # Plot training data
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'], mode='lines', name='Training Data',
        line=dict(color='blue')
    ))

    # Plot 1-year forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines', name='1-Year Forecast',
        line=dict(color='green')
    ))

    # Plot 5-year forecast
    fig.add_trace(go.Scatter(
        x=forecast_full['ds'], y=forecast_full['yhat'], mode='lines', name='5-Year Forecast',
        line=dict(color='orange')
    ))

    # Add prediction intervals for the 5-year forecast as a filled area
    fig.add_trace(go.Scatter(
        x=list(forecast_full['ds']) + list(forecast_full['ds'])[::-1],
        y=list(forecast_full['yhat_upper']) + list(forecast_full['yhat_lower'])[::-1],
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255, 165, 0, 0)'),
        name='5-Year Prediction Interval'
    ))

    fig.update_layout(
        title='Training Data and Forecast with Prediction Intervals',
        xaxis_title='Date',
        yaxis_title='Price (EUR/MWh)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    fig.show()

# Update main function to call the revised plot function
def main():
    """
    Main function to load data, tune Prophet hyperparameters, and display results.
    """
    # File path to the CSV data
    file_path = '../data/Day Ahead Auction Prices.csv'

    # Load and preprocess the data
    day_ahead = load_day_ahead_prices(file_path)

    # Prepare the DataFrame for Prophet
    df = day_ahead.reset_index().rename(columns={'datetime': 'ds', 'price': 'y'})

    # Get the last date in the dataset
    last_date = df['ds'].max()

    # Define train-test split
    train_end_date = last_date - pd.Timedelta(days=365)
    train_df = df[df['ds'] <= train_end_date]
    test_df = df[df['ds'] > train_end_date]

    # Define the parameter grid
    param_grid = {
        'growth': ['flat'],
        'yearly_seasonality': [True],
        'weekly_seasonality': [True],
        'daily_seasonality': [True],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1, 10],
        'holidays_prior_scale': [0.01, 0.1, 1, 10]
    }

    # Perform hyperparameter tuning
    horizon = '365 days'
    tuning_results = tune_hyperparameters(train_df, param_grid, horizon)

    # Fit the best model
    best_params = tuning_results.loc[tuning_results['rmse'].idxmin()].to_dict()
    print("best_params", best_params)
    m = Prophet(**{k: v for k, v in best_params.items() if k != 'rmse'})
    m.add_country_holidays(country_name='DE')
    m.fit(train_df)

    # Forecast future values
    future = m.make_future_dataframe(periods=365 * 24, freq='h')  # 1-year forecast
    forecast = m.predict(future)

    # Evaluate model on test set
    y_test = test_df['y'].values
    y_pred = forecast.loc[forecast['ds'].isin(test_df['ds']), 'yhat'].values

    metrics = evaluate_model(y_test, y_pred)
    print("Evaluation Metrics:", metrics)

    # Refit the model on the entire dataset
    print("Refitting the model on the entire dataset for a 5-year forecast...")
    m_full = Prophet(**{k: v for k, v in best_params.items() if k != 'rmse'})
    m_full.add_country_holidays(country_name='DE')
    m_full.fit(df)

    # Forecast 5 years into the future
    future_full = m_full.make_future_dataframe(periods=5 * 365 * 24, freq='h')  # 5-year forecast
    forecast_full = m_full.predict(future_full)

    # Save the 5-year forecast to a CSV file
    forecast_full.to_csv('../data/fb_prophet_5year_forecast.csv', index=False)
    print("5-year forecast saved as fb_prophet_5year_forecast.csv")

    # Plot the training data, 1-year forecast, and 5-year forecast
    plot_training_and_forecast(train_df, forecast, forecast_full)

if __name__ == "__main__":
    main()
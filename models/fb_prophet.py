import itertools
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

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

import itertools
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation

def tune_hyperparameters(df, param_grid, horizon):
    """
    Perform grid search for hyperparameter tuning using cross-validation with 5 equal parts.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'ds' (datetime) and 'y' (values to forecast).
        param_grid (dict): Dictionary of hyperparameter lists for grid search.
        horizon (str): Forecast horizon (e.g., '30 days').

    Returns:
        pd.DataFrame: DataFrame containing hyperparameters and their corresponding RMSE, MAE, and CV fold results.
    """
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    avg_rmses = []  # Store average RMSEs for each parameter set
    avg_maes = []   # Store average MAEs for each parameter set
    cv_results = []  # Store individual fold results (cutoff date, RMSE, MAE)

    # Calculate the total training period duration
    train_start_date = df['ds'].min()
    train_end_date = df['ds'].max()

    print(f"Training period: {train_start_date} to {train_end_date}")
    
    for params in all_params:
        print(f"Training model with hyperparameters: {params}")
        m = Prophet(**params)
        m.add_country_holidays(country_name='DE')
        m.fit(df)
        
        # Perform cross-validation
        df_cv = cross_validation(m, initial='731 days', period='365 days', horizon=horizon, parallel="processes")
        
        # Calculate RMSE and MAE manually
        df_cv['squared_error'] = (df_cv['yhat'] - df_cv['y']) ** 2  # Squared errors for RMSE
        df_cv['absolute_error'] = abs(df_cv['yhat'] - df_cv['y'])  # Absolute errors for MAE
        
        # Aggregate by cutoff date to calculate mean RMSE and MAE per fold
        grouped = df_cv.groupby('cutoff').agg(
            fold_rmse=('squared_error', lambda x: (x.mean()) ** 0.5),  # RMSE = sqrt(mean squared error)
            fold_mae=('absolute_error', 'mean')  # MAE = mean absolute error
        ).reset_index()

        # Add RMSE and MAE to the main lists (averaged over all folds)
        avg_rmses.append(grouped['fold_rmse'].mean())  # Average RMSE over all folds
        avg_maes.append(grouped['fold_mae'].mean())    # Average MAE over all folds

        # Add results for each CV fold (cutoff date, RMSE, MAE)
        for _, row in grouped.iterrows():
            fold_result = {
                'cutoff_date': row['cutoff'],
                'fold_rmse': row['fold_rmse'],
                'fold_mae': row['fold_mae']
            }
            # Append the fold results with the hyperparameters
            cv_results.append({**params, **fold_result})

    # Store tuning results in a DataFrame
    tuning_results = pd.DataFrame(all_params)
    tuning_results['avg_rmse'] = avg_rmses
    tuning_results['avg_mae'] = avg_maes

    # Convert the cross-validation results into a DataFrame
    cv_results_df = pd.DataFrame(cv_results)

    # Merge the tuning results with the CV fold details
    tuning_results = pd.merge(tuning_results, cv_results_df, how='left', on=list(param_grid.keys()))

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
    eps = 1e-10
    mape = np.mean(np.abs(((y_true+eps) - y_pred) / (y_true + eps))) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}




import plotly.graph_objects as go
import numpy as np

def plot_residuals(y_true, y_pred, start_date=None, end_date=None):
    """
    Plot residuals as a bar plot for the 1-year forecast using Plotly.

    Parameters:
        y_true (pd.DataFrame): DataFrame with columns 'ds' (datetime) and 'y' (actual values).
        y_pred (pd.Series or pd.DataFrame): Series or DataFrame containing predicted values indexed by 'ds'.
        start_date (str or pd.Timestamp, optional): Start date to filter the residuals (e.g., '2023-01-01').
        end_date (str or pd.Timestamp, optional): End date to filter the residuals (e.g., '2023-12-31').
    """

    # Align actual and predicted values based on the 'ds' index
    y_true = y_true.set_index('ds')
    y_pred.index = y_true.index

    print(y_true)
    print(y_pred)

    residuals = y_true['y'] - y_pred
    residuals.index = y_true.index

    print(residuals)

    # Filter residuals by date range if specified
    if start_date:
        residuals = residuals.loc[start_date:]
    if end_date:
        residuals = residuals.loc[:end_date]

    # Calculate error metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))

    # Create bar colors (red for negative, green for positive residuals)
    colors = ['red' if res < 0 else 'green' for res in residuals]

    # Create residual plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=residuals.index, 
        y=residuals.values, 
        marker_color=colors, 
        name='Residuals'
    ))

    # Add horizontal lines for MAE and RMSE with annotations
    fig.add_hline(y=mae, line_dash="dot", line_color="blue", annotation_text=f"MAE: {mae:.2f}", annotation_position="bottom right")
    fig.add_hline(y=rmse, line_dash="dash", line_color="purple", annotation_text=f"RMSE: {rmse:.2f}", annotation_position="top right")

    fig.update_layout(
        title=f"Residuals of 1-Year Forecast ({start_date} to {end_date})",
        xaxis_title="Date",
        yaxis_title="Residuals",
        template="plotly_white",
        bargap=0.2,
        height=500,
        width=1000
    )

    return fig



import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

def plot_residuals2(y_true, y_pred, start_date=None, end_date=None):
    """
    Plot actual vs forecast values with residuals using Plotly subplots.

    Parameters:
        y_true (pd.DataFrame): DataFrame with columns 'ds' (datetime) and 'y' (actual values).
        y_pred (pd.Series or pd.DataFrame): Series or DataFrame containing predicted values indexed by 'ds'.
        start_date (str, optional): Start date to filter the data.
        end_date (str, optional): End date to filter the data.
    """
    # Align actual and predicted values based on the 'ds' index
    y_true = y_true.set_index('ds')
    y_pred.index = y_true.index

    # Filter based on the given date range
    if start_date and end_date:
        y_true = y_true.loc[start_date:end_date]
        y_pred = y_pred.loc[start_date:end_date]

    residuals = y_true['y'] - y_pred

    # Calculate error metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))

    # Create bar colors (red for negative, green for positive residuals)
    colors = ['red' if res < 0 else 'green' for res in residuals]

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0)

    # Top subplot: actual vs forecast values
    fig.add_trace(go.Scatter(x=y_true.index, y=y_true['y'], mode='lines', name='Actual', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred.values, mode='lines', name='Forecast', line=dict(color='orange')), row=1, col=1)


    # Bottom subplot: residuals
    fig.add_trace(go.Bar(x=residuals.index, y=residuals.values, marker_color=colors), row=2, col=1)


    # Add horizontal lines for MAE and RMSE with annotations
    fig.add_hline(y=mae, line_dash="dot", line_color="blue", row=2, col=1)
    fig.add_annotation(
        x=residuals.index[len(residuals)-1], 
        y=mae - 20,  # Move MAE annotation up
        text=f"MAE: {mae:.2f}", 
        showarrow=False, 
        font=dict(color="blue", size=12), 
        row=2, col=1
    )

    fig.add_hline(y=rmse, line_dash="dash", line_color="purple", row=2, col=1)
    fig.add_annotation(
        x=residuals.index[len(residuals)-1], 
        y=rmse + 20,  # Move RMSE annotation down
        text=f"RMSE: {rmse:.2f}", 
        showarrow=False, 
        font=dict(color="purple", size=12), 
        row=2, col=1
    )

    title = f"Residuals of 1 Year Forecast ({start_date} to {end_date})" if start_date and end_date else "Residuals of 1 Year Forecast"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (EUR/MWh)",
        template="plotly_white",
        height=700,
        width=1000,
        showlegend=True
    )

    return fig




import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_cv_results(df, tuning_results, horizon, param_grid):
    """
    Plot the cross-validation results for the best hyperparameter set (lowest avg_rmse).

    Parameters:
        df (pd.DataFrame): The original time series data with columns 'ds' and 'y'.
        tuning_results (pd.DataFrame): The tuning results containing hyperparameters,
                                       'cutoff_date', 'fold_rmse', and 'fold_mae'.
        horizon (str): Forecast horizon (e.g., '30 days') used in cross-validation.
    """
    # Select the best parameter set with the lowest average RMSE
    best_param_set = tuning_results.groupby(list(param_grid.keys())).agg({'avg_rmse': 'mean'}).reset_index()
    best_params = best_param_set.loc[best_param_set['avg_rmse'].idxmin()]
    print("Best Hyperparameters:", best_params.to_dict())


    # Filter all 5 folds of the best parameter set
    best_folds = tuning_results[
        (tuning_results['growth'] == best_params['growth']) &
        (tuning_results['yearly_seasonality'] == best_params['yearly_seasonality']) &
        (tuning_results['weekly_seasonality'] == best_params['weekly_seasonality']) &
        (tuning_results['daily_seasonality'] == best_params['daily_seasonality']) &
        (tuning_results['seasonality_mode'] == best_params['seasonality_mode']) &
        (tuning_results['seasonality_prior_scale'] == best_params['seasonality_prior_scale']) &
        (tuning_results['holidays_prior_scale'] == best_params['holidays_prior_scale'])
    ]

    if len(best_folds) != 5:
        print(len(best_folds))
        print(f"Warning: {len(best_folds)} folds found for the best parameter set!")

    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(10, 7))
    for i, (ax, cutoff_date) in enumerate(zip(axes, best_folds['cutoff_date'].unique())):
        train_data = df[df['ds'] <= cutoff_date]
        forecast_end = cutoff_date + pd.Timedelta(horizon)
        forecast_data = df[(df['ds'] > cutoff_date) & (df['ds'] <= forecast_end)]

        # Get RMSE and MAE for this fold
        fold_data = best_folds[best_folds['cutoff_date'] == cutoff_date]
        rmse = fold_data['fold_rmse'].values[0]
        mae = fold_data['fold_mae'].values[0]

        ax.plot(df['ds'], df['y'], color='gray', alpha=0.3)
        ax.plot(train_data['ds'], train_data['y'], color='blue', label='Training Data')
        ax.plot(forecast_data['ds'], forecast_data['y'], color='red', label='Forecast Horizon')
        ax.axhline(y=0, color='grey', linewidth=0.5)  # Add horizontal line at y=0
        ax.grid(axis='x', which='both')
        # ax.grid(True)
        ax.set_ylim(bottom=-200)

        label_text = f"Fold {i+1}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"
        ax.text(forecast_data['ds'].iloc[-1] + pd.Timedelta(horizon) * 0.15, forecast_data['y'].min() * 1.1,
                label_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        # Get the unique years from the 'ds' column
        year_ticks = pd.date_range(start='2016-01-01', end='2024-01-01', freq='AS')  # 'AS' means 'Year Start'
        year_labels = [str(year.year) for year in year_ticks]

        if i == 4:
            ax.set_xlabel('Date')
            ax.set_xticks(year_ticks)  # Set the x-ticks to be at the start of each year
            ax.set_xticklabels(year_labels)  # Set the x-tick labels to be the years
        else:
            ax.set_xticklabels([])

        ax.set_ylabel('Price [EUR/MWh]')
        ax.legend()

    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    return fig


import plotly.graph_objects as go

def plot_1_year_forecast(df, forecast):
    """
    Plot the training data, in-sample fit, and 1-year out-of-sample forecast with prediction intervals.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'ds' (datetime) and 'y' (values to forecast).
        forecast (pd.DataFrame): DataFrame containing both in-sample and 1-year out-of-sample forecasted values and intervals.
    """
    fig = go.Figure()

    # Plot training data
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'], mode='lines', name='Training Data',
        line=dict(color='blue')
    ))

    # Define the cutoff date for the in-sample fit
    cutoff_date = df['ds'].max()

    # Plot in-sample fit (before cutoff_date)
    in_sample = forecast[forecast['ds'] <= cutoff_date]
    fig.add_trace(go.Scatter(
        x=in_sample['ds'], y=in_sample['yhat'], mode='lines', name='In-Sample Fit',
        line=dict(color='lightblue')
    ))

    # Add prediction intervals for the in-sample fit
    fig.add_trace(go.Scatter(
        x=list(in_sample['ds']) + list(in_sample['ds'])[::-1],
        y=list(in_sample['yhat_upper']) + list(in_sample['yhat_lower'])[::-1],
        fill='toself',
        fillcolor='rgba(173, 216, 230, 0.3)',  # lightblue shade
        line=dict(color='rgba(173, 216, 230, 0)'),
        name='In-Sample Prediction Interval'
    ))

    # Plot 1-year out-of-sample forecast (after cutoff_date)
    out_of_sample = forecast[forecast['ds'] > cutoff_date]
    fig.add_trace(go.Scatter(
        x=out_of_sample['ds'], y=out_of_sample['yhat'], mode='lines', name='1-Year Forecast',
        line=dict(color='red')
    ))

    # Add prediction intervals for the 1-year forecast
    fig.add_trace(go.Scatter(
        x=list(out_of_sample['ds']) + list(out_of_sample['ds'])[::-1],
        y=list(out_of_sample['yhat_upper']) + list(out_of_sample['yhat_lower'])[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',  # red shade
        line=dict(color='rgba(255, 0, 0, 0)'),
        name='1-Year Prediction Interval'
    ))

    fig.update_layout(
        title='Out-of-Sample Model Validation',
        xaxis_title='Date',
        yaxis_title='Price (EUR/MWh)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        width=1000,  # Adjust width for LaTeX slide
        height=500  # Adjust height for LaTeX slide
    )

    return fig

def plot_5_year_forecast(df, forecast_full):
    """
    Plot the training data, in-sample fit, and 5-year out-of-sample forecast with prediction intervals.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'ds' (datetime) and 'y' (values to forecast).
        forecast_full (pd.DataFrame): DataFrame containing both in-sample and 5-year out-of-sample forecasted values and intervals.
    """
    fig = go.Figure()

    # Plot training data
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'], mode='lines', name='Training Data',
        line=dict(color='blue')
    ))

    # Define the cutoff date for the in-sample fit
    cutoff_date = df['ds'].max()

    # Plot in-sample fit (before cutoff_date)
    in_sample = forecast_full[forecast_full['ds'] <= cutoff_date]
    fig.add_trace(go.Scatter(
        x=in_sample['ds'], y=in_sample['yhat'], mode='lines', name='In-Sample Fit',
        line=dict(color='lightblue')
    ))

    # Add prediction intervals for the in-sample fit
    fig.add_trace(go.Scatter(
        x=list(in_sample['ds']) + list(in_sample['ds'])[::-1],
        y=list(in_sample['yhat_upper']) + list(in_sample['yhat_lower'])[::-1],
        fill='toself',
        fillcolor='rgba(173, 216, 230, 0.3)',  # lightblue shade
        line=dict(color='rgba(173, 216, 230, 0)'),
        name='In-Sample Prediction Interval'
    ))

    # Plot 5-year out-of-sample forecast (after cutoff_date)
    out_of_sample = forecast_full[forecast_full['ds'] > cutoff_date]
    fig.add_trace(go.Scatter(
        x=out_of_sample['ds'], y=out_of_sample['yhat'], mode='lines', name='5-Year Forecast',
        line=dict(color='orange')
    ))

    # Add prediction intervals for the 5-year forecast
    fig.add_trace(go.Scatter(
        x=list(out_of_sample['ds']) + list(out_of_sample['ds'])[::-1],
        y=list(out_of_sample['yhat_upper']) + list(out_of_sample['yhat_lower'])[::-1],
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',  # orange shade
        line=dict(color='rgba(255, 165, 0, 0)'),
        name='5-Year Prediction Interval'
    ))

    fig.update_layout(
        title='5-year Prophet Forecast',
        xaxis_title='Date',
        yaxis_title='Price (EUR/MWh)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        width=1000,  # Adjust width for LaTeX slide
        height=500  # Adjust height for LaTeX slide
    )

    return fig


# Update main function to call the revised plot function
def main():
    """
    Main function to load data, tune Prophet hyperparameters, and display results.
    """
    # File path to the CSV data
    file_path = 'data/Day Ahead Auction Prices.csv'

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
        # 'changepoint_prior_scale': [0.001, 0.01, 0.1], #  range [0.001, 0.5] 
        'seasonality_prior_scale': [0.01, 0.1, 1, 10], # range:[0.01, 10]
        'holidays_prior_scale': [0.01, 0.1, 1, 10] # range [0.01, 10]
    }

    # Perform hyperparameter tuning
    horizon = '365 days'
    tuning_results = tune_hyperparameters(train_df, param_grid, horizon)

    # Call the function with your data
    fig = plot_cv_results(df, tuning_results, horizon, param_grid)
    fig.show()
    fig.savefig('images/cross_validation.svg',  bbox_inches="tight")

    # Fit the best model
    best_params = tuning_results.loc[tuning_results['avg_rmse'].idxmin()].to_dict()
    m = Prophet(**{k: v for k, v in best_params.items() if k not in ['avg_rmse', 'avg_mae', 'fold_rmse', 'fold_mae', 'cutoff_date']})
    m.add_country_holidays(country_name='DE')
    m.fit(train_df)

    # Forecast future values
    future = m.make_future_dataframe(periods=365 * 24, freq='h')  # 1-year forecast
    forecast = m.predict(future)

    # Evaluate model on test set
    y_test = test_df['y']
    y_pred = forecast.loc[forecast['ds'].isin(test_df['ds']), 'yhat']

    metrics = evaluate_model(y_test.values, y_pred.values)

    print(f"Out-of-sample RMSE: {metrics['RMSE']:.2f}")
    print(f"Out-of-sample MAE: {metrics['MAE']:.2f}")
    print(f"Out-of-sample MAPE: {metrics['MAPE']:.2f}")

    # Plot the training data, 1-year forecast, and 5-year forecast
    fig = plot_1_year_forecast(train_df, forecast)
    fig.show()

    fig = plot_residuals(y_true=test_df, y_pred=y_pred, start_date='2024-01-01', end_date='2024-02-01')
    fig.show()

    fig = plot_residuals2(y_true=test_df, y_pred=y_pred, start_date='2024-01-01', end_date='2024-02-01')
    fig.show()

    # Refit the model on the entire dataset
    print("Refitting the model on the entire dataset for a 5-year forecast...")
    m_full = Prophet(**{k: v for k, v in best_params.items() if k not in ['avg_rmse', 'avg_mae', 'fold_rmse', 'fold_mae', 'cutoff_date']})
    m_full.add_country_holidays(country_name='DE')
    m_full.fit(df)

    # Forecast 5 years into the future
    future_full = m_full.make_future_dataframe(periods=5 * 365 * 24, freq='h')  # 5-year forecast
    forecast_full = m_full.predict(future_full)

    # Save the 5-year forecast to a CSV file
    forecast_full.to_csv('data/fb_prophet_5year_forecast.csv', index=False)
    print("5-year forecast saved as fb_prophet_5year_forecast.csv")

    # Plot the training data, and 5-year forecast
    fig = plot_5_year_forecast(df, forecast_full)

    # Save as SVG
    # fig.write_image("images/5_year_forecast.svg")
    fig.show()

    fig = m_full.plot_components(forecast_full, weekly_start=1)
    fig.savefig('images/final_prophet_components.svg',  bbox_inches="tight")

if __name__ == "__main__":
    main()
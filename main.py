from models.fb_prophet import *
from utils import *
import argparse
import json
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

def main(cutoff_date=None,params=None):
    """
    Main function to load data, tune Prophet hyperparameters, and display results.
    """
    # File path to the CSV data
    file_path = 'data/Day Ahead Auction Prices.csv'

    # Load and preprocess the data
    day_ahead = load_day_ahead_prices(file_path)

    # Prepare the DataFrame for Prophet
    df = day_ahead.reset_index().rename(columns={'datetime': 'ds', 'price': 'y'})

    #cutoff the data
    if cutoff_date is not None:
        df = df[df['ds'] <= cutoff_date]

    # Get the last date in the dataset
    last_date = df['ds'].max()

    # Define train-test split
    train_end_date = last_date - pd.Timedelta(days=365)
    train_df = df[df['ds'] <= train_end_date]
    test_df = df[df['ds'] > train_end_date]

    if params is None:
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
        best_params = tuning_results.loc[tuning_results['avg_rmse'].idxmin()].to_dict()
        best_params.pop('cutoff_date')
        #save the best params
        if cutoff_date is None:
            with open('pfc/best_params.json', 'w') as f:
                json.dump(best_params, f)
        else:
            with open(f'pfc/best_params_{cutoff_date}.json', 'w') as f:
                json.dump(best_params, f)
    else:
        with open(params) as f:
            best_params = json.load(f)

    
    # Fit the best model
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

    # Refit the model on the entire dataset
    print("Refitting the model on the entire dataset for a 5-year forecast...")
    m_full = Prophet(**{k: v for k, v in best_params.items() if k not in ['avg_rmse', 'avg_mae', 'fold_rmse', 'fold_mae', 'cutoff_date']})
    m_full.add_country_holidays(country_name='DE')
    m_full.fit(df)

    # Forecast 5 years into the future
    future_full = m_full.make_future_dataframe(periods=5 * 365 * 24, freq='h')  # 5-year forecast
    forecast_full = m_full.predict(future_full)

    # Save the 5-year forecast to a CSV file
    if cutoff_date is None:
        forecast_full.to_csv('pfc/fb_prophet_5year_forecast.csv', index=False)
        print("5-year forecast saved as fb_prophet_5year_forecast.csv")
    else:
        forecast_full.to_csv(f'pfc/fb_prophet_5year_forecast_{cutoff_date}.csv', index=False)
        print(f"5-year forecast saved as fb_prophet_5year_forecast_{cutoff_date}.csv")

    forecast_full = forecast_full[forecast_full['ds'] >= last_date]
    forecast_full["timestamp"] = forecast_full["ds"]

    pfc, arb_free = arbitrage_pipeline_benth(forecast_full,plot_figure=True,print_arbitrage=False)
    pfc = pfc[['timestamp','corrected']]

    if cutoff_date is None:
        pfc.to_csv('pfc/pfc.csv', index=False)
        print("PFC saved as pfc.csv")
    else:
        pfc.to_csv(f'pfc/pfc_{cutoff_date}.csv', index=False)
        print(f"PFC saved as pfc_{cutoff_date}.csv")

    

#parse cutoff date
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hourly PFC')
    parser.add_argument('--cutoff_date', type=str, help='Cutoff date for the PFC')
    parser.add_argument('--parameters', type=str, help='Path to the parameters file')
    args = parser.parse_args()
    main(cutoff_date=args.cutoff_date, params=args.parameters)
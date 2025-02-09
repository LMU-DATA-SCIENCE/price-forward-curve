# Generating an hourly Price Forward Curve

## Overview
This method main.py automatically creates an arbitrage free HPFC. It includes data preprocessing, hyperparameter tuning, model evaluation, forecasting seasonality five years into the future and arbitrage correction with the Benth  method

## Features
- Loads and preprocesses day-ahead auction price data
- Allows optional cutoff date for training data to create PFC for another day
- Tunes Prophet hyperparameters automatically (or allows manual parameter input if best parameters have already been fitted)
- Evaluates model performance using RMSE, MAE, and MAPE
- Generates hourly forecasts for up to five years
- Saves forecast and performs arbitrage correction to output a final PFC to CSV files

## Requirements
- Python 3.x

   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script with optional arguments for a cutoff date and hyperparameter file:
```bash
python main.py --cutoff_date YYYY-MM-DD --parameters path/to/params.json
```

### Arguments:
- `--cutoff_date`: (Optional) Specify a date to create the PFC on, YYYY-MM-DD format, must be within dataset range
- `--parameters`: (Optional) Path to a JSON file with prefitted model hyperparameters

## Output Files
- `pfc/best_params.json`: Stores the best hyperparameters found
- `pfc/fb_prophet_5year_forecast.csv`: 5-year hourly seasonality forecast with Prophet
- `pfc/pfc.csv`: Arbitrage corrected Price Forward Curve

## Example Usage
To run with automatic hyperparameter tuning:
```bash
python main.py
```
To run with a cutoff date for the PFC and predefined hyperparameters:
```bash
python main.py --cutoff_date 2023-01-01 --parameters pfc/best_params.json
```

## License
This project is licensed under the MIT License.



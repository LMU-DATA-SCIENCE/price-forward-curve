import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
from scipy.optimize import minimize


def load_forwards_df(directory='data/forwards', columns_to_drop=['Price', 'ChangeComment', 'ValidationStatus']):
    """
    Load and preprocess multiple forward curves datasets from a specified directory.

    Parameters:
    - directory (str): Path to the directory containing the forward data CSV files.
    - columns_to_drop (list): List of column names to drop, if present in each dataset.

    Returns:
    - DataFrame: Concatenated and cleaned DataFrame with all forward data.
    """
    dataframes = []
    file_paths = glob(os.path.join(directory, '*.csv'))

    for path in file_paths:
        # Extract identifier from filename, e.g., 'D' from 'EEX_POWER_FUT_DE_BASE_D_2021-2024.csv'
        identifier = os.path.basename(path).split('_')[-2]
        df = pd.read_csv(path, sep=';')
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
        df['Identifier'] = identifier
        dataframes.append(df)

    forwards = pd.concat(dataframes)
    forwards.dropna(subset=['Settlement'], inplace=True)
    forwards['TimeStamp'] = pd.to_datetime(forwards['TimeStamp'], errors='coerce')
    forwards['TimeStamp'] = forwards['TimeStamp'].apply(lambda x: x.replace(tzinfo=None))
    forwards['Begin'] = pd.to_datetime(forwards['Begin'], errors='coerce')
    forwards['Begin'] = forwards['Begin'].apply(lambda x: x.replace(tzinfo=None))
    forwards['End'] = pd.to_datetime(forwards['End'], errors='coerce')
    forwards['End'] = forwards['End'].apply(lambda x: x.replace(tzinfo=None))


    forwards[['Settlement', 'Open', 'High', 'Low', 'Close']] = forwards[['Settlement', 'Open', 'High', 'Low', 'Close']].apply(
        lambda x: x.str.replace(',', '.').astype(float)
    )
    forwards.sort_values(by=['TimeStamp', 'Identifier'], inplace=True)
    return forwards

def get_forwards(timestamp, start=None, end=None, periods=['D', 'W', 'WE', 'M', 'Q', 'Y']):
    """
    Loads forward data from CSV files, processes it, and returns a filtered DataFrame.

    Parameters:
        timestamp (str): The date or timestamp to filter records.
        start (str, optional): Starting date for filtering 'Begin' column.
        end (str, optional): Ending date for filtering 'End' column.
        periods (list, optional): List of period identifiers to include (default is ['D', 'W', 'WE', 'M', 'Q', 'Y']).
    
    Returns:
        pd.DataFrame: Filtered DataFrame sorted by 'Begin'.
    """
    
    forwards = pd.DataFrame()
    for file in glob('data/forwards/*.csv'):
        forwards = pd.concat([forwards, pd.read_csv(file, sep=';')], ignore_index=True)
    
    forwards.drop(columns=['Price', 'ChangeComment', 'ValidationStatus'], inplace=True)
    forwards.dropna(subset=['Settlement'], inplace=True)
    
    forwards['TimeStamp'] = pd.to_datetime(forwards['TimeStamp'], errors='coerce')
    forwards['Begin'] = pd.to_datetime(forwards['Begin'], errors='coerce')
    forwards['End'] = pd.to_datetime(forwards['End'], errors='coerce')
    
    forwards['Settlement'] = forwards['Settlement'].str.replace(',', '.').astype(float)
    forwards['Open'] = forwards['Open'].str.replace(',', '.').astype(float)
    forwards['High'] = forwards['High'].str.replace(',', '.').astype(float)
    forwards['Low'] = forwards['Low'].str.replace(',', '.').astype(float)
    forwards['Close'] = forwards['Close'].str.replace(',', '.').astype(float)
    
    forwards['Identifier'] = forwards['Identifier'].str[22:]
    forwards.sort_values(by=['TimeStamp'], inplace=True)

    # print(f"wrong contracts: \n {forwards[(forwards['End'] <= forwards['TimeStamp'])]}")

    mask_relevant_contracts = (
        (
            (forwards['TimeStamp'].astype(str).str.contains(timestamp))
            | (forwards['TimeStamp'].astype(str) == timestamp)
        )
        & (forwards['End'] > forwards['TimeStamp'])
    )
    mask_relevant_periods = forwards['Identifier'].isin(periods)

    data = forwards[mask_relevant_contracts & mask_relevant_periods]
    
    if end:
        data = data[data['End'] <= pd.to_datetime(end, utc=2)]
    if start:
        data = data[data['Begin'] + pd.Timedelta(hours=2) >= pd.to_datetime(start, utc=2)]
    
    return data.sort_values(by=['Begin'])


def plot_forwards(forwards, date, periods=['D', 'W', 'WE', 'M', 'Q', 'Y']):
    """
    Plots forward contracts for the given date and periods.

    Parameters:
    forwards (pd.DataFrame): DataFrame containing forward data with columns ['TimeStamp', 'Identifier', 'Settlement', 'Begin', 'End']
    date (str): Date string to filter the forwards data
    periods (list): List of period identifiers to include in the plot

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    data = forwards[(forwards['TimeStamp'].astype(str).str.contains(date)) & (forwards['Identifier'].isin(periods))]
    color_map = {
        'D': 'red',
        'W': 'blue',
        'WE': 'green',
        'M': 'purple',
        'Q': 'orange',
        'Y': 'black'
    }
    
    fig, ax = plt.subplots(figsize=(15, 7))
    for index, row in data.iterrows():
        ax.hlines(y=row['Settlement'], xmin=row['Begin'], xmax=row['End'], 
                  color=color_map.get(row['Identifier'], 'gray'), lw=2)
    
    # Set up legend and labels only for unique periods
    handles = [plt.Line2D([0], [0], color=color_map[period], lw=2) for period in periods if period in color_map]
    ax.legend(handles, periods, title="Periods")
    
    # Labels and Title
    ax.set_xlabel("Time")
    ax.set_ylabel("Settlement Price")
    ax.set_title(f"Forwards on {date}")

    plt.show()
    return fig

def plot_forecast_forwards(timestamp, forecast=None, data=None):
    """
    Plot a forecast line along with historical forward data for specified periods.

    Parameters:
    ----------
    timestamp : str
        The specific date or timestamp used to fetch historical forward data if `data` is not provided.
    forecast : pd.DataFrame, optional
        A DataFrame containing forecast data with columns 'timestamp' and 'yhat'. The forecast is
        plotted as a single black line.
    data : pd.DataFrame, optional
        A DataFrame containing historical forward data with columns 'Begin', 'End', 'Settlement', 
        and 'Identifier'. If not provided, data is fetched using `get_forwards`.
    
    Returns:
    -------
    None
        Displays a Plotly figure with historical forwards and forecast data.
    """
    if not data:
        data = get_forwards(timestamp, start=forecast['timestamp'].min(), end=forecast['timestamp'].max(), periods=['D', 'W', 'WE', 'M', 'Q', 'Y'])
    fig = go.Figure()
    unique_identifiers = data['Identifier'].unique()
    color_map = {identifier: color for identifier, color in zip(unique_identifiers, pc.qualitative.Plotly)}
    added_identifiers = set()
    # Plot the forecast data as a black line if provided
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=forecast['timestamp'],
            y=forecast['yhat'],
            mode='lines',
            name='forecast',
            line=dict(color='black')  
        ))
    # Plot each forward entry as a horizontal line, with unique colors per identifier
    for index, row in data.iterrows():
        show_legend = row['Identifier'] not in added_identifiers
        fig.add_trace(go.Scatter(
            x=[row['Begin'], row['End']],
            y=[row['Settlement'], row['Settlement']],
            mode='lines',
            name=row['Identifier'] if show_legend else None,
            showlegend=show_legend,
            line=dict(color=color_map[row['Identifier']])
        ))
        added_identifiers.add(row['Identifier'])
    fig.show()
    return fig


def get_arbitrage_opportunities_in_forwards(forwards, date):
    """
    Check for arbitrage opportunities between different forward contracts on a given date 
    and return the resulting opportunities as a DataFrame.utils.py§

    Parameters:
    - forwards (DataFrame): DataFrame with forward curve data.
    - date (str): Date to evaluate for arbitrage opportunities.

    Returns:
    - DataFrame: DataFrame with arbitrage opportunities, if any.
    """

    # Filter data for the given date
    df = forwards[(forwards['TimeStamp'].astype(str).str.contains(date))|(forwards['TimeStamp'].astype(str)==date)].copy()

    # Calculate contract length in days
    df['contract_length_days'] = (pd.to_datetime(df['End'], utc=True) - pd.to_datetime(df['Begin'], utc=True)).dt.days #### check if calculation is correct

    # Define possible contract pairs (short vs long term)
    pairs = [('D', 'W'), ('D', 'WE'), ('W', 'M'), ('M', 'Q'), ('Q', 'Y')]

    arbitrage_opportunities = []

    # Iterate over all possible pairs of contracts
    for short, long in pairs:
        short_contracts = df[df['Identifier'] == short]
        long_contracts = df[df['Identifier'] == long]

        for _, long_contract in long_contracts.iterrows():
            long_begin, long_end = long_contract['Begin'], long_contract['End']
            short_begin = long_begin
            short_contract_series = []
            # Find all short contracts that are within the long contract period
            # --> TODO: OPTIMIZE THIS LOOP!
            while short_begin < long_end:
                short_contract = short_contracts[short_contracts['Begin'] == short_begin]
                if short_contract.empty:
                    break
                short_contract_series.append(short_contract)
                short_begin = short_contract['End'].iloc[0]
            # Calculate the average price of the short contracts in the short_contract_series
            if short_begin == long_end:
                total_days = np.sum([short["contract_length_days"] for short in short_contract_series])
                mean_short = np.sum([(short["contract_length_days"]/total_days) * short['Settlement'] for short in short_contract_series]) ### add weighting by short contract length!
                long_settlement = long_contract['Settlement']
                # Arbitrage opportunity exists if long contract price differs more than €0.00 
                # from average price of short contracts
                if abs(long_settlement - mean_short) > 0.01:
                    arbitrage_opportunities.append({
                        'timestamp': date,
                        'long': long,
                        'short': short,
                        'begin': long_begin,
                        'end': long_end,
                        'long_settlement': long_settlement,
                        'short_settlement': mean_short,
                        'profit': abs(long_settlement - mean_short) # should this be the absolute value?
                    })

    return pd.DataFrame(arbitrage_opportunities)


def get_restrictions(timestamp, start_date, end_date, data=None):
    """
    Generate a restriction matrix and adjusted settlement values for forward contracts over a specified date range.

    Parameters:
        timestamp (str): Date or timestamp used to fetch forward contracts if `data` is not provided.
        start_date (str): Starting date for the restriction matrix.
        end_date (str): Ending date for the restriction matrix.
        data (pd.DataFrame, optional): DataFrame with contract data ('Begin', 'End', 'Settlement'). If not provided, 
                                       data is retrieved using `get_forwards`.

    Returns:
        tuple (np.ndarray, np.ndarray):
            - `C`: A 2D array where each row represents a contract and each column represents an hour in the specified 
              range. Entries are 1 if the contract is active during that hour; otherwise, 0.
            - `s`: A 1D array of settlement values adjusted by each contract’s active duration in hours.
    """
    length = ((pd.to_datetime(end_date, utc=2) - pd.to_datetime(start_date, utc=2)).days + 1) * 24
    if data is None:
        contracts = get_forwards(timestamp, start=start_date, end=end_date)
    else:
        contracts = data
    s = contracts['Settlement'].values
    d = len(s)
    C = np.zeros((d, length))
    for i in range(d):
        start_contract = contracts.iloc[i].Begin
        end_contract = contracts.iloc[i].End
        
        start_index = (start_contract - pd.to_datetime(start_date, utc=2) + pd.Timedelta(hours=1)).days * 24
        end_index = (end_contract - pd.to_datetime(start_date, utc=2) + pd.Timedelta(hours=1)).days * 24
        
        C[i, start_index:end_index] = 1
        s[i] = s[i] * (end_index - start_index)
    
    return C, s

def arbitrage_correction(timestamp, forecast, lambda_1=0, optimizer='trust-constr', loss='L2'):
    """
    Perform arbitrage correction on a forecast to ensure consistency with forward contract data.

    Parameters:
        timestamp (str): Date or timestamp used for retrieving forward contract data.
        forecast (pd.DataFrame): DataFrame with forecast data, including 'timestamp' and 'yhat' columns.
        lambda_1 (float, optional): Regularization parameter for smoothness in the optimization.
        optimizer (str, optional): Optimization method used by `scipy.optimize.minimize` (default is 'trust-constr').
        loss (str, optional): Loss function for optimization, either 'L1' for absolute error or 'L2' for squared error.

    Returns:
        tuple: 
            - np.ndarray: Corrected forecast values, either optimized or original if optimization fails.
            - pd.DataFrame: DataFrame containing any detected arbitrage opportunities.
    """

    start = forecast['timestamp'].min()
    end = forecast['timestamp'].max()

    # Plot initial forecast
    plot_forecast_forwards(timestamp, forecast)

    # Check for arbitrage in forward data
    forwards = get_forwards(timestamp, start, end)
    arbitrage_df = get_arbitrage_opportunities_in_forwards(forwards, timestamp)
    if not arbitrage_df.empty:
        print(f"Arbitrage opportunities found: {arbitrage_df}")
    else:
        print("No arbitrage opportunities found in forwards")

    # Set up initial forecast values and objective function
    yhat = np.array(forecast['yhat'].values)
    x0 = yhat

    def objective_function(x):
        if loss == 'L1':
            return np.sum(np.abs(x - yhat)) + lambda_1 * np.sum(np.square(np.diff(x))) ## diff only between days, months, years -> contract cut-off
        elif loss == 'L2':
            return np.sum(np.square(x - yhat)) + lambda_1 * np.sum(np.square(np.diff(x)))
        else:
            raise ValueError("Invalid loss function. Choose 'L1' or 'L2'.")

    # Get restriction matrix and constraints
    A_eq, b_eq = get_restrictions(timestamp, start, end)

    # Run optimization to correct forecast
    result = minimize(
        objective_function,
        x0,
        method=optimizer,
        constraints={'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq}
    )

    #calculate arrays of  Daily, weekly and yearly seasonalities pre and post optimization and plot comparisons
    daily = np.zeros(24)
    weekly = np.zeros(24*7)
    yearly = np.zeros(24*365)
    daily_opt = np.zeros(24)
    weekly_opt = np.zeros(24*7)
    yearly_opt = np.zeros(24*365)
    for i in range(24):
        daily[i] = np.mean(yhat[i::24])
        daily_opt[i] = np.mean(result.x[i::24])
    for i in range(24*7):
        weekly[i] = np.mean(yhat[i::24*7])
        weekly_opt[i] = np.mean(result.x[i::24*7])
    for i in range(24*365):
        yearly[i] = np.mean(yhat[i::24*365])
        yearly_opt[i] = np.mean(result.x[i::24*365])
                                
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Daily Average", "Weekly Average", "Yearly Average"))
    fig.add_trace(go.Scatter(x=np.arange(24), y=daily, name='Initial Prediction', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(24), y=daily_opt, name='Arbitrage corrected', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(24*7), y=weekly, name='Initial Prediction', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(24*7), y=weekly_opt, name='Arbitrage corrected', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(24*365), y=yearly, name='Initial Prediction', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=np.arange(24*365), y=yearly_opt, name='Arbitrage corrected', line=dict(color='red')), row=3, col=1)
    fig.update_layout(title_text="Seasonalities comparison")
    fig.show()

    # Check optimization result and plot corrected forecast
    if result.success:
        print("Optimum found")
        plot_forecast_forwards(timestamp, pd.DataFrame({'timestamp': forecast['timestamp'], 'yhat': result.x}))
        return result.x, arbitrage_df
    else:
        print("Optimization failed:", result.message)
        plot_forecast_forwards(timestamp, pd.DataFrame({'timestamp': forecast['timestamp'], 'yhat': result.x}))
        return yhat, arbitrage_df
    
def get_restrictions_adrian(forwards, test_forecast):
    """
    Generate a restriction matrix and adjusted settlement values for forward contracts over a specified forecast period.

    Parameters:
        forwards (pd.DataFrame): DataFrame with forward contracts data.
        test_forecast (pd.DataFrame): DataFrame with forecast data containing 'timestamp' column.

    Returns:
        tuple (np.ndarray, np.ndarray):
            - `A_eq`: A 2D array where each row represents a contract and each column represents an hour in the forecast period.
              Entries are 1 if the contract is active during that hour; otherwise, 0.
            - `b_eq`: A 1D array of settlement values adjusted by each contract’s active duration in hours.
    """
    forwards = forwards.sort_values(by=['Begin', "Identifier"])

    # Filter the forwards to only include the contracts that are active during the test forecast
    # fwds = forwards[(forwards['Begin'] >= test_forecast['timestamp'].min()) & (forwards['End'] <= test_forecast['timestamp'].max())].copy()
    fwds = forwards.copy()

    # Get the first and last date of the test forecast
    first_forecast_hour = test_forecast['timestamp'].min()
    last_forecast_hour = test_forecast['timestamp'].max()

    # Calculate the number of hours between the first and last timestamp including both boundaries
    hours = (last_forecast_hour - first_forecast_hour).days * 24 + (last_forecast_hour - first_forecast_hour).seconds // 3600

    # Create a zeros matrix with the number of rows equal to the number of contracts
    # and the number of columns equal to the number of forecast hours
    A_eq = np.zeros((len(fwds), len(test_forecast)))

    for i, row in fwds.iterrows():
        # Find the indices of the forecast hours that are within the contract hours
        begin = row['Begin']
        end = row['End']
        begin_index = (begin - first_forecast_hour).days * 24
        end_index = (end - first_forecast_hour).days * 24
        
        print("-----")
        print(row['Identifier'])
        print(begin, first_forecast_hour)
        print((begin - first_forecast_hour).days)
        print(end_index-begin_index)
        print("-----")

        A_eq[i, begin_index:end_index] = 1

    # Calculate the total settlement price of the contracts for all contract hours
    fwds["contract_hours"] = (fwds["End"] - fwds["Begin"]).dt.days * 24
    b_eq = fwds['Settlement'].values * fwds['contract_hours'].values

    return A_eq, b_eq

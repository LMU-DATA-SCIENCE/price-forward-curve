import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from glob import glob
from datetime import datetime


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

def get_arbitrage_opportunities_in_forwards(forwards, date):
    """
    Check for arbitrage opportunities between different forward contracts on a given date 
    and return the resulting opportunities as a DataFrame.

    Parameters:
    - forwards (DataFrame): DataFrame with forward curve data.
    - date (str): Date to evaluate for arbitrage opportunities.

    Returns:
    - DataFrame: DataFrame with arbitrage opportunities, if any.
    """

    # Filter data for the given date
    df = forwards[(forwards['TimeStamp'].astype(str).str.contains(date))|(forwards['TimeStamp'].astype(str)==date)].copy()

    # Calculate contract length in days
    df['contract_length_days'] = (pd.to_datetime(df['End'], utc=True) - pd.to_datetime(df['Begin'], utc=True)).dt.days

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
                mean_short = np.mean([short['Settlement'] for short in short_contract_series])
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


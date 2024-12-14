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
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix

def get_forwards(timestamp=None, start=None, end=None, periods=['D', 'W', 'WE', 'M', 'Q', 'Y']):
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
    if timestamp:
        mask_relevant_contracts = (
            (
                (forwards['TimeStamp'].astype(str).str.contains(timestamp))
                | (forwards['TimeStamp'].astype(str) == timestamp)
            )
            & (forwards['End'] > forwards['TimeStamp'])
        )
    else:
        mask_relevant_contracts = [True] * len(forwards)

    mask_relevant_periods = forwards['Identifier'].isin(periods)

    filtered_forwards = forwards[mask_relevant_contracts & mask_relevant_periods]
    
    if end:
        filtered_forwards = filtered_forwards[filtered_forwards['End'] <= pd.to_datetime(end, utc=2)] ## TODO: check TZ handling
    if start:
        filtered_forwards = filtered_forwards[filtered_forwards['Begin'] + pd.Timedelta(hours=2) >= pd.to_datetime(start, utc=2)] ## TODO: check TZ handling
    
    return filtered_forwards.sort_values(by=['Begin'])

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
    if not data: # does not  with data given
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

    # Calculate contract length in hours
    df['contract_length_hours'] = (pd.to_datetime(df['End'], utc=True) - pd.to_datetime(df['Begin'], utc=True)).dt.total_seconds() // 3600

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
                total_hours = np.sum([short["contract_length_hours"] for short in short_contract_series])
                mean_short = np.sum([(short["contract_length_hours"]/total_hours) * short['Settlement'] for short in short_contract_series]) ### add weighting by short contract length!
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
                        'profit': abs(long_settlement - mean_short) 
                    })

    return pd.DataFrame(arbitrage_opportunities)


def get_restrictions(timestamp, start_date, end_date, data=None):
    """
    Generate a restriction matrix, adjusted settlement values and a contract cut-off difference matrix
    for forward contracts over a specified date range.

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
            - `D`: A 2D array representing the difference matrix at contract cut-off times, used in penalty calculation.
    """
    length = int((pd.to_datetime(end_date, utc=True) - pd.to_datetime(start_date, utc=True) + pd.Timedelta(hours=1)).total_seconds() // 3600)
    if data is None:
        contracts = get_forwards(timestamp, start=start_date, end=end_date)
    else:
        contracts = data
    
    s = contracts['Settlement'].values
    d = len(s)
    C = np.zeros((d, length))
    D = np.zeros((d, length)) # difference matrix at contract cut-off -> used in penalty

    for i in range(d):
        start_contract = contracts.iloc[i].Begin
        end_contract = contracts.iloc[i].End
        
        start_index = int((start_contract - pd.to_datetime(start_date, utc=2)).total_seconds() // 3600)
        end_index = int((end_contract - pd.to_datetime(start_date, utc=2)).total_seconds() // 3600)
        
        # CONTRAINTS
        C[i, start_index:end_index] = 1
        s[i] = s[i] * (end_index - start_index)

        # PENALTY
        if start_index > 0:
            D[i, start_index-1] = -1
            D[i, start_index] = 1
    
    return C, s, D

def optimize_with_cvxpy(yhat, A_eq, b_eq, D, lambda_1, loss):
    """
    Perform optimization using cvxpy and OSQP solver, aligning with scipy's objective function.

    Parameters:
        yhat (np.ndarray): Forecast values.
        A_eq (np.ndarray): Equality constraint matrix.
        b_eq (np.ndarray): Equality constraint vector.
        D (np.ndarray): Matrix for penalty computation.
        lambda_1 (float): Regularization parameter.
        loss (str): Loss function ('L1' or 'L2').

    Returns:
        np.ndarray: Optimized forecast values.
    """
    x = cp.Variable(len(yhat))

    # Deviation from forecast (L1 or L2 loss)
    if loss == 'L1':
        deviation_from_forecast = cp.norm1(x - yhat)
    elif loss == 'L2':
        deviation_from_forecast = cp.sum_squares(x - yhat)
    else:
        raise ValueError("Invalid loss function. Choose 'L1' or 'L2'.")

    # Penalty term
    penalty = lambda_1 * cp.sum_squares(D @ x)

    # Objective function: Deviation + Penalty
    objective = cp.Minimize(deviation_from_forecast + penalty)

    # Constraints
    constraints = [A_eq @ x == b_eq]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if problem.status == cp.OPTIMAL:
        return x.value
    else:
        raise ValueError(f"Optimization with cvxpy failed: {problem.status}")


def optimize_with_scipy(yhat, A_eq, b_eq, D, lambda_1, loss, scipy_method):
    """
    Perform optimization using scipy.optimize.minimize.

    Parameters:
        yhat (np.ndarray): Forecast values.
        A_eq (np.ndarray): Equality constraint matrix.
        b_eq (np.ndarray): Equality constraint vector.
        D (np.ndarray): Matrix for penalty computation.
        lambda_1 (float): Regularization parameter.
        loss (str): Loss function ('L1' or 'L2').
        scipy_method (str): Optimization method for scipy.optimize.minimize.

    Returns:
        np.ndarray: Optimized forecast values.
    """
    x0 = yhat

    def objective_function(x):
        if loss == 'L1':
            deviation_from_forecast = np.sum(np.abs(x - yhat))
        elif loss == 'L2':
            deviation_from_forecast = np.sum(np.square(x - yhat))
        else:
            raise ValueError("Invalid loss function. Choose 'L1' or 'L2'.")
        penalty = lambda_1 * np.sum(np.square(D @ x))
        return deviation_from_forecast + penalty

    constraints = {'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq}

    result = minimize(
        objective_function,
        x0,
        method=scipy_method,
        constraints=constraints
    )

    if result.success:
        return result.x
    else:
        raise ValueError(f"Optimization with scipy failed: {result.message}")


def arbitrage_correction(forecast, forwards, lambda_1=0.1, optimizer='scipy', scipy_method='trust-constr', loss='L2'):
    """
    Perform arbitrage correction with scipy or cvxpy-based solvers.

    Parameters:
        forecast (pd.DataFrame): Forecast DataFrame with 'timestamp' and 'yhat' columns.
        forwards: Forward data for arbitrage checks.
        lambda_1 (float): Regularization parameter.
        optimizer (str): Optimization framework ('scipy' or 'cvxpy').
        scipy_method (str): Method for scipy optimizer (default: 'trust-constr').
        loss (str): Loss function for scipy optimizer ('L1' or 'L2').

    Returns:
        pd.DataFrame: Corrected forecast with columns 'timestamp' and 'yhat'.
    """
    yhat = np.array(forecast['yhat'].values)

    start_date = forecast['timestamp'].min()
    end_date = forecast['timestamp'].max()
    C, s, D = get_restrictions(forecast['timestamp'].iloc[0], start_date, end_date, forwards)
    A_eq, b_eq = C, s

    # Perform optimization
    if optimizer == 'scipy':
        corrected_values = optimize_with_scipy(yhat, A_eq, b_eq, D, lambda_1, loss, scipy_method)
    elif optimizer == 'cvxpy':
        corrected_values = optimize_with_cvxpy(yhat, A_eq, b_eq, D, lambda_1, loss)
    else:
        raise ValueError("Invalid optimizer. Choose 'scipy' or 'cvxpy'.")

    # Create DataFrame with corrected forecast
    corrected_forecast = pd.DataFrame({
        'timestamp': forecast['timestamp'].values,
        'yhat': corrected_values
    })

    return corrected_forecast


###################################################
############# Benth 2007 implementation ###########
###################################################

def partition_forwards(forwards, begin_forecast):
    """
    Partition timeline into granular intervals based on forward contract data and calculate the corresponding prices.

    Parameters:
        forwards (pd.DataFrame): DataFrame with forward contract data.

    Returns:
        tuple (list, np.ndarray):
            - `t`: List of timestamps t_0, t_1, ..., t_n.
            - `F`: DataFrame with (F_C,T_s,T_e) for each contract

    """
    timestamps = sorted(set(forwards['Begin']).union(set(forwards['End'])))

    #convert timestamps to hours since begin_forecast
    if type(begin_forecast) == str:
        begin_forecast = pd.to_datetime(begin_forecast,utc=True)
    t = [int((timestamp - begin_forecast).total_seconds() // 3600) for timestamp in timestamps]
    #bring forwards into the same format
    F = pd.DataFrame(columns=['F_C', 'T_s', 'T_e'])
    for i, row in forwards.iterrows():
        T_s = int((row['Begin'] - begin_forecast).total_seconds() // 3600)
        T_e = int((row['End'] - begin_forecast).total_seconds() // 3600)
        F.loc[i] = [row['Settlement'], T_s, T_e]
    F['T_s'] = F['T_s'].astype(int)
    F['T_e'] = F['T_e'].astype(int)
    return t, F

def construct_H(t):
    """
    Constructs the symmetric matrix H for the quadratic term based on the knot vector t.
    
    Parameters:
        t (array-like): Granular vector of knots, with n+1 elements (t_0, t_1, ..., t_n).
    
    Returns:
        H (ndarray): (5n x 5n) symmetric matrix.
    """
    n = len(t) - 1  # Number of intervals
    H = np.zeros((5 * n, 5 * n))  # Initialize the full matrix
    
    # Numerical coefficients matrix (static values)
    coeff_matrix = np.array([
        [144/5, 18, 8, 0, 0],
        [18,    12, 6, 0, 0],
        [8,      6, 4, 0, 0],
        [0,      0, 0, 0, 0],
        [0,      0, 0, 0, 0]
    ])

    for i in range(n):
        idx = 5 * i  # Starting index for the block
        
        delta_1 = t[i+1] - t[i]
        delta_2 = t[i+1]**2 - t[i]**2
        delta_3 = t[i+1]**3 - t[i]**3
        delta_4 = t[i+1]**4 - t[i]**4
        delta_5 = t[i+1]**5 - t[i]**5

        # Calculate delta terms for this interval
        delta_matrix = np.array([
            [delta_5, delta_4, delta_3, 0, 0],
            [delta_4, delta_3, delta_2, 0, 0],
            [delta_3, delta_2, delta_1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # Define the 5x5 block for this interval
        h_i = coeff_matrix * delta_matrix
        
        # Place the block in the appropriate location in H
        H[idx:idx+5, idx:idx+5] = h_i
    
    return H

def construct_A_and_b(t, F, s_t):
    """
    Constructs the restriction matrix A and the vector b for constraints (6)-(10).
    
    Parameters:
        t (array-like): Knot vector with n+1 elements [t_0, t_1, ..., t_n].
        F (array-like): Future price vector for m contracts splitted up into disjunct periods t.
        s_t (array-like): Seasonality vector with hourly price forecast covering at least the period [t_0, t_n].
        
    Returns:
        A (ndarray): Restriction matrix for constraints.
        b (ndarray): Vector for the right-hand side of constraints.
    """
    n = len(t) - 1  # Number of intervals
    m = len(F)  # Number of market prices
    
    # Total number of constraints
    num_constraints = (n - 1) * 3 + 2 + m
    num_variables = 5 * n
    A = np.zeros((num_constraints, num_variables))
    b = np.zeros(num_constraints)
    
    constraint_idx = 0
    
    # Continuity constraints (6)
    for i in range(n - 1):
        contract1_start_idx = 5 * i
        contract2_end_idx = 5 * (i + 2) # spline i and i+1

        A[constraint_idx, contract1_start_idx:contract2_end_idx] = [
            t[i+1]**4, t[i+1]**3, t[i+1]**2, t[i+1], 1, 
            -t[i+1]**4, -t[i+1]**3, -t[i+1]**2, -t[i+1], -1
        ]
        constraint_idx += 1

    # First derivative continuity (7)
    for i in range(n - 1):
        contract1_start_idx = 5 * i
        contract2_end_idx = 5 * (i + 2) # spline i and i+1

        A[constraint_idx, contract1_start_idx:contract2_end_idx] = [
            4 * t[i+1]**3, 3 * t[i+1]**2, 2 * t[i+1], 1, 0,
            -4 * t[i+1]**3, -3 * t[i+1]**2, -2 * t[i+1], -1, 0
        ]
        constraint_idx += 1

    # Second derivative continuity (8)
    for i in range(n - 1):
        contract1_start_idx = 5 * i
        contract2_end_idx = 5 * (i + 2) # spline i and i+1

        A[constraint_idx, contract1_start_idx:contract2_end_idx] = [
            12 * t[i+1]**2, 6 * t[i+1], 2, 0, 0,
            -12 * t[i+1]**2, -6 * t[i+1], -2, 0, 0
        ]
        constraint_idx += 1

    # Natural boundary conditions (9)
    A[constraint_idx, -5:] = [4 * t[-1]**3, 3 * t[-1]**2, 2 * t[-1], 1, 0] # 1st derivative = 0 for last spline
    constraint_idx += 1

    # Market price constraints (10)
    for i in range(m):
        T_s, T_e = t[i], t[i + 1]  # Settlement period
        A[constraint_idx, 5 * i:5 * (i + 1)] = [
            (T_e**5 - T_s**5) / 5,
            (T_e**4 - T_s**4) / 4,
            (T_e**3 - T_s**3) / 3,
            (T_e**2 - T_s**2) / 2,
            (T_e - T_s),
        ]

        integral_s = np.sum(s_t[(s_t.index >= T_s) & (s_t.index < T_e)])
        # F[i] and integral_s are both average hourly prices, so we multiply by the length of the contract to get the total price
        b[constraint_idx] = (F[i] + integral_s) * (T_e - T_s)
        constraint_idx += 1

    return A, b

def solve_linear_system(H, A, b):
    """
    Solves the linear system:
        [2H  A^T] [x]   = [0]
        [A   0 ] [λ]     [b]
    
    Parameters:
        H (ndarray): Symmetric matrix H (quadratic term).
        A (ndarray): Restriction matrix.
        b (ndarray): Right-hand side vector of constraints.
        
    Returns:
        tuple: (x, λ) where
            x (ndarray): Solution vector for the main variable.
            λ (ndarray): Lagrange multipliers for constraints.
    """
    # Ensure H is symmetric
    assert H.shape[0] == H.shape[1], "H must be a square matrix"

    # Dimensions
    n = H.shape[0]  # Size of H
    m = A.shape[0]  # Number of constraints

    # Build the full linear system
    K = np.block([
        [2 * H, A.T],
        [A, np.zeros((m, m))]
    ])
    rhs = np.concatenate((np.zeros(n), b))

    # Solve the system using a solver for linear equations
    solution = np.linalg.solve(K, rhs)

    # Extract x and λ
    x = solution[:n]
    lam = solution[n:]

    return x, lam
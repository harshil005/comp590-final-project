import pandas as pd
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata

def calculate_historical_volatility(historical_data: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Calculates the annualized historical volatility.
    """
    log_returns = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
    daily_volatility = log_returns.rolling(window=window).std()
    annualized_volatility = daily_volatility * np.sqrt(window)
    return annualized_volatility

def prepare_volatility_surface_data(options_chain):
    """
    Prepares the data for the 3D volatility surface plot.
    Extracts strike price, time to expiration, and implied volatility.
    Returns both raw scattered points and interpolated mesh grid.
    """
    strikes = []
    times = []
    ivs = []
    
    today = datetime.now()

    for expiration_str, chain in options_chain.items():
        expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d')
        days_to_expiry = (expiration_date - today).days
        
        # Filter expired or immediate expirations to avoid division by zero or logic errors
        if days_to_expiry < 1:
            continue
        
        # Combine calls and puts, and drop contracts with no IV
        df = pd.concat([chain.calls, chain.puts])
        df = df[df['impliedVolatility'] > 0]

        for index, row in df.iterrows():
            strikes.append(row['strike'])
            times.append(days_to_expiry)
            ivs.append(row['impliedVolatility'])
            
    # If no data, return empty structure
    if not strikes:
        return {}

    # --- Interpolation (The Mesh) ---
    # Create a uniform grid
    # 100j means 100 points in that dimension
    grid_x, grid_y = np.mgrid[
        min(strikes):max(strikes):100j, 
        min(times):max(times):100j
    ]

    # Interpolate unstructured data onto the grid using cubic interpolation
    grid_z = griddata(
        (strikes, times), 
        ivs, 
        (grid_x, grid_y), 
        method='cubic'
    )

    # Return data ready for Plotly
    # We return lists because they are JSON serializable
    return {
        "raw_x": strikes,
        "raw_y": times,
        "raw_z": ivs,
        "mesh_x": grid_x.tolist(),
        "mesh_y": grid_y.tolist(),
        "mesh_z": np.nan_to_num(grid_z).tolist() # Handle NaNs from interpolation (outside convex hull)
    }


import sys
from unittest.mock import MagicMock

# --- MONKEY PATCH START ---
# Hack for Python 3.12 compatibility with py_lets_be_rational
# This mock must be injected BEFORE py_vollib or py_lets_be_rational is imported
if "_testcapi" not in sys.modules:
    mock_testcapi = MagicMock()
    # Define the constants py_lets_be_rational looks for
    mock_testcapi.DBL_MAX = 1.7976931348623157e+308
    mock_testcapi.DBL_MIN = 2.2250738585072014e-308
    sys.modules["_testcapi"] = mock_testcapi
# --- MONKEY PATCH END ---

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata
from py_vollib_vectorized import get_all_greeks

def calculate_historical_volatility(historical_data: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Calculates the annualized historical volatility.
    """
    log_returns = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
    daily_volatility = log_returns.rolling(window=window).std()
    annualized_volatility = daily_volatility * np.sqrt(window)
    return annualized_volatility

def prepare_volatility_surface_data(options_chain, underlying_price=None):
    """
    Prepares the data for the 3D volatility surface plot and Greek calculations.
    Extracts strike price, time to expiration, and implied volatility.
    Returns both raw scattered points, interpolated mesh grid, and detailed Greek data.
    """
    strikes = []
    times = []
    ivs = []
    greeks_data = []
    
    today = datetime.now()
    risk_free_rate = 0.045 # Approximate risk free rate (4.5%)

    for expiration_str, chain in options_chain.items():
        expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d')
        days_to_expiry = (expiration_date - today).days
        
        # Filter expired or immediate expirations to avoid division by zero or logic errors
        if days_to_expiry < 1:
            continue
        
        # Time in years for BSM
        T = days_to_expiry / 365.0
        
        # Combine calls and puts for surface, but keep track of type for Greeks
        # We'll iterate calls and puts separately to assign correct flags
        
        for opt_type, df in [('call', chain.calls), ('put', chain.puts)]:
            if df.empty: continue
            
            if 'impliedVolatility' in df.columns:
                df = df[df['impliedVolatility'] > 0]
            
            # --- VECTORIZED CALCULATION START ---
            if underlying_price and not df.empty:
                flag = 'c' if opt_type == 'call' else 'p'
                
                # py_vollib_vectorized expects arrays/Series
                # We ensure inputs are float/numeric series
                try:
                    # Convert Pandas Series to numpy array with explicit float type
                    # get_all_greeks expects (flag, S, K, t, r, sigma) - checking docs, parameter names are usually just positional or sigma, not iv
                    greeks_df = get_all_greeks(
                        flag, 
                        float(underlying_price), 
                        df['strike'].values, 
                        T, 
                        risk_free_rate, 
                        df['impliedVolatility'].values,
                        model='black_scholes'
                    )
                    
                    # --- FINITE DIFFERENCE FOR VANNA & CHARM ---
                    # Vanna: dDelta/dSigma
                    bump_vol = 0.01
                    greeks_vol_plus = get_all_greeks(
                        flag, 
                        float(underlying_price), 
                        df['strike'].values, 
                        T, 
                        risk_free_rate, 
                        df['impliedVolatility'].values + bump_vol,
                        model='black_scholes'
                    )
                    # Vanna = (Delta_new - Delta_old) / bump_vol (Sensitivity to 1 unit change in vol, usually 1%)
                    # Standard definition is dDelta/dSigma (unitless vol). So we divide by bump_vol.
                    greeks_df['vanna'] = (greeks_vol_plus['delta'] - greeks_df['delta']) / bump_vol

                    # Charm: -dDelta/dTime (Decay per day)
                    # We calculate Delta at T - 1/365
                    bump_time = 1.0/365.0
                    T_new = np.maximum(1e-5, T - bump_time) # Ensure positive time
                    
                    greeks_time_minus = get_all_greeks(
                        flag, 
                        float(underlying_price), 
                        df['strike'].values, 
                        T_new, 
                        risk_free_rate, 
                        df['impliedVolatility'].values,
                        model='black_scholes'
                    )
                    
                    # Charm is rate of change per unit time (year). 
                    # Often traders want decay per day. Let's provide decay per day: Delta(today) - Delta(tomorrow)
                    # Delta(T) - Delta(T-dt)
                    greeks_df['charm'] = greeks_df['delta'] - greeks_time_minus['delta']
                    
                    # The result is a DataFrame with columns: delta, gamma, theta, vega, rho, etc.
                    # We need to merge this back with our iteration logic or just append it
                    
                    # Let's attach metadata
                    greeks_df['strike'] = df['strike']
                    greeks_df['iv'] = df['impliedVolatility']
                    greeks_df['expiry'] = expiration_str
                    greeks_df['days_to_expiry'] = days_to_expiry
                    greeks_df['type'] = opt_type
                    
                    # Attach Open Interest and Volume if available, else 0
                    greeks_df['openInterest'] = df['openInterest'] if 'openInterest' in df.columns else 0
                    greeks_df['volume'] = df['volume'] if 'volume' in df.columns else 0
                    
                    # Convert to list of dicts for JSON response
                    # We use 'records' orient
                    # Fill NaNs with 0 to ensure JSON compliance
                    current_records = greeks_df[['strike', 'expiry', 'days_to_expiry', 'type', 'iv', 'delta', 'gamma', 'theta', 'vega', 'vanna', 'charm', 'openInterest', 'volume']].fillna(0).to_dict('records')
                    greeks_data.extend(current_records)
                    
                    # Also collect points for the 3D Surface (just raw x, y, z)
                    strikes.extend(df['strike'].tolist())
                    times.extend([days_to_expiry] * len(df))
                    # Handle potential NaNs in IV as well just in case
                    ivs.extend(df['impliedVolatility'].fillna(0).tolist())
                    
                except Exception as e:
                    print(f"Error calculating greeks for {expiration_str}: {e}")
                    # Fallback: If vectorization fails, we skip adding to greeks_data but maybe still add to surface?
                    # For now, let's just continue
                    pass
            else:
                # Fallback if no underlying price (just show surface points, no Greeks)
                strikes.extend(df['strike'].tolist())
                times.extend([days_to_expiry] * len(df))
                ivs.extend(df['impliedVolatility'].fillna(0).tolist())
            # --- VECTORIZED CALCULATION END ---

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
    
    # Clip interpolated values to be non-negative (IV cannot be negative)
    grid_z = np.maximum(grid_z, 0)

    # Return data ready for Plotly
    # We return lists because they are JSON serializable
    return {
        "raw_x": strikes,
        "raw_y": times,
        "raw_z": ivs,
        "mesh_x": grid_x.tolist(),
        "mesh_y": grid_y.tolist(),
        "mesh_z": np.nan_to_num(grid_z).tolist(), # Handle NaNs from interpolation
        "greeks": greeks_data
    }

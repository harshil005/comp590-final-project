# external
import pandas as pd
import yfinance as yf

# internal


def get_ticker_data(ticker_symbol: str) -> yf.Ticker:
    """
    Creates a yfinance Ticker object for the given symbol.
    
    This is a lightweight operation that doesn't fetch data immediately.
    The ticker object is used later to fetch options chains and historical data
    on-demand, which is more efficient than pre-loading all data.
    
    Args:
        ticker_symbol: Stock ticker symbol (e.g., 'SPY', 'AAPL')
        
    Returns:
        yf.Ticker object ready for data fetching operations
    """
    ticker = yf.Ticker(ticker_symbol)
    return ticker

def clean_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes invalid option contracts that would cause calculation errors.
    
    Filters out contracts with zero bid (no liquidity) and invalid implied volatility
    values. These often appear in yfinance data for illiquid strikes and would
    produce incorrect Greeks calculations or visualization artifacts.
    
    Args:
        df: Raw options DataFrame from yfinance
        
    Returns:
        Cleaned DataFrame with only valid, tradeable contracts
    """
    if df.empty:
        return df
        
    # Zero bid indicates no market maker willing to buy - contract is effectively dead
    if 'bid' in df.columns:
        df = df[df['bid'] > 0]
        
    # Negative or zero IV is mathematically invalid and breaks Black-Scholes calculations
    # yfinance sometimes returns -1 as a placeholder for missing data
    if 'impliedVolatility' in df.columns:
         df = df[df['impliedVolatility'] > 0]
         
    return df

def get_options_chain(ticker: yf.Ticker) -> dict:
    """
    Fetches and cleans options chains for all available expiration dates.
    
    Iterates through all expirations and cleans the data to remove invalid contracts.
    Uses a simple wrapper class to maintain the calls/puts structure expected by
    downstream consumers while allowing cleaned data to be stored.
    
    Args:
        ticker: yfinance Ticker object for the underlying asset
        
    Returns:
        Dictionary mapping expiration dates (YYYY-MM-DD) to OptionChain objects
        containing cleaned calls and puts DataFrames
    """
    # yfinance provides expiration dates as a tuple
    expirations = ticker.options
    
    chain = {}
    for expiration in expirations:
        try:
            options = ticker.option_chain(expiration)
            
            # Remove invalid contracts that would break calculations
            cleaned_calls = clean_options_data(options.calls)
            cleaned_puts = clean_options_data(options.puts)
            
            # Wrap in a simple class to maintain the calls/puts interface
            # yfinance returns NamedTuples which are immutable, so we need a wrapper
            class OptionChain:
                def __init__(self, calls, puts):
                    self.calls = calls
                    self.puts = puts
            
            chain[expiration] = OptionChain(cleaned_calls, cleaned_puts)
            
        except Exception as e:
            # Skip failed expirations rather than failing entire request
            # Some expirations may be unavailable due to data issues
            print(f"Error fetching chain for {expiration}: {e}")
            continue
        
    return chain

def get_historical_data(ticker: yf.Ticker, period: str = "1y") -> pd.DataFrame:
    """
    Fetches historical price data for volatility calculations.
    
    Used primarily for calculating historical volatility (HV) to compare
    against implied volatility (IV). The default 1-year period provides
    sufficient data points for rolling volatility calculations while avoiding
    excessive API calls.
    
    Args:
        ticker: yfinance Ticker object
        period: Time period string (e.g., "1y", "6mo", "90d")
        
    Returns:
        DataFrame with OHLCV data indexed by date
    """
    return ticker.history(period=period)


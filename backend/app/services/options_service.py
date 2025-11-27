import yfinance as yf
import pandas as pd

def get_ticker_data(ticker_symbol: str):
    """
    Fetches the ticker object from yfinance.
    """
    ticker = yf.Ticker(ticker_symbol)
    return ticker

def clean_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the options dataframe by removing invalid contracts.
    - Removes contracts with bid == 0 (no liquidity)
    - Ensures impliedVolatility is valid
    """
    if df.empty:
        return df
        
    # Filter out contracts with no market (Bid == 0)
    if 'bid' in df.columns:
        df = df[df['bid'] > 0]
        
    # Filter out contracts with invalid IV (often -1 or extremely high/low artifacts)
    if 'impliedVolatility' in df.columns:
         df = df[df['impliedVolatility'] > 0]
         
    return df

def get_options_chain(ticker: yf.Ticker):
    """
    Fetches the full options chain for a given ticker.
    """
    # yfinance returns a tuple of expiration dates
    expirations = ticker.options
    
    chain = {}
    for expiration in expirations:
        try:
            options = ticker.option_chain(expiration)
            
            # Clean the data
            cleaned_calls = clean_options_data(options.calls)
            cleaned_puts = clean_options_data(options.puts)
            
            # Assign back (using a simple object or named tuple wrapper might be better, 
            # but for now we just update the objects if they are mutable, 
            # or we can just construct a simple object since we consume it in calculation_service)
            
            # yfinance returns a NamedTuple, so we can't modify it in place easily. 
            # We'll store it in a dictionary or custom class.
            # However, existing consumers expect the named tuple structure (calls, puts).
            # Let's create a simple class to mimic the structure or just overwrite the attributes if possible (unlikely).
            # Better: create a named tuple replacement or just a class.
            
            class OptionChain:
                def __init__(self, calls, puts):
                    self.calls = calls
                    self.puts = puts
            
            chain[expiration] = OptionChain(cleaned_calls, cleaned_puts)
            
        except Exception as e:
            print(f"Error fetching chain for {expiration}: {e}")
            continue
        
    return chain

def get_historical_data(ticker: yf.Ticker, period: str = "1y"):
    """
    Fetches historical stock data for a given period.
    """
    return ticker.history(period=period)


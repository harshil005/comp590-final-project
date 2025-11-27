from fastapi import FastAPI
from .services import options_service, calculation_service
import pandas as pd

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/v1/ticker/{ticker_symbol}/volatility-surface")
def get_volatility_surface(ticker_symbol: str):
    """
    Endpoint to get the data for the volatility surface plot.
    """
    ticker = options_service.get_ticker_data(ticker_symbol)
    options_chain = options_service.get_options_chain(ticker)
    surface_data = calculation_service.prepare_volatility_surface_data(options_chain)
    return surface_data

@app.get("/api/v1/ticker/{ticker_symbol}/open-interest")
def get_open_interest(ticker_symbol: str):
    """
    Endpoint to get open interest and volume by strike for the nearest expiration.
    """
    ticker = options_service.get_ticker_data(ticker_symbol)
    
    # Get the first expiration date
    expirations = ticker.options
    if not expirations:
        return {"error": "No options expirations found for this ticker."}
    
    nearest_expiration = expirations[0]
    options = ticker.option_chain(nearest_expiration)
    
    calls = options.calls[['strike', 'openInterest', 'volume']].rename(
        columns={'openInterest': 'callOpenInterest', 'volume': 'callVolume'}
    )
    puts = options.puts[['strike', 'openInterest', 'volume']].rename(
        columns={'openInterest': 'putOpenInterest', 'volume': 'putVolume'}
    )
    
    merged_data = pd.merge(calls, puts, on="strike", how="outer").fillna(0)
    
    return merged_data.to_dict(orient='records')

@app.get("/api/v1/ticker/{ticker_symbol}/iv-hv-chart")
def get_iv_hv_chart(ticker_symbol: str):
    """
    Endpoint to get data for the IV vs HV chart.
    """
    ticker = options_service.get_ticker_data(ticker_symbol)
    
    # Calculate Historical Volatility
    hist_data = options_service.get_historical_data(ticker)
    hv = calculation_service.calculate_historical_volatility(hist_data)
    
    # Get current average Implied Volatility from the nearest options chain
    expirations = ticker.options
    if not expirations:
        avg_iv = None
    else:
        nearest_expiration = expirations[0]
        options = ticker.option_chain(nearest_expiration)
        df = pd.concat([options.calls, options.puts])
        avg_iv = df['impliedVolatility'].mean()

    # Prepare data for response
    hv_data = hv.dropna().reset_index()
    hv_data.columns = ['date', 'hv']
    hv_data['date'] = hv_data['date'].dt.strftime('%Y-%m-%d')
    
    return {
        "historicalVolatility": hv_data.to_dict(orient='records'),
        "impliedVolatility": avg_iv
    }

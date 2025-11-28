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
    
    # Need current price for Greeks calculation
    try:
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
    except:
        current_price = None
        
    options_chain = options_service.get_options_chain(ticker)
    surface_data = calculation_service.prepare_volatility_surface_data(options_chain, current_price)
    return surface_data

@app.get("/api/v1/ticker/{ticker_symbol}/open-interest")
def get_open_interest(ticker_symbol: str):
    """
    Endpoint to get open interest and volume by strike for the nearest expiration.
    Also returns derived metrics (PCR, Avg IV) for the Market Summary dashboard.
    """
    ticker = options_service.get_ticker_data(ticker_symbol)
    
    # Get the first expiration date
    expirations = ticker.options
    if not expirations:
        return {"error": "No options expirations found for this ticker."}
    
    nearest_expiration = expirations[0]
    options = ticker.option_chain(nearest_expiration)
    
    # 1. Create DataFrame for OI Walls
    calls = options.calls[['strike', 'openInterest', 'volume', 'impliedVolatility']].rename(
        columns={'openInterest': 'callOpenInterest', 'volume': 'callVolume'}
    )
    puts = options.puts[['strike', 'openInterest', 'volume', 'impliedVolatility']].rename(
        columns={'openInterest': 'putOpenInterest', 'volume': 'putVolume'}
    )
    
    merged_data = pd.merge(calls, puts, on="strike", how="outer").fillna(0)
    
    # 2. Derived Metrics for "Market Summary"
    total_call_vol = merged_data['callVolume'].sum()
    total_put_vol = merged_data['putVolume'].sum()
    
    pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 0
    
    # Calculate Average IV (weighted by volume usually better, but simple mean is fine for now)
    # We combine both IV columns
    all_ivs = pd.concat([calls['impliedVolatility'], puts['impliedVolatility']])
    # Filter out bad data
    all_ivs = all_ivs[all_ivs > 0]
    avg_iv = all_ivs.mean() if not all_ivs.empty else 0
    
    summary = {
        "putCallRatio": round(pcr, 2),
        "averageIV": round(avg_iv, 4),
        "totalVolume": int(total_call_vol + total_put_vol),
        "nearestExpiration": nearest_expiration
    }
    
    return {
        "data": merged_data.to_dict(orient='records'),
        "summary": summary
    }

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

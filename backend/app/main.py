# external
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Optional

# internal
from .models import (
    HistoricalPriceItem,
    HistoricalPriceResponse,
    HistoricalVolatilityItem,
    IVHVChartResponse,
    OpenInterestDataItem,
    OpenInterestResponse,
    Summary,
    VolatilitySurfaceResponse,
)
from .services import calculation_service, options_service

app: FastAPI = FastAPI()


@app.get("/")
def read_root() -> dict:
    return {"Hello": "World"}


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/ticker/{ticker_symbol}/volatility-surface", response_model=VolatilitySurfaceResponse)
def get_volatility_surface(ticker_symbol: str) -> VolatilitySurfaceResponse:
    """
    Returns volatility surface data with interpolated mesh and Greeks.
    
    Fetches all available option chains and calculates implied volatility
    across strikes and expirations. Also computes Greeks for risk analysis.
    The current spot price is needed for accurate Greeks calculations.
    
    Args:
        ticker_symbol: Stock ticker symbol
        
    Returns:
        VolatilitySurfaceResponse with raw data, mesh grids, and Greeks
    """
    ticker = options_service.get_ticker_data(ticker_symbol)
    
    # Greeks require current price for Black-Scholes calculations
    # If unavailable, surface data can still be generated without Greeks
    try:
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
    except:
        current_price = None
        
    options_chain = options_service.get_options_chain(ticker)
    surface_data = calculation_service.prepare_volatility_surface_data(options_chain, current_price)
    
    if not surface_data:
        raise HTTPException(status_code=404, detail="No volatility surface data available for this ticker")
    
    return VolatilitySurfaceResponse(**surface_data)


@app.get("/api/v1/ticker/{ticker_symbol}/open-interest", response_model=OpenInterestResponse)
def get_open_interest(ticker_symbol: str, expiration_date: Optional[str] = None) -> OpenInterestResponse:
    """
    Returns open interest and volume data with market summary metrics.
    
    Aggregates call and put open interest/volume by strike to identify
    support/resistance levels (put/call walls). Also calculates Put/Call
    Ratio and average IV as market sentiment indicators.
    
    Args:
        ticker_symbol: Stock ticker symbol
        expiration_date: Optional specific expiration (defaults to nearest)
        
    Returns:
        OpenInterestResponse with strike-level data and summary metrics
    """
    ticker = options_service.get_ticker_data(ticker_symbol)
    
    # Get the first expiration date
    expirations = ticker.options
    if not expirations:
        raise HTTPException(status_code=404, detail="No options expirations found for this ticker")
    
    target_expiration = expiration_date if expiration_date in expirations else expirations[0]
    options = ticker.option_chain(target_expiration)
    
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
    
    summary = Summary(
        putCallRatio=round(pcr, 2),
        averageIV=round(avg_iv, 4),
        totalVolume=int(total_call_vol + total_put_vol),
        targetExpiration=target_expiration,
        availableExpirations=expirations
    )
    
    data_items = [OpenInterestDataItem(**item) for item in merged_data.to_dict(orient='records')]
    
    return OpenInterestResponse(data=data_items, summary=summary)


@app.get("/api/v1/ticker/{ticker_symbol}/iv-hv-chart", response_model=IVHVChartResponse)
def get_iv_hv_chart(ticker_symbol: str) -> IVHVChartResponse:
    """
    Returns historical volatility time series and current implied volatility.
    
    Compares realized volatility (HV) against market expectations (IV) to
    identify when options are relatively cheap or expensive. High IV relative
    to HV suggests options are overpriced, while low IV suggests bargains.
    
    Args:
        ticker_symbol: Stock ticker symbol
        
    Returns:
        IVHVChartResponse with HV time series and current IV value
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
    
    hv_items = [HistoricalVolatilityItem(**item) for item in hv_data.to_dict(orient='records')]
    
    return IVHVChartResponse(historicalVolatility=hv_items, impliedVolatility=avg_iv)


@app.get("/api/v1/ticker/{ticker_symbol}/historical-price", response_model=HistoricalPriceResponse)
def get_historical_price(ticker_symbol: str) -> HistoricalPriceResponse:
    """
    Returns recent historical price data for price action visualization.
    
    Provides 90 days of closing prices to overlay with options-derived
    support/resistance levels. This timeframe balances detail with
    performance and covers most relevant trading periods.
    
    Args:
        ticker_symbol: Stock ticker symbol
        
    Returns:
        HistoricalPriceResponse with date and price pairs
    """
    ticker = options_service.get_ticker_data(ticker_symbol)
    hist_data = ticker.history(period="90d")
    
    if hist_data.empty:
        raise HTTPException(status_code=404, detail="Could not fetch historical price data")
        
    hist_data = hist_data[['Close']].reset_index()
    hist_data.columns = ['date', 'price']
    hist_data['date'] = hist_data['date'].dt.strftime('%Y-%m-%d')
    
    price_items = [HistoricalPriceItem(**item) for item in hist_data.to_dict(orient='records')]
    
    return HistoricalPriceResponse(data=price_items)

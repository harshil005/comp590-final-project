# external
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# internal


class ErrorResponse(BaseModel):
    detail: str = Field(description="Error message")


class Summary(BaseModel):
    putCallRatio: float = Field(description="Put/Call volume ratio")
    averageIV: float = Field(description="Average implied volatility")
    totalVolume: int = Field(description="Total volume across all contracts")
    targetExpiration: str = Field(description="Target expiration date used for analysis")
    availableExpirations: List[str] = Field(description="List of available expiration dates")


class OpenInterestDataItem(BaseModel):
    strike: float = Field(description="Strike price")
    callOpenInterest: float = Field(description="Call open interest at this strike")
    putOpenInterest: float = Field(description="Put open interest at this strike")
    callVolume: float = Field(description="Call volume at this strike")
    putVolume: float = Field(description="Put volume at this strike")


class OpenInterestResponse(BaseModel):
    data: List[OpenInterestDataItem] = Field(description="Open interest data by strike")
    summary: Summary = Field(description="Market summary metrics")


class VolatilitySurfaceResponse(BaseModel):
    raw_x: List[float] = Field(description="Raw strike prices")
    raw_y: List[float] = Field(description="Raw days to expiry")
    raw_z: List[float] = Field(description="Raw implied volatility values")
    mesh_x: List[List[float]] = Field(description="Interpolated strike price grid")
    mesh_y: List[List[float]] = Field(description="Interpolated days to expiry grid")
    mesh_z: List[List[float]] = Field(description="Interpolated implied volatility grid")
    greeks: List[Dict] = Field(description="Greeks data for all options contracts")


class HistoricalVolatilityItem(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    hv: float = Field(description="Historical volatility value")


class IVHVChartResponse(BaseModel):
    historicalVolatility: List[HistoricalVolatilityItem] = Field(description="Historical volatility time series")
    impliedVolatility: Optional[float] = Field(description="Current average implied volatility", default=None)


class HistoricalPriceItem(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    price: float = Field(description="Closing price")


class HistoricalPriceResponse(BaseModel):
    data: List[HistoricalPriceItem] = Field(description="Historical price data")


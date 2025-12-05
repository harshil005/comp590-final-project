# built-in
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

# external
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import norm

# internal

# --- COLOR SCHEME HELPER FUNCTIONS ---
def get_call_color(is_dark_mode: bool, is_color_blind: bool) -> str:
    """Returns call color based on mode and color blind preference."""
    if is_color_blind:
        # Blue for color blind mode - very bright for visibility
        return '#42A5F5' if is_dark_mode else '#1976D2'  # Very bright blue
    else:
        # Very bright green for maximum visibility in dark mode
        return '#66BB6A' if is_dark_mode else '#2E7D32'  # Very bright green

def get_put_color(is_dark_mode: bool, is_color_blind: bool) -> str:
    """Returns put color based on mode and color blind preference."""
    if is_color_blind:
        # Orange for color blind mode - very bright for visibility
        return '#FFB74D' if is_dark_mode else '#F57C00'  # Very bright orange
    else:
        # Very bright red for maximum visibility in dark mode
        return '#EF5350' if is_dark_mode else '#C62828'  # Very bright red

def get_heatmap_colorscale(is_color_blind: bool) -> str:
    """Returns color blind friendly colorscale for heatmaps."""
    if is_color_blind:
        # Blue to orange scale for color blind mode
        return 'Blues'  # Can be customized further
    else:
        # Traditional red scale (but we'll use a darker variant)
        return 'Reds'

# Legacy color constants (kept for backward compatibility, but should use functions above)
CALL_COLOR_LIGHT = '#1B5E20'  # Very dark green for better contrast
CALL_COLOR_DARK = '#1B5E20'   # Very dark green for better contrast
PUT_COLOR_LIGHT = '#B71C1C'   # Very dark red for better contrast
PUT_COLOR_DARK = '#B71C1C'    # Very dark red for better contrast

# --- TEXT AND UI COLOR CONSTANTS ---
# Text colors for dark/light mode
TEXT_COLOR_DARK = '#FFFFFF'
TEXT_COLOR_LIGHT = '#000000'
AXIS_TITLE_COLOR_DARK = '#FFFFFF'
AXIS_TITLE_COLOR_LIGHT = '#000000'
TICK_COLOR_DARK = '#CCCCCC'
TICK_COLOR_LIGHT = '#333333'
GRID_COLOR_DARK = 'rgba(128,128,128,0.2)'
GRID_COLOR_LIGHT = 'rgba(128,128,128,0.1)'
LEGEND_COLOR_DARK = '#FFFFFF'
LEGEND_COLOR_LIGHT = '#000000'

def get_chart_theme_colors(is_dark_mode: bool) -> dict:
    """
    Returns a dictionary of all theme colors for chart configuration.
    
    Args:
        is_dark_mode: Whether dark mode is enabled
        
    Returns:
        Dictionary with all theme color values
    """
    return {
        'text_color': TEXT_COLOR_DARK if is_dark_mode else TEXT_COLOR_LIGHT,
        'axis_title_color': AXIS_TITLE_COLOR_DARK if is_dark_mode else AXIS_TITLE_COLOR_LIGHT,
        'tick_color': TICK_COLOR_DARK if is_dark_mode else TICK_COLOR_LIGHT,
        'grid_color': GRID_COLOR_DARK if is_dark_mode else GRID_COLOR_LIGHT,
        'legend_color': LEGEND_COLOR_DARK if is_dark_mode else LEGEND_COLOR_LIGHT,
        'plot_bg_color': 'rgba(0,0,0,0)' if is_dark_mode else 'rgba(255,255,255,0)',
        'paper_bg_color': 'rgba(0,0,0,0)' if is_dark_mode else 'rgba(255,255,255,0)'
    }

# --- HELPER FUNCTIONS ---
def create_sync_script(chart_ids: list, sync_key: str = 'x') -> str:
    """
    Creates JavaScript code to synchronize hover/selection events across multiple Plotly charts.
    Uses a simpler approach that finds charts by their order in the DOM.
    
    Args:
        chart_ids: List of chart keys to synchronize (used for identification)
        sync_key: The key to match on ('x' for strike price/date, 'pointNumber' for index)
        
    Returns:
        HTML string with JavaScript synchronization code
    """
    num_charts = len(chart_ids)
    return f"""
    <script>
    (function() {{
        // Wait for Plotly to be available
        function initSync() {{
            if (typeof window.Plotly === 'undefined') {{
                setTimeout(initSync, 100);
                return;
            }}
            
            const syncKey = '{sync_key}';
            let charts = [];
            let initialized = false;
            
            // Function to find point index by x-value
            function findPointByX(trace, xValue) {{
                if (!trace || !trace.x) return null;
                
                for (let i = 0; i < trace.x.length; i++) {{
                    const x = trace.x[i];
                    if (typeof x === 'number' && typeof xValue === 'number') {{
                        // For numeric values, allow small tolerance
                        if (Math.abs(x - xValue) < 0.01) return i;
                    }} else if (String(x) === String(xValue)) {{
                        return i;
                    }}
                }}
                return null;
            }}
            
            // Function to synchronize hover across charts
            function syncHover(sourceIndex, hoverData) {{
                if (!hoverData || !hoverData.points || hoverData.points.length === 0) return;
                
                const sourcePoint = hoverData.points[0];
                const syncValue = sourcePoint[syncKey];
                
                // Sync to all other charts
                charts.forEach((gd, index) => {{
                    if (index === sourceIndex || !gd || !gd.data || gd.data.length === 0) return;
                    
                    // Try all traces to find matching point
                    for (let traceIdx = 0; traceIdx < gd.data.length; traceIdx++) {{
                        const trace = gd.data[traceIdx];
                        let pointIndex = null;
                        
                        if (syncKey === 'x') {{
                            pointIndex = findPointByX(trace, syncValue);
                        }} else if (syncKey === 'pointNumber') {{
                            pointIndex = sourcePoint.pointNumber;
                        }}
                        
                        if (pointIndex !== null && pointIndex < trace.x.length) {{
                            try {{
                                Plotly.Fx.hover(gd, {{
                                    points: [{{
                                        curveNumber: traceIdx,
                                        pointNumber: pointIndex,
                                        x: trace.x[pointIndex],
                                        y: trace.y ? trace.y[pointIndex] : null
                                    }}]
                                }});
                                break; // Found and hovered, move to next chart
                            }} catch(e) {{
                                // Continue to next trace
                            }}
                        }}
                    }}
                }});
            }}
            
            // Find and register charts
            function registerCharts() {{
                // Find all Plotly charts in the document
                const allCharts = Array.from(document.querySelectorAll('.js-plotly-plot'));
                
                // Filter to only charts that are fully initialized
                const readyCharts = allCharts.filter(gd => gd && gd._fullLayout && gd.data && gd.data.length > 0);
                
                // If we have the expected number of charts and they're different from before
                if (readyCharts.length >= {num_charts} && (charts.length !== readyCharts.length || !charts.every((c, i) => c === readyCharts[i]))) {{
                    charts = readyCharts.slice(0, {num_charts});
                    
                    // Remove old event listeners and add new ones
                    charts.forEach((gd, index) => {{
                        // Remove existing listeners by cloning (Plotly doesn't have removeEventListener)
                        if (gd._hoverListeners) {{
                            gd.removeAllListeners('plotly_hover');
                            gd.removeAllListeners('plotly_click');
                        }}
                        
                        // Add new listeners
                        gd.on('plotly_hover', function(data) {{
                            syncHover(index, data);
                        }});
                        
                        gd.on('plotly_click', function(data) {{
                            syncHover(index, data);
                        }});
                    }});
                    
                    initialized = true;
                }}
            }}
            
            // Register charts with multiple attempts
            setTimeout(registerCharts, 300);
            setTimeout(registerCharts, 800);
            setTimeout(registerCharts, 1500);
            setTimeout(registerCharts, 2500);
            
            // Watch for new charts being added
            const observer = new MutationObserver(function() {{
                if (!initialized || charts.length < {num_charts}) {{
                    registerCharts();
                }}
            }});
            
            observer.observe(document.body, {{ 
                childList: true, 
                subtree: true,
                attributes: true,
                attributeFilter: ['class']
            }});
        }}
        
        // Start initialization
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initSync);
        }} else {{
            initSync();
        }}
    }})();
    </script>
    """

def calculate_gex_profile(df_greeks: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """
    Calculates Gamma Exposure (GEX) Profile per strike.
    
    GEX measures the notional dollar exposure that market makers must hedge
    as the underlying price moves. This creates support/resistance levels
    because dealers buy on dips (positive GEX) and sell on rallies (negative GEX).
    
    The calculation uses: GEX = Gamma * Open Interest * 100 * Spot^2 * 0.01
    where 0.01 normalizes for a 1% move, and 100 is the contract multiplier.
    
    Args:
        df_greeks: DataFrame containing Greeks data with columns: type, gamma, openInterest, strike
        spot_price: Current spot price of the underlying asset
        
    Returns:
        DataFrame with columns: strike, gex (aggregated GEX per strike)
    """
    if df_greeks.empty or spot_price is None:
        return pd.DataFrame()
        
    # Coerce to numeric to handle any string or mixed-type columns from API
    df = df_greeks.copy()
    df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce').fillna(0)
    df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').fillna(0)
    
    # Normalize GEX calculation for a 1% move to make values comparable across strikes
    # The spot_price^2 term accounts for the non-linear relationship between price and gamma
    const_factor = 100 * (spot_price**2) * 0.01
    
    # Calls create positive GEX (dealers buy stock to hedge, creating support)
    df.loc[df['type'] == 'call', 'gex'] = df['gamma'] * df['openInterest'] * const_factor
    
    # Puts create negative GEX (dealers sell stock to hedge, creating resistance)
    df.loc[df['type'] == 'put', 'gex'] = df['gamma'] * df['openInterest'] * const_factor * -1
    
    # Aggregate by strike to find net GEX at each price level
    gex_profile = df.groupby('strike')['gex'].sum().reset_index()
    return gex_profile

def calculate_zero_gamma(df_greeks: pd.DataFrame, current_spot: float) -> Optional[float]:
    """
    Estimates the Zero Gamma Flip Level by finding where cumulative GEX crosses zero.
    
    The zero gamma level is critical because it's where market maker hedging behavior
    flips from buying (support) to selling (resistance) or vice versa. This creates
    a significant price level that often acts as a magnet or pivot point.
    
    Args:
        df_greeks: DataFrame containing Greeks data
        current_spot: Current spot price of the underlying asset
        
    Returns:
        Estimated zero gamma price level, or None if cannot be determined
    """
    if df_greeks.empty or current_spot is None:
        return None
        
    gex_profile = calculate_gex_profile(df_greeks, current_spot)
    if gex_profile.empty:
        return None
        
    # Sort by strike to enable cumulative calculation
    gex_profile = gex_profile.sort_values('strike')
    
    # Cumulative GEX shows net hedging pressure as price moves up
    # When this crosses zero, dealer behavior flips
    gex_profile['cum_gex'] = gex_profile['gex'].cumsum()
    
    # Find the strike where cumulative GEX crosses zero
    # This indicates where market maker hedging switches from buying to selling
    flip_price = None
    for i in range(1, len(gex_profile)):
        prev_gex = gex_profile.iloc[i-1]['cum_gex']
        curr_gex = gex_profile.iloc[i]['cum_gex']
        
        # Detect sign change indicating zero crossing
        if (prev_gex > 0 and curr_gex < 0) or (prev_gex < 0 and curr_gex > 0):
            # Interpolate between strikes to find approximate zero level
            strike_prev = gex_profile.iloc[i-1]['strike']
            strike_curr = gex_profile.iloc[i]['strike']
            
            # Simple linear interpolation (could be improved with weighted average)
            flip_price = (strike_prev + strike_curr) / 2
            break
            
    return flip_price

def generate_probability_cone_data(spot_price: float, current_iv: float, start_date: datetime, days: int = 60) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates probability cone data showing expected price range based on implied volatility.
    
    The cone uses the Black-Scholes assumption that price movements follow a log-normal
    distribution. The 1SD and 2SD bands represent 68% and 95% probability ranges respectively.
    This helps traders visualize where the stock is likely to trade over time.
    
    Args:
        spot_price: Current spot price to anchor the cone
        current_iv: Current implied volatility (annualized)
        start_date: Starting date for the cone projection
        days: Number of days to project forward (default 60)
        
    Returns:
        Tuple of (dates, upper_1sd, lower_1sd, upper_2sd, lower_2sd) arrays
    """
    days_future = np.arange(1, days)
    
    # Anchor cone to current price to show forward projection from known point
    cone_anchor_price = spot_price

    # Calculate 1 standard deviation bands (68% probability range)
    # Volatility scales with square root of time (annualized IV converted to daily)
    upper_1sd = cone_anchor_price * (1 + current_iv * np.sqrt(days_future/365))
    lower_1sd = cone_anchor_price * (1 - current_iv * np.sqrt(days_future/365))
    
    # Calculate 2 standard deviation bands (95% probability range)
    upper_2sd = cone_anchor_price * (1 + 2 * current_iv * np.sqrt(days_future/365))
    lower_2sd = cone_anchor_price * (1 - 2 * current_iv * np.sqrt(days_future/365))
    
    # Generate future dates starting from the day after the anchor date
    dates_future = [start_date + timedelta(days=int(d)) for d in days_future]
    
    # Prepend anchor point to ensure visual continuity with historical price chart
    dates_future.insert(0, start_date)
    upper_1sd = np.insert(upper_1sd, 0, cone_anchor_price)
    lower_1sd = np.insert(lower_1sd, 0, cone_anchor_price)
    upper_2sd = np.insert(upper_2sd, 0, cone_anchor_price)
    lower_2sd = np.insert(lower_2sd, 0, cone_anchor_price)

    return dates_future, upper_1sd, lower_1sd, upper_2sd, lower_2sd

# Set page config first - this must be the first Streamlit command
st.set_page_config(page_title="Options Viz", layout="wide")

# Get API URL
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

# Display title immediately
st.title("Stock Options Visualization Engine")

# --- CSS for Production Polish (Dark Mode & Alignment) ---
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

def apply_theme():
    if st.session_state["dark_mode"]:
        st.markdown("""
            <style>
            /* Force Pure Black Background */
            .stApp {
                background-color: #000000 !important;
                color: #FFFFFF !important;
            }
            /* Fix Sidebar contrast in dark mode */
            section[data-testid="stSidebar"] {
                background-color: #111111 !important;
            }
            /* Metric Cards Polish */
            div[data-testid="stMetricValue"] {
                color: #4CAF50 !important; /* Green for values */
            }
            /* Custom Container Borders */
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
                border-color: #333333;
            }
            /* Fix all text colors in dark mode */
            .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #FFFFFF !important;
            }
            /* Fix selectbox and radio button labels */
            label[data-testid="stWidgetLabel"], .stSelectbox label, .stRadio label, .stCheckbox label, .stSlider label {
                color: #FFFFFF !important;
            }
            /* Fix caption text */
            .stCaption {
                color: #CCCCCC !important;
            }
            /* Fix all widget labels */
            div[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] {
                color: #FFFFFF !important;
            }
            /* Fix text input labels */
            label[for*="ticker"], label {
                color: #FFFFFF !important;
            }
            /* Fix expander headers */
            .streamlit-expanderHeader, .streamlit-expanderHeader p {
                color: #FFFFFF !important;
            }
            /* Fix expander content - CRITICAL for Visualization Settings visibility */
            [data-testid="stExpander"] p {
                color: #FFFFFF !important;
            }
            [data-testid="stExpander"] .st-emotion-cache-fqgod8 p,
            [data-testid="stExpander"] .st-emotion-cache-1sh7rz9 p,
            [data-testid="stExpander"] .st-emotion-cache-1n8dvl8 p,
            [data-testid="stExpander"] .st-emotion-cache-ai037n p {
                color: #FFFFFF !important;
            }
            /* Fix container text */
            [data-testid="stVerticalBlock"], [data-testid="stVerticalBlock"] p, [data-testid="stVerticalBlock"] div {
                color: #FFFFFF !important;
            }
            /* Fix all p tags and divs in sidebar */
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] div,
            section[data-testid="stSidebar"] span {
                color: #FFFFFF !important;
            }
            /* Fix header text */
            h1, h2, h3, h4, h5, h6 {
                color: #FFFFFF !important;
            }
            /* Fix selectbox options text */
            .stSelectbox > div > div, .stRadio > div > label {
                color: #FFFFFF !important;
            }
            /* Fix selectbox dropdown text - critical for visibility */
            [data-baseweb="select"] {
                color: #FFFFFF !important;
            }
            [data-baseweb="select"] > div {
                color: #FFFFFF !important;
            }
            /* Fix dropdown options */
            ul[role="listbox"] li, [data-baseweb="popover"] {
                color: #000000 !important;
                background-color: #FFFFFF !important;
            }
            [data-baseweb="popover"] li {
                color: #000000 !important;
            }
            /* Fix sidebar button text */
            .stButton > button {
                color: #FFFFFF !important;
            }
            /* Fix sidebar header */
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
                color: #FFFFFF !important;
            }
            /* Fix expander text in sidebar */
            [data-testid="stSidebar"] .streamlit-expanderHeader {
                color: #FFFFFF !important;
            }
            /* Fix metric labels */
            [data-testid="stMetricLabel"] {
                color: #FFFFFF !important;
            }
            /* Fix delta text in metrics */
            [data-testid="stMetricDelta"] {
                color: #FFFFFF !important;
            }
            /* Fix radio button text */
            .stRadio label {
                color: #FFFFFF !important;
            }
            /* Fix selectbox selected text */
            .stSelectbox [data-baseweb="select"] {
                color: #FFFFFF !important;
            }
            /* Fix all text in main content */
            main .block-container {
                color: #FFFFFF !important;
            }
            /* Fix header text */
            header h1, .stApp > header, h1, [data-testid="stHeader"] {
                color: #FFFFFF !important;
            }
            /* Fix main title */
            .stApp h1 {
                color: #FFFFFF !important;
            }
            /* Fix sidebar button */
            section[data-testid="stSidebar"] button {
                color: #FFFFFF !important;
                background-color: #1f77b4 !important;
            }
            section[data-testid="stSidebar"] button:hover {
                background-color: #1565c0 !important;
            }
            /* Fix sidebar expander text */
            section[data-testid="stSidebar"] .streamlit-expanderHeader {
                color: #FFFFFF !important;
            }
            section[data-testid="stSidebar"] .streamlit-expanderHeader p {
                color: #FFFFFF !important;
            }
            section[data-testid="stSidebar"] .streamlit-expanderContent {
                color: #FFFFFF !important;
            }
            section[data-testid="stSidebar"] .streamlit-expanderContent p {
                color: #FFFFFF !important;
            }
            /* Fix all text in sidebar expander - comprehensive */
            section[data-testid="stSidebar"] [data-testid="stExpander"] {
                color: #FFFFFF !important;
            }
            section[data-testid="stSidebar"] [data-testid="stExpander"] * {
                color: #FFFFFF !important;
            }
            /* Fix button text */
            button p, button span, button div {
                color: #FFFFFF !important;
            }
            section[data-testid="stSidebar"] button {
                color: #FFFFFF !important;
            }
            section[data-testid="stSidebar"] button * {
                color: #FFFFFF !important;
            }
            /* Force all sidebar text white */
            section[data-testid="stSidebar"] {
                color: #FFFFFF !important;
            }
            section[data-testid="stSidebar"] * {
                color: #FFFFFF !important;
            }
            /* Fix selectbox value display */
            [data-baseweb="select"] [aria-live] {
                color: #FFFFFF !important;
            }
            /* Fix selectbox placeholder and selected value */
            [data-baseweb="select"] > div:first-child {
                color: #FFFFFF !important;
            }
            </style>
        """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", help="Stock ticker symbol to analyze (e.g., SPY, AAPL, TSLA)").upper()

# Sidebar Visualization Settings (New)
if "color_blind_mode" not in st.session_state:
    st.session_state["color_blind_mode"] = False

with st.sidebar.expander("Visualization Settings", expanded=False):
    st.session_state["dark_mode"] = st.checkbox("Dark Mode (Pure Black)", value=st.session_state["dark_mode"])
    st.session_state["color_blind_mode"] = st.checkbox("Color Blind Mode (Blue/Orange)", value=st.session_state["color_blind_mode"], help="Uses blue/orange instead of green/red for better color blind accessibility")
    
    st.caption("Heatmap Outlier Handling")
    percentile_cap = st.slider("Color Cap (Percentile)", 50, 100, 99, help="Cap colors at this percentile to ignore extreme outliers.")

apply_theme() # Apply CSS based on state

# Helper function for tooltips - comprehensive vocabulary from removed sidebar
def get_tooltip(term: str) -> str:
    """Returns tooltip text for various terms and metrics."""
    tooltips = {
        # Market Activity Metrics
        "Put/Call Ratio": "Ratio of put volume to call volume. High (>1.0) = bearish, Low (<0.8) = bullish sentiment.",
        "Average IV": "Market's expectation of future price volatility. High IV = expensive options, Low IV = cheap options.",
        "Total Volume": "Total number of option contracts traded. Shows overall market activity and liquidity.",
        "Open Interest": "Total number of option contracts currently 'open' (not yet closed or expired). Think of it as the size of the 'wall' at each strike price. High OI creates support/resistance levels where the stock price often gets 'pinned' near expiration.",
        "Volume": "Number of contracts traded during a specific time period. Unlike OI, volume resets each period and shows current trading activity. High volume indicates where 'smart money' is placing new bets right now.",
        
        # Basic Options Concepts
        "Strike Price": "The predetermined price at which you can buy (call) or sell (put) the underlying stock. It's like a 'target price' you're betting the stock will reach.",
        "Expiration Date": "The date when an option contract expires and becomes worthless if not exercised. After this date, the option no longer exists.",
        "Call Option": "The right (not obligation) to BUY a stock at the strike price. You buy calls when you think the stock will go UP. Profit when stock price > strike price + premium paid.",
        "Put Option": "The right (not obligation) to SELL a stock at the strike price. You buy puts when you think the stock will go DOWN. Profit when stock price < strike price - premium paid.",
        
        # Volatility Metrics
        "Implied Volatility": "The market's expectation of how much a stock's price will fluctuate in the future. High IV = expensive options (high fear/uncertainty). Low IV = cheap options (low expected movement).",
        "IV": "Implied Volatility - The market's expectation of future price volatility. High IV = expensive options, Low IV = cheap options.",
        
        # The Greeks
        "Delta": "How much an option's price changes for a $1 move in the stock price. Calls: 0 to 1 (ATM calls ≈ 0.50). Puts: -1 to 0 (ATM puts ≈ -0.50). Example: Delta of 0.75 means option gains $0.75 for every $1 stock move up.",
        "Gamma": "The rate of change of Delta. Measures how quickly your position's sensitivity changes. High Gamma = Delta changes rapidly = more risk (profits/losses accelerate quickly). Highest at-the-money.",
        "Theta": "Time decay - how much value an option loses EVERY DAY just from the passage of time. Always negative for long positions. Options lose value as expiration approaches, even if stock price doesn't move.",
        
        # Market Structure Concepts
        "Support Level": "A strike price with extremely high put open interest (Put Wall). Acts like a floor - the stock price often bounces up from here because market makers hedge by buying stock at these levels.",
        "Resistance Level": "A strike price with extremely high call open interest (Call Wall). Acts like a ceiling - the stock price often gets pushed down from here because market makers hedge by selling stock at these levels.",
        "Resistance": "A price level where selling pressure is strong, preventing the stock from rising further. In options, this is often the Call Wall - a strike with high call open interest.",
        "Put Wall": "A strike price with extremely high put open interest. Acts like a floor - the stock price often bounces up from here because market makers hedge by buying stock at these levels.",
        "Call Wall": "A strike price with extremely high call open interest. Acts like a ceiling - the stock price often gets pushed down from here because market makers hedge by selling stock at these levels.",
        "Gamma Exposure": "Measures how much market makers need to hedge as stock price changes. High positive GEX = market makers buy stock on dips, creating support. High negative GEX = market makers sell stock on rallies, creating resistance.",
        "GEX": "Gamma Exposure - Measures how much market makers need to hedge as stock price changes. High positive GEX = support, High negative GEX = resistance.",
        "Liquidity Walls": "Concentrations of open interest at specific strike prices that create barriers. The stock often struggles to move through these levels, especially near expiration.",
        "Max Pain": "The strike price where the most options expire worthless, causing maximum losses to option buyers. Stock prices often gravitate toward max pain near expiration.",
        
        # UI Elements
        "Start Expiration": "First expiration date in the range to analyze for the 2D volatility heatmap.",
        "End Expiration": "Last expiration date in the range to analyze for the 2D volatility heatmap.",
        "Start Date": "First date in the range for detailed open interest and volume analysis.",
        "End Date": "Last date in the range for detailed open interest and volume analysis.",
        "X-Axis": "Choose whether to display data grouped by Strike Price or Expiration Date on the horizontal axis.",
        "Option Type": "Select whether to view Greeks (Delta, Gamma, Theta) for Call options or Put options.",
        "Selected Date": "Specific expiration date to view detailed strike-level distribution of open interest and volume."
    }
    return tooltips.get(term, "")

# Legacy function for backward compatibility
def get_metric_tooltip(metric_name: str) -> str:
    """Returns tooltip text for metric labels."""
    return get_tooltip(metric_name)

# Initialize Session State for Data Persistence
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'oi_data' not in st.session_state:
    st.session_state['oi_data'] = None
if 'summary' not in st.session_state:
    st.session_state['summary'] = None
if 'spy_summary' not in st.session_state:
    st.session_state['spy_summary'] = None
if 'historical_price' not in st.session_state:
    st.session_state['historical_price'] = None

analyze_btn = st.sidebar.button("Analyze")

if analyze_btn:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # Fetch Volatility Surface & Greeks
            response = requests.get(f"{API_URL}/ticker/{ticker}/volatility-surface")
            if response.status_code == 200:
                st.session_state['data'] = response.json()
            else:
                st.error(f"Failed to fetch volatility data. Status: {response.status_code}")
            
            # Fetch Open Interest & Summary
            resp_oi = requests.get(f"{API_URL}/ticker/{ticker}/open-interest")
            if resp_oi.status_code == 200:
                oi_json = resp_oi.json()
                st.session_state['oi_data'] = pd.DataFrame(oi_json['data'])
                st.session_state['summary'] = oi_json.get('summary')

            # Always fetch SPY comparison data
            resp_spy = requests.get(f"{API_URL}/ticker/SPY/open-interest")
            if resp_spy.status_code == 200:
                st.session_state['spy_summary'] = resp_spy.json().get('summary')
            else:
                st.session_state['spy_summary'] = None

            # Fetch Historical Price Data
            resp_hist = requests.get(f"{API_URL}/ticker/{ticker}/historical-price")
            if resp_hist.status_code == 200:
                hist_json = resp_hist.json()
                st.session_state['historical_price'] = pd.DataFrame(hist_json.get('data', hist_json))
                
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# --- MAIN CONTENT ---
if st.session_state['data']:
    data = st.session_state['data']
    is_dark_mode = st.session_state["dark_mode"]
    chart_template = "plotly_dark" if is_dark_mode else "plotly"
    
    # Theme-aware colors with color blind support
    is_color_blind = st.session_state.get("color_blind_mode", False)
    call_color = get_call_color(is_dark_mode, is_color_blind)
    put_color = get_put_color(is_dark_mode, is_color_blind)
    heatmap_colorscale = get_heatmap_colorscale(is_color_blind)
    
    # Get all theme colors from centralized function
    theme_colors = get_chart_theme_colors(is_dark_mode)
    text_color = theme_colors['text_color']
    axis_title_color = theme_colors['axis_title_color']
    tick_color = theme_colors['tick_color']
    grid_color = theme_colors['grid_color']
    legend_color = theme_colors['legend_color']
    plot_bg_color = theme_colors['plot_bg_color']
    paper_bg_color = theme_colors['paper_bg_color']
    
    # --- MAIN DASHBOARD ---
    # --- Top Row Metrics: Comparative Market Pulse ---
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown(f"**Market Pulse: {ticker}**")
            if st.session_state['summary']:
                summ = st.session_state['summary']
                m1, m2, m3 = st.columns(3)
                pcr = summ['putCallRatio']
                m1.metric("Put/Call Ratio", pcr, delta="Bearish" if pcr > 1.0 else ("Bullish" if pcr < 0.7 else "Neutral"), delta_color="inverse", help=get_metric_tooltip("Put/Call Ratio"))
                avg_iv = summ['averageIV']
                m2.metric("Average IV", f"{avg_iv:.2%}", help=get_metric_tooltip("Average IV"))
                vol = summ['totalVolume']
                m3.metric("Total Volume", f"{vol:,}", help=get_metric_tooltip("Total Volume"))
    
    with c2:
        with st.container(border=True):
            st.markdown("**Market Pulse: SPY**")
            if st.session_state['spy_summary']:
                spy_summ = st.session_state['spy_summary']
                m1, m2, m3 = st.columns(3)
                spy_pcr = spy_summ['putCallRatio']
                m1.metric("Put/Call Ratio", spy_pcr, delta="Bearish" if spy_pcr > 1.0 else ("Bullish" if spy_pcr < 0.7 else "Neutral"), delta_color="inverse", help=get_metric_tooltip("Put/Call Ratio"))
                spy_avg_iv = spy_summ['averageIV']
                m2.metric("Average IV", f"{spy_avg_iv:.2%}", help=get_metric_tooltip("Average IV"))
                spy_vol = spy_summ['totalVolume']
                m3.metric("Total Volume", f"{spy_vol:,}", help=get_metric_tooltip("Total Volume"))
            else:
                st.info("SPY comparison data not available")
    
    # --- Main Chart: Price Action with Support & Resistance ---
    with st.container(border=True):
            st.markdown("#### Price Action vs. Support & Resistance Levels")
            with st.expander("How to Use This Chart", expanded=False):
                st.markdown("""
                This chart shows the stock's price action with key support and resistance levels derived from options market data.
                
                **Support (Put Wall):** The strike price with the highest put open interest, often acting as a floor.
                - **How it works:** Market makers selling puts must buy stock to hedge. High put OI creates buying pressure that supports the price.
                - **Example Scenario:** If stock is trading at `$150` and there's massive put OI at `$145`, expect the stock to bounce off `$145` as dealers hedge.
                
                **Resistance (Call Wall):** The strike price with the highest call open interest, often acting as a ceiling.
                - **How it works:** Market makers selling calls must sell stock to hedge. High call OI creates selling pressure that resists upward moves.
                - **Example Scenario:** If stock rallies toward `$160` where massive call OI exists, expect selling pressure as dealers hedge, causing the rally to stall.
                
                **Trading Application:**
                - Price bouncing off support = potential long entry
                - Price failing at resistance = potential short entry
                - Price breaking through walls = strong momentum, possible continuation
                """)
            
            if "greeks" in data and data["greeks"] and st.session_state['historical_price'] is not None:
                df_greeks = pd.DataFrame(data["greeks"])
                hist_df = st.session_state['historical_price']
                
                # Ensure date column is in datetime format
                hist_df['date'] = pd.to_datetime(hist_df['date'])
                
                # Calculate Support & Resistance from GEX Profile
                spot_price = float(data['raw_x'][0]) if data['raw_x'] else 400.0
                gex_profile = calculate_gex_profile(df_greeks, spot_price)
                
                if not gex_profile.empty:
                    call_wall_idx = gex_profile['gex'].idxmax()
                    put_wall_idx = gex_profile['gex'].idxmin()
                    call_wall = gex_profile.loc[call_wall_idx, 'strike']
                    put_wall = gex_profile.loc[put_wall_idx, 'strike']
                else:
                    call_wall, put_wall = None, None
                
                fig_map = go.Figure()

                # Price line color adapts to theme for visibility
                price_line_color = '#FFFFFF' if is_dark_mode else '#1f77b4'
                fig_map.add_trace(go.Scatter(
                    x=hist_df['date'], 
                    y=hist_df['price'], 
                    mode='lines', 
                    name='Historical Price', 
                    line=dict(color=price_line_color, width=2),
                    hovertemplate='Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
                ))

                # Support and resistance lines use theme-aware colors
                if call_wall is not None:
                    fig_map.add_hline(y=call_wall, line_dash="dot", line_color=put_color, annotation_text=f"Resistance (Call Wall): ${call_wall:,.0f}")
                if put_wall is not None:
                    fig_map.add_hline(y=put_wall, line_dash="dot", line_color=call_color, annotation_text=f"Support (Put Wall): ${put_wall:,.0f}")
                
                fig_map.update_layout(
                    title="Historical Price vs. Options-Derived Support & Resistance",
                    template=chart_template,
                    hovermode="x unified",
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    plot_bgcolor=plot_bg_color,
                    paper_bgcolor=paper_bg_color,
                    font=dict(color=text_color),
                    title_font=dict(color=text_color),
                    xaxis=dict(
                        title_font=dict(color=axis_title_color),
                        tickfont=dict(color=tick_color)
                    ),
                    yaxis=dict(
                        title_font=dict(color=axis_title_color),
                        tickfont=dict(color=tick_color)
                    ),
                    legend=dict(font=dict(color=legend_color))
                )
                st.plotly_chart(fig_map, use_container_width=True)
    
    # --- Volatility Analysis Section ---
    with st.container(border=True):
        st.markdown("#### Volatility Analysis")
        with st.expander("How to Use This Chart", expanded=False):
            st.markdown("""
            Volatility analysis helps you identify where options are cheap or expensive relative to historical patterns and market expectations.
            
            **3D Volatility Surface:**
            - **What it shows:** A three-dimensional view of implied volatility across all strike prices and expiration dates
            - **Red peaks = High IV:** Options are expensive (high fear/uncertainty)
            - **Valleys = Low IV:** Options are relatively cheap
            - **Example Scenario:** If you see a peak at `$150` strike with 30-day expiration but a valley at `$155` strike with 60-day expiration, those `$155` strike options with 60-day expiration are cheaper - good for buyers but poor premium for sellers.
            
            **2D Volatility Heatmap:**
            - **What it shows:** A color-coded view of implied volatility for calls and puts across selected expiration dates
            - **Hot colors (red/orange):** High implied volatility = expensive options
            - **Cool colors (blue/green):** Low implied volatility = cheaper options
            - **Example Scenario:** If calls show high IV at ATM strikes but puts show low IV, it suggests bullish sentiment - traders are paying more for upside protection. This could indicate an overbought condition.
            
            **Trading Applications:**
            - **Buying Opportunities:** Look for valleys in the surface where IV is low - you can buy options cheaper
            - **Selling Opportunities:** Look for peaks where IV is high - you can collect more premium selling options
            - **Skew Analysis:** Compare call IV vs put IV - significant differences indicate directional bias in the market
            - **Term Structure:** Compare IV across expirations - steep slopes indicate time-based volatility expectations
            """)
        
        if "greeks" in data and data["greeks"]:
            df_greeks = pd.DataFrame(data["greeks"])
            spot_price = float(data['raw_x'][0]) if data['raw_x'] else 400.0
            expirations = sorted(df_greeks['expiry'].unique())
            
            # 3D Volatility Surface
            if "mesh_z" in data and data["mesh_z"]:
                st.markdown("**3D Volatility Surface**")
                # Convert mesh_z from decimal to percentage (multiply by 100)
                mesh_z_percent = np.array(data['mesh_z']) * 100
                
                fig_surf = go.Figure(data=[go.Surface(
                    z=mesh_z_percent.tolist(), 
                    x=data['mesh_x'], 
                    y=data['mesh_y'], 
                    colorscale=heatmap_colorscale, 
                    cmin=0, 
                    opacity=0.9, 
                    colorbar=dict(
                        title=dict(text='Implied Volatility (%)', font=dict(color=axis_title_color)),
                        tickfont=dict(color=tick_color),
                        tickformat='.0f',
                        ticksuffix='%'
                    ), 
                    lighting=dict(ambient=0.5, diffuse=0.5),
                    hovertemplate='Strike: $%{x:.2f}<br>Days to Expiry: %{y:.1f}<br>IV: %{z:.1f}%<extra></extra>'
                )])
                fig_surf.update_layout(
                    title=dict(
                        text='3D Volatility Surface: Find Cheap Options in Valleys',
                        font=dict(color=text_color)
                    ),
                    template=chart_template,
                    height=600,
                    plot_bgcolor=plot_bg_color,
                    paper_bgcolor=paper_bg_color,
                    font=dict(color=text_color),
                    scene=dict(
                        bgcolor=plot_bg_color,
                        xaxis=dict(
                            title=dict(
                                text='Strike Price ($)',
                                font=dict(color=axis_title_color)
                            ),
                            tickfont=dict(color=tick_color),
                            gridcolor=grid_color
                        ),
                        yaxis=dict(
                            title=dict(
                                text='Days to Expiry',
                                font=dict(color=axis_title_color)
                            ),
                            tickfont=dict(color=tick_color),
                            gridcolor=grid_color
                        ),
                        zaxis=dict(
                            title=dict(
                                text='Implied Volatility (%)',
                                font=dict(color=axis_title_color)
                            ),
                            tickfont=dict(color=tick_color),
                            gridcolor=grid_color,
                            tickformat='.0f',
                            ticksuffix='%'
                        )
                    )
                )
                st.plotly_chart(fig_surf, use_container_width=True)
            
            # Expiration range selector for 2D heatmap
            st.markdown("---")
            st.markdown("**Configure Range Analysis**")
            col_vol1, col_vol2 = st.columns(2)
            with col_vol1:
                start_exp_idx = st.selectbox("Start Expiration", range(len(expirations)), format_func=lambda x: expirations[x], key="vol_start_exp", index=0, help=get_tooltip("Start Expiration"))
            with col_vol2:
                end_exp_idx = st.selectbox("End Expiration", range(len(expirations)), format_func=lambda x: expirations[x], key="vol_end_exp", index=min(4, len(expirations)-1), help=get_tooltip("End Expiration"))
            
            selected_expirations = expirations[start_exp_idx:end_exp_idx+1]
            filtered_df = df_greeks[df_greeks['expiry'].isin(selected_expirations)].copy()
            
            # 2D Heatmap for selected expiration range
            st.markdown(f"**2D Volatility Heatmap (Range: {selected_expirations[0]} to {selected_expirations[-1]})**")
            if not filtered_df.empty:
                # Create heatmap: strikes on Y-axis, expirations on X-axis, color = IV
                # We'll show Call IV and Put IV separately, or average IV
                all_strikes = sorted(filtered_df['strike'].unique())
                all_expirations = sorted(selected_expirations)
                
                # Create IV matrix: rows = strikes, cols = expirations
                iv_matrix_calls = []
                iv_matrix_puts = []
                
                for strike in all_strikes:
                    row_calls = []
                    row_puts = []
                    for exp in all_expirations:
                        call_data = filtered_df[(filtered_df['strike'] == strike) & (filtered_df['expiry'] == exp) & (filtered_df['type'] == 'call')]
                        put_data = filtered_df[(filtered_df['strike'] == strike) & (filtered_df['expiry'] == exp) & (filtered_df['type'] == 'put')]
                        
                        call_iv = call_data['iv'].values[0] if not call_data.empty and len(call_data['iv'].values) > 0 else None
                        put_iv = put_data['iv'].values[0] if not put_data.empty and len(put_data['iv'].values) > 0 else None
                        
                        # Convert from decimal to percentage (multiply by 100)
                        row_calls.append(call_iv * 100 if call_iv is not None else None)
                        row_puts.append(put_iv * 100 if put_iv is not None else None)
                    
                    iv_matrix_calls.append(row_calls)
                    iv_matrix_puts.append(row_puts)
                
                # Create two heatmaps side by side
                heatmap_col1, heatmap_col2 = st.columns(2)
                
                with heatmap_col1:
                    fig_heatmap_calls = go.Figure(data=go.Heatmap(
                        z=iv_matrix_calls,
                        x=[exp[:10] for exp in all_expirations],  # Shorten date format
                        y=[f"${s:.0f}" for s in all_strikes],
                        colorscale=heatmap_colorscale,
                        colorbar=dict(
                            title=dict(text='Call IV (%)', font=dict(color=axis_title_color)),
                            tickfont=dict(color=tick_color),
                            tickformat='.0f',
                            ticksuffix='%'
                        ),
                        hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>IV: %{z:.1f}%<extra></extra>'
                    ))
                    fig_heatmap_calls.update_layout(
                        title='Call IV Heatmap',
                        template=chart_template,
                        height=600,
                        yaxis_title='Strike Price ($)',
                        xaxis_title='Expiration Date',
                        plot_bgcolor=plot_bg_color,
                        paper_bgcolor=paper_bg_color,
                        font=dict(color=text_color),
                        title_font=dict(color=text_color),
                        xaxis=dict(
                            title_font=dict(color=axis_title_color),
                            tickfont=dict(color=tick_color)
                        ),
                        yaxis=dict(
                            title_font=dict(color=axis_title_color),
                            tickfont=dict(color=tick_color)
                        )
                    )
                    st.plotly_chart(fig_heatmap_calls, use_container_width=True)
                
                with heatmap_col2:
                    fig_heatmap_puts = go.Figure(data=go.Heatmap(
                        z=iv_matrix_puts,
                        x=[exp[:10] for exp in all_expirations],
                        y=[f"${s:.0f}" for s in all_strikes],
                        colorscale=heatmap_colorscale,
                        colorbar=dict(
                            title=dict(text='Put IV (%)', font=dict(color=axis_title_color)),
                            tickfont=dict(color=tick_color),
                            tickformat='.0f',
                            ticksuffix='%'
                        ),
                        hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>IV: %{z:.1f}%<extra></extra>'
                    ))
                    fig_heatmap_puts.update_layout(
                        title='Put IV Heatmap',
                        template=chart_template,
                        height=600,
                        yaxis_title='Strike Price ($)',
                        xaxis_title='Expiration Date',
                        plot_bgcolor=plot_bg_color,
                        paper_bgcolor=paper_bg_color,
                        font=dict(color=text_color),
                        title_font=dict(color=text_color),
                        xaxis=dict(
                            title_font=dict(color=axis_title_color),
                            tickfont=dict(color=tick_color)
                        ),
                        yaxis=dict(
                            title_font=dict(color=axis_title_color),
                            tickfont=dict(color=tick_color)
                        )
                    )
                    st.plotly_chart(fig_heatmap_puts, use_container_width=True)
    
    # --- Liquidity Walls Section ---
    with st.container(border=True):
        st.markdown("#### Liquidity Walls: Open Interest & Volume")
        with st.expander("Understanding Open Interest vs Volume", expanded=False):
            st.markdown("""
            **Open Interest (OI)** and **Volume** are both important metrics, but they tell you different stories:
            
            **Open Interest (OI):**
            - **What it is:** The total number of option contracts that are currently "open" (not yet closed or expired)
            - **Think of it as:** The size of the "wall" - how many contracts are sitting at each strike price
            - **Key insight:** High OI at a strike creates a "magnet" effect - the stock price often gets pinned to that level near expiration
            - **Changes when:** New positions are opened OR existing positions are closed (closing reduces OI)
            - **Use it to:** Identify support/resistance levels, find where "max pain" might occur at expiration
            
            **Volume:**
            - **What it is:** The number of contracts traded during a specific time period (today, this week, etc.)
            - **Think of it as:** The "activity" - how much trading is happening right now
            - **Key insight:** High volume shows where smart money is placing NEW bets or closing positions
            - **Changes every day:** Resets to zero each period - tells you what's happening NOW
            - **Use it to:** Identify unusual activity, find where institutions are positioning, spot potential breakouts
            
            **How They Work Together:**
            - **High OI + High Volume:** Strong conviction - lots of existing positions being actively traded
            - **High OI + Low Volume:** Stale positions - walls are established but not much new activity
            - **Low OI + High Volume:** New positioning - fresh bets being placed that could move the market
            
            **Trading Application:**
            - **Open Interest Walls:** Look for strikes with massive OI - these act as support/resistance
            - **Volume Spikes:** Look for unusual volume - this shows where the "whales" are trading right now
            
            **Example Scenarios:**
            - **Scenario 1:** Stock at `$150`, massive put OI at `$145`, low volume. The `$145` level acts as strong support due to dealer hedging, even without new trading activity.
            - **Scenario 2:** Stock at `$150`, small OI at `$155`, sudden high volume. This indicates new bullish positioning - smart money is betting on a move to `$155+`.
            - **Scenario 3:** Stock approaching `$160` at expiration, with high call OI and increasing volume. Dealers hedge by selling stock, creating resistance that may pin the price below `$160`.
            
            **Key Takeaway:** Combine OI and Volume analysis to identify both established support/resistance levels (OI) and emerging positioning (Volume).
            """)
        
        if "greeks" in data and data["greeks"]:
            df_greeks = pd.DataFrame(data["greeks"])
            expirations = sorted(df_greeks['expiry'].unique())
            
            # Convert expiry strings to datetime for parsing
            df_greeks['expiry_date'] = pd.to_datetime(df_greeks['expiry'])
            
            # Date range selection
            st.markdown("### Select Date Range for Detailed Analysis")
            
            # Get all available dates
            all_dates = sorted(df_greeks['expiry_date'].dt.strftime('%Y-%m-%d').unique())
            
            if len(all_dates) > 1:
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date_idx = st.selectbox("Start Date", range(len(all_dates)), format_func=lambda x: all_dates[x], key="start_date", help=get_tooltip("Start Date"))
                with col_date2:
                    end_date_idx = st.selectbox("End Date", range(len(all_dates)), format_func=lambda x: all_dates[x], key="end_date", index=len(all_dates)-1, help=get_tooltip("End Date"))
                date_range = all_dates[start_date_idx:end_date_idx+1]
            else:
                date_range = all_dates
            
            # Ensure date_range is defined and not empty
            if not date_range:
                date_range = all_dates[:1] if all_dates else []
            
            # Filter df_greeks to selected date range
            df_range = df_greeks[df_greeks['expiry_date'].dt.strftime('%Y-%m-%d').isin(date_range)].copy()
            
            # --- Strike-Level Analysis Section ---
            if date_range and len(date_range) > 0:
                if len(date_range) == 1:
                    st.markdown(f"### Strike-Level Analysis (Date: {date_range[0]})")
                else:
                    st.markdown(f"### Strike-Level Analysis (Range: {date_range[0]} to {date_range[-1]})")
            else:
                st.markdown("### Strike-Level Analysis")
            
            # X-axis toggle for range analysis
            if not df_range.empty and len(date_range) > 1:
                x_axis_option = st.radio(
                    "X-Axis",
                    ["Strike Price", "Expiration Date"],
                    horizontal=True,
                    key="x_axis_range_analysis",
                    index=0,
                    help=get_tooltip("X-Axis")
                )
            else:
                x_axis_option = "Strike Price"
            
            # Aggregate data based on x-axis selection
            if not df_range.empty:
                if x_axis_option == "Strike Price":
                    # Group by strike, sum across all dates
                    calls_agg = df_range[df_range['type'] == 'call'].groupby('strike').agg({
                        'openInterest': 'sum',
                        'volume': 'sum'
                    }).reset_index()
                    puts_agg = df_range[df_range['type'] == 'put'].groupby('strike').agg({
                        'openInterest': 'sum',
                        'volume': 'sum'
                    }).reset_index()
                    
                    # Get all strikes
                    all_x_values = sorted(set(list(calls_agg['strike']) + list(puts_agg['strike'])))
                    
                    # Create mapping for quick lookup
                    calls_oi_dict = dict(zip(calls_agg['strike'], calls_agg['openInterest']))
                    calls_vol_dict = dict(zip(calls_agg['strike'], calls_agg['volume']))
                    puts_oi_dict = dict(zip(puts_agg['strike'], puts_agg['openInterest']))
                    puts_vol_dict = dict(zip(puts_agg['strike'], puts_agg['volume']))
                    
                    call_oi_values = [calls_oi_dict.get(x, 0) for x in all_x_values]
                    put_oi_values = [puts_oi_dict.get(x, 0) for x in all_x_values]
                    call_vol_values = [calls_vol_dict.get(x, 0) for x in all_x_values]
                    put_vol_values = [puts_vol_dict.get(x, 0) for x in all_x_values]
                    
                    x_axis_title = 'Strike Price ($)'
                else:  # Expiration Date
                    # Group by expiration date, sum across all strikes
                    calls_agg = df_range[df_range['type'] == 'call'].groupby('expiry_date').agg({
                        'openInterest': 'sum',
                        'volume': 'sum'
                    }).reset_index()
                    puts_agg = df_range[df_range['type'] == 'put'].groupby('expiry_date').agg({
                        'openInterest': 'sum',
                        'volume': 'sum'
                    }).reset_index()
                    
                    # Get all expiration dates
                    all_x_values = sorted(set(list(calls_agg['expiry_date']) + list(puts_agg['expiry_date'])))
                    
                    # Create mapping for quick lookup
                    calls_oi_dict = dict(zip(calls_agg['expiry_date'], calls_agg['openInterest']))
                    calls_vol_dict = dict(zip(calls_agg['expiry_date'], calls_agg['volume']))
                    puts_oi_dict = dict(zip(puts_agg['expiry_date'], puts_agg['openInterest']))
                    puts_vol_dict = dict(zip(puts_agg['expiry_date'], puts_agg['volume']))
                    
                    call_oi_values = [calls_oi_dict.get(x, 0) for x in all_x_values]
                    put_oi_values = [puts_oi_dict.get(x, 0) for x in all_x_values]
                    call_vol_values = [calls_vol_dict.get(x, 0) for x in all_x_values]
                    put_vol_values = [puts_vol_dict.get(x, 0) for x in all_x_values]
                    
                    x_axis_title = 'Expiration Date'
                    # Convert dates to strings for display
                    all_x_values = [x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x) for x in all_x_values]
                
                # Open Interest Chart (full width)
                st.markdown("**Open Interest**")
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(
                    x=all_x_values,
                    y=call_oi_values,
                    name='Call OI',
                    marker_color=call_color,
                    opacity=1.0
                ))
                fig_oi.add_trace(go.Bar(
                    x=all_x_values,
                    y=[-oi for oi in put_oi_values],  # Negative for below
                    name='Put OI',
                    marker_color=put_color,
                    opacity=1.0
                ))
                fig_oi.update_layout(
                    barmode='overlay',
                    title=f"Open Interest by {x_axis_option}",
                    template=chart_template,
                    height=400,
                    xaxis_title=x_axis_title,
                    yaxis_title='Open Interest (Contracts)',
                    showlegend=True,
                    hovermode='x unified',
                    plot_bgcolor=plot_bg_color,
                    paper_bgcolor=paper_bg_color,
                    font=dict(color=text_color),
                    title_font=dict(color=text_color),
                    xaxis=dict(
                        title_font=dict(color=axis_title_color),
                        tickfont=dict(color=tick_color)
                    ),
                    yaxis=dict(
                        title_font=dict(color=axis_title_color),
                        tickfont=dict(color=tick_color)
                    ),
                    legend=dict(font=dict(color=legend_color))
                )
                if x_axis_option == "Expiration Date":
                    fig_oi.update_layout(xaxis=dict(tickangle=-45))
                st.plotly_chart(fig_oi, use_container_width=True, key=f"oi_range_{x_axis_option}")
                
                # Volume Chart (full width)
                st.markdown("**Volume**")
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=all_x_values,
                    y=call_vol_values,
                    name='Call Volume',
                    marker_color=call_color,
                    opacity=1.0
                ))
                fig_vol.add_trace(go.Bar(
                    x=all_x_values,
                    y=[-vol for vol in put_vol_values],  # Negative for below
                    name='Put Volume',
                    marker_color=put_color,
                    opacity=1.0
                ))
                fig_vol.update_layout(
                    barmode='overlay',
                    title=f"Volume by {x_axis_option}",
                    template=chart_template,
                    height=400,
                    xaxis_title=x_axis_title,
                    yaxis_title='Volume (Contracts)',
                    showlegend=True,
                    hovermode='x unified',
                    plot_bgcolor=plot_bg_color,
                    paper_bgcolor=paper_bg_color,
                    font=dict(color=text_color),
                    title_font=dict(color=text_color),
                    xaxis=dict(
                        title_font=dict(color=axis_title_color),
                        tickfont=dict(color=tick_color)
                    ),
                    yaxis=dict(
                        title_font=dict(color=axis_title_color),
                        tickfont=dict(color=tick_color)
                    ),
                    legend=dict(font=dict(color=legend_color))
                )
                if x_axis_option == "Expiration Date":
                    fig_vol.update_layout(xaxis=dict(tickangle=-45))
                st.plotly_chart(fig_vol, use_container_width=True, key=f"vol_range_{x_axis_option}")
                
                # Add synchronization script for OI and Volume range charts
                sync_ids_oi_vol = [f"oi_range_{x_axis_option}", f"vol_range_{x_axis_option}"]
                components.html(create_sync_script(sync_ids_oi_vol, sync_key='x'), height=0)
            
            # --- Selected Date Drill-Down Section ---
            if not df_range.empty:
                st.markdown("---")
                st.markdown("### Strike-Level Analysis for Selected Date")
                
                sorted_dates = sorted(date_range)
                selected_date = st.selectbox(
                    "Select Expiration Date to View Strike Distribution",
                    sorted_dates,
                    key="selected_exp_date_oi",
                    help=get_tooltip("Selected Date")
                )
                
                # Filter data for selected date
                selected_date_data = df_range[df_range['expiry_date'].dt.strftime('%Y-%m-%d') == selected_date]
                
                if not selected_date_data.empty:
                    # Aggregate OI by strike for the selected date
                    calls_oi_by_strike = selected_date_data[selected_date_data['type'] == 'call'].groupby('strike')['openInterest'].sum()
                    puts_oi_by_strike = selected_date_data[selected_date_data['type'] == 'put'].groupby('strike')['openInterest'].sum()
                    
                    # Get all strikes for this date
                    all_strikes = sorted(set(list(calls_oi_by_strike.index) + list(puts_oi_by_strike.index)))
                    
                    call_oi_values = [calls_oi_by_strike.get(strike, 0) for strike in all_strikes]
                    put_oi_values = [puts_oi_by_strike.get(strike, 0) for strike in all_strikes]
                    
                    # Aggregate Volume by strike for the selected date
                    calls_vol_by_strike = selected_date_data[selected_date_data['type'] == 'call'].groupby('strike')['volume'].sum()
                    puts_vol_by_strike = selected_date_data[selected_date_data['type'] == 'put'].groupby('strike')['volume'].sum()
                    
                    call_vol_values = [calls_vol_by_strike.get(strike, 0) for strike in all_strikes]
                    put_vol_values = [puts_vol_by_strike.get(strike, 0) for strike in all_strikes]
                    
                    # Create two charts side by side
                    col_oi, col_vol = st.columns(2)
                    
                    with col_oi:
                        st.markdown("**Open Interest by Strike**")
                        fig_oi_by_strike = go.Figure()
                        fig_oi_by_strike.add_trace(go.Bar(
                            x=all_strikes,
                            y=call_oi_values,
                            name='Call OI',
                            marker_color=call_color,
                            opacity=1.0
                        ))
                        fig_oi_by_strike.add_trace(go.Bar(
                            x=all_strikes,
                            y=[-oi for oi in put_oi_values],  # Negative for below
                            name='Put OI',
                            marker_color=put_color,
                            opacity=1.0
                        ))
                        fig_oi_by_strike.update_layout(
                            title=f'Open Interest by Strike ({selected_date})',
                            template=chart_template,
                            height=400,
                            xaxis_title='Strike Price ($)',
                            yaxis_title='Open Interest (Contracts)',
                            barmode='overlay',
                            showlegend=True,
                            hovermode='x unified',
                            plot_bgcolor=plot_bg_color,
                            paper_bgcolor=paper_bg_color,
                            font=dict(color=text_color),
                            title_font=dict(color=text_color),
                            xaxis=dict(
                                title_font=dict(color=axis_title_color),
                                tickfont=dict(color=tick_color)
                            ),
                            yaxis=dict(
                                title_font=dict(color=axis_title_color),
                                tickfont=dict(color=tick_color)
                            ),
                            legend=dict(font=dict(color=legend_color))
                        )
                        st.plotly_chart(fig_oi_by_strike, use_container_width=True, key=f"oi_strike_{selected_date}")
                    
                    with col_vol:
                        st.markdown("**Volume by Strike**")
                        fig_vol_by_strike = go.Figure()
                        fig_vol_by_strike.add_trace(go.Bar(
                            x=all_strikes,
                            y=call_vol_values,
                            name='Call Volume',
                            marker_color=call_color,
                            opacity=1.0
                        ))
                        fig_vol_by_strike.add_trace(go.Bar(
                            x=all_strikes,
                            y=[-vol for vol in put_vol_values],  # Negative for below
                            name='Put Volume',
                            marker_color=put_color,
                            opacity=1.0
                        ))
                        fig_vol_by_strike.update_layout(
                            title=f'Volume by Strike ({selected_date})',
                            template=chart_template,
                            height=400,
                            xaxis_title='Strike Price ($)',
                            yaxis_title='Volume (Contracts)',
                            barmode='overlay',
                            showlegend=True,
                            hovermode='x unified',
                            plot_bgcolor=plot_bg_color,
                            paper_bgcolor=paper_bg_color,
                            font=dict(color=text_color),
                            title_font=dict(color=text_color),
                            xaxis=dict(
                                title_font=dict(color=axis_title_color),
                                tickfont=dict(color=tick_color)
                            ),
                            yaxis=dict(
                                title_font=dict(color=axis_title_color),
                                tickfont=dict(color=tick_color)
                            ),
                            legend=dict(font=dict(color=legend_color))
                        )
                        st.plotly_chart(fig_vol_by_strike, use_container_width=True, key=f"vol_strike_{selected_date}")
                    
                    # Add synchronization script for OI and Volume selected date charts
                    sync_ids_oi_vol_strike = [f"oi_strike_{selected_date}", f"vol_strike_{selected_date}"]
                    components.html(create_sync_script(sync_ids_oi_vol_strike, sync_key='x'), height=0)
                    
                    # Add Greeks Slice Analysis
                    st.markdown("---")
                    st.markdown("#### Greeks Slice Analysis")
                    with st.expander("How to Use This Chart", expanded=False):
                        st.markdown("""
                        The Greeks measure the risk of a specific options contract at different strike prices.
                        
                        **Delta:** Measures direction risk - how much the option price changes for each `$1` move in the stock.
                        - **Range:** Calls (0 to 1), Puts (-1 to 0)
                        - **Example Scenario:** A call with Delta `0.75` will gain `$0.75` for every `$1` the stock moves up. If you own `10` contracts, that's `$750` profit per `$1` move.
                        - **What to Look For:** Higher Delta = more stock-like behavior. At-the-money options have Delta around `0.50`.
                        
                        **Gamma:** Measures acceleration risk - how quickly Delta changes as the stock moves.
                        - **Example Scenario:** An option with high Gamma sees its Delta jump from `0.50` to `0.80` after a `$2` stock move, causing the option price to accelerate faster.
                        - **What to Look For:** Peak Gamma at-the-money means Delta changes fastest when stock is near the strike. This creates volatility in option prices.
                        
                        **Theta:** Measures time decay - how much value the option loses each day.
                        - **Example Scenario:** An option with Theta of `-0.10` loses `$0.10` per day. Over `10` days, you lose `$1.00` even if stock doesn't move.
                        - **What to Look For:** Time decay accelerates as expiration approaches. Options with less time to expiration have higher absolute Theta values.
                        - **Trading Application:** Sellers profit from Theta decay, buyers need stock movement to overcome time decay.
                        """)
                    
                    selected_type = st.radio("Option Type", ["call", "put"], horizontal=True, key="greeks_type_selected_date", help=get_tooltip("Option Type"))
                    
                    # Filter data for selected date and type
                    filtered_df = selected_date_data[selected_date_data['type'] == selected_type].sort_values('strike')
                    
                    if not filtered_df.empty:
                        # Store strike prices for synchronization
                        strikes_list = filtered_df['strike'].tolist()
                        
                        col_g1, col_g2, col_g3 = st.columns(3)
                        with col_g1:
                            fig_delta = go.Figure(go.Scatter(
                                x=filtered_df['strike'], 
                                y=filtered_df['delta'], 
                                mode='lines', 
                                name='Delta',
                                hovertemplate='Strike: $%{x:,.0f}<br>Delta: %{y:.4f}<extra></extra>',
                                customdata=strikes_list
                            ))
                            fig_delta.update_layout(
                                title="Delta",
                                template=chart_template,
                                height=350,
                                xaxis_title='Strike Price ($)',
                                yaxis_title='Delta (Price Sensitivity)',
                                plot_bgcolor=plot_bg_color,
                                paper_bgcolor=paper_bg_color,
                                font=dict(color=text_color),
                                title_font=dict(color=text_color),
                                xaxis=dict(
                                    title_font=dict(color=axis_title_color),
                                    tickfont=dict(color=tick_color)
                                ),
                                yaxis=dict(
                                    title_font=dict(color=axis_title_color),
                                    tickfont=dict(color=tick_color)
                                ),
                                legend=dict(font=dict(color=legend_color))
                            )
                            st.plotly_chart(fig_delta, use_container_width=True, key=f"delta_chart_{selected_date}")
                        with col_g2:
                            fig_gamma = go.Figure(go.Scatter(
                                x=filtered_df['strike'], 
                                y=filtered_df['gamma'], 
                                mode='lines', 
                                name='Gamma',
                                hovertemplate='Strike: $%{x:,.0f}<br>Gamma: %{y:.6f}<extra></extra>',
                                customdata=strikes_list
                            ))
                            fig_gamma.update_layout(
                                title="Gamma",
                                template=chart_template,
                                height=350,
                                xaxis_title='Strike Price ($)',
                                yaxis_title='Gamma (Delta Sensitivity)',
                                plot_bgcolor=plot_bg_color,
                                paper_bgcolor=paper_bg_color,
                                font=dict(color=text_color),
                                title_font=dict(color=text_color),
                                xaxis=dict(
                                    title_font=dict(color=axis_title_color),
                                    tickfont=dict(color=tick_color)
                                ),
                                yaxis=dict(
                                    title_font=dict(color=axis_title_color),
                                    tickfont=dict(color=tick_color)
                                ),
                                legend=dict(font=dict(color=legend_color))
                            )
                            st.plotly_chart(fig_gamma, use_container_width=True, key=f"gamma_chart_{selected_date}")
                        with col_g3:
                            fig_theta = go.Figure(go.Bar(
                                x=filtered_df['strike'], 
                                y=filtered_df['theta'], 
                                name='Theta',
                                hovertemplate='Strike: $%{x:,.0f}<br>Theta: $%{y:.4f}/day<extra></extra>',
                                customdata=strikes_list
                            ))
                            fig_theta.update_layout(
                                title="Theta",
                                template=chart_template,
                                height=350,
                                xaxis_title='Strike Price ($)',
                                yaxis_title='Theta ($/day)',
                                plot_bgcolor=plot_bg_color,
                                paper_bgcolor=paper_bg_color,
                                font=dict(color=text_color),
                                title_font=dict(color=text_color),
                                xaxis=dict(
                                    title_font=dict(color=axis_title_color),
                                    tickfont=dict(color=tick_color)
                                ),
                                yaxis=dict(
                                    title_font=dict(color=axis_title_color),
                                    tickfont=dict(color=tick_color)
                                ),
                                legend=dict(font=dict(color=legend_color))
                            )
                            st.plotly_chart(fig_theta, use_container_width=True, key=f"theta_chart_{selected_date}")
                        
                        # Add synchronization script for Greek charts
                        sync_ids = [f"delta_chart_{selected_date}", f"gamma_chart_{selected_date}", f"theta_chart_{selected_date}"]
                        components.html(create_sync_script(sync_ids, sync_key='x'), height=0)

else:
    st.info("Enter a ticker symbol and click 'Analyze' to begin.")

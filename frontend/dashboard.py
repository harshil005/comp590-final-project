# built-in
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

# external
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import norm

# --- LOGGING SETUP ---
# This ensures you see ALL errors in your terminal
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Verify logger is working
logger.info("Application starting... Logging enabled.")

# internal

# --- COLOR SCHEME HELPER FUNCTIONS ---
def get_call_color(is_color_blind: bool) -> str:
    """Returns call color based on color blind preference."""
    if is_color_blind:
        return '#1976D2'  # Blue for color blind mode
    else:
        return '#2E7D32'  # Green

def get_put_color(is_color_blind: bool) -> str:
    """Returns put color based on color blind preference."""
    if is_color_blind:
        return '#F57C00'  # Orange for color blind mode
    else:
        return '#C62828'  # Red

def get_heatmap_colorscale(is_color_blind: bool) -> str:
    """Returns color blind friendly colorscale for heatmaps."""
    if is_color_blind:
        # Blue to orange scale for color blind mode
        return 'Blues'  # Can be customized further
    else:
        # Traditional red scale (but we'll use a darker variant)
        return 'Reds'

# --- TEXT AND UI COLOR CONSTANTS ---
TEXT_COLOR = '#000000'
AXIS_TITLE_COLOR = '#000000'
TICK_COLOR = '#333333'
GRID_COLOR = 'rgba(128,128,128,0.1)'
LEGEND_COLOR = '#000000'

def get_chart_theme_colors() -> dict:
    """
    Returns a dictionary of all theme colors for chart configuration.
    
    Returns:
        Dictionary with all theme color values
    """
    return {
        'text_color': TEXT_COLOR,
        'axis_title_color': AXIS_TITLE_COLOR,
        'tick_color': TICK_COLOR,
        'grid_color': GRID_COLOR,
        'legend_color': LEGEND_COLOR,
        'plot_bg_color': 'rgba(255,255,255,0)',
        'paper_bg_color': 'rgba(255,255,255,0)'
    }

def create_xaxis_sync_script(chart_keys: list) -> str:
    """
    Creates JavaScript to synchronize x-axis zoom/pan between charts.
    Uses Plotly's relayout event to sync xaxis.range.
    
    Args:
        chart_keys: List of Streamlit chart keys to synchronize
        
    Returns:
        HTML string containing JavaScript code
    """
    script = f"""
    <script>
    (function() {{
        const chartKeys = {chart_keys};
        let isSyncing = false;
        
        function waitForCharts() {{
            const charts = [];
            chartKeys.forEach(key => {{
                // Streamlit creates divs with data-testid containing the key
                const div = document.querySelector(`[data-testid*="${{key}}"]`);
                if (div) {{
                    // Find the actual Plotly div (usually a child)
                    const plotlyDiv = div.querySelector('.js-plotly-plot') || div;
                    if (plotlyDiv && plotlyDiv.data) {{
                        charts.push({{key: key, div: plotlyDiv}});
                    }}
                }}
            }});
            
            if (charts.length === chartKeys.length) {{
                setupSync(charts);
            }} else {{
                setTimeout(waitForCharts, 100);
            }}
        }}
        
        function setupSync(charts) {{
            charts.forEach(({{key, div}}) => {{
                div.on('plotly_relayout', function(eventData) {{
                    if (isSyncing) return;
                    
                    // Check if x-axis range changed
                    if (eventData['xaxis.range[0]'] !== undefined && 
                        eventData['xaxis.range[1]'] !== undefined) {{
                        
                        isSyncing = true;
                        const xRange = [
                            eventData['xaxis.range[0]'],
                            eventData['xaxis.range[1]']
                        ];
                        
                        // Sync to all other charts
                        charts.forEach(({{div: targetDiv}}) => {{
                            if (targetDiv !== div && targetDiv.data) {{
                                Plotly.relayout(targetDiv, {{
                                    'xaxis.range': xRange
                                }});
                            }}
                        }});
                        
                        setTimeout(() => {{ isSyncing = false; }}, 50);
                    }}
                }});
            }});
        }}
        
        // Start waiting for charts after a short delay
        setTimeout(waitForCharts, 500);
    }})();
    </script>
    """
    return script


# --- HELPER FUNCTIONS ---
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

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", help="Stock ticker symbol to analyze (e.g., SPY, AAPL, TSLA)").upper()

# Sidebar Visualization Settings
if "color_blind_mode" not in st.session_state:
    st.session_state["color_blind_mode"] = False

with st.sidebar.expander("Visualization Settings", expanded=False):
    st.session_state["color_blind_mode"] = st.checkbox("Color Blind Mode (Blue/Orange)", value=st.session_state["color_blind_mode"], help="Uses blue/orange instead of green/red for better color blind accessibility")
    
    st.caption("Heatmap Outlier Handling")
    percentile_cap = st.slider("Color Cap (Percentile)", 50, 100, 99, help="Cap colors at this percentile to ignore extreme outliers.")

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
            
            # Reset filters when new data is loaded
            if "selected_range_from_state" in st.session_state:
                del st.session_state["selected_range_from_state"]
            if "selected_date_from_chart" in st.session_state:
                del st.session_state["selected_date_from_chart"]
                
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# --- MAIN CONTENT ---
if st.session_state['data']:
    data = st.session_state['data']
    chart_template = "plotly"
    
    # Check for clicked date from URL query params (set by JavaScript click handler)
    query_params = st.query_params
    if "clicked_date" in query_params:
        clicked_date_from_url = query_params["clicked_date"]
        # Store in session state and clear the query param
        st.session_state['selected_date_from_chart'] = clicked_date_from_url
        # Clear the query param to avoid re-triggering
        del st.query_params["clicked_date"]
        st.rerun()
    
    # Theme-aware colors with color blind support
    is_color_blind = st.session_state.get("color_blind_mode", False)
    call_color = get_call_color(is_color_blind)
    put_color = get_put_color(is_color_blind)
    heatmap_colorscale = get_heatmap_colorscale(is_color_blind)
    
    # Get all theme colors from centralized function
    theme_colors = get_chart_theme_colors()
    text_color = theme_colors['text_color']
    axis_title_color = theme_colors['axis_title_color']
    tick_color = theme_colors['tick_color']
    grid_color = theme_colors['grid_color']
    legend_color = theme_colors['legend_color']
    plot_bg_color = theme_colors['plot_bg_color']
    paper_bg_color = theme_colors['paper_bg_color']
    
    # --- MAIN DASHBOARD ---
    
    # Pre-process Greeks data for global use
    df_greeks_global = pd.DataFrame()
    all_dates_str = []
    if "greeks" in data and data["greeks"]:
        df_greeks_global = pd.DataFrame(data["greeks"])
        # Ensure expiry is standardized YYYY-MM-DD
        df_greeks_global['expiry_dt'] = pd.to_datetime(df_greeks_global['expiry'])
        df_greeks_global['expiry_str'] = df_greeks_global['expiry_dt'].dt.strftime('%Y-%m-%d')
        all_dates_str = sorted(df_greeks_global['expiry_str'].unique())

    # Initialize default values (moved to inline controls in respective sections)
    selected_range_start = None
    selected_range_end = None
    selected_date = None
    available_dates_in_range = []

    if all_dates_str:
        # Default to full range if not set
        if "selected_range_from_state" not in st.session_state:
            st.session_state["selected_range_from_state"] = (all_dates_str[0], all_dates_str[-1])

        # Initialize selected_date_from_chart if not set
        if 'selected_date_from_chart' not in st.session_state:
            st.session_state['selected_date_from_chart'] = all_dates_str[0]

        # Set initial values for use in charts
        selected_range_start, selected_range_end = st.session_state.get("selected_range_from_state", (all_dates_str[0], all_dates_str[-1]))
        available_dates_in_range = [d for d in all_dates_str if selected_range_start <= d <= selected_range_end]
        selected_date = st.session_state.get('selected_date_from_chart', available_dates_in_range[0] if available_dates_in_range else None)
    
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
                price_line_color = '#1f77b4'
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
                    title=dict(
                        text="Historical Price vs. Options-Derived Support & Resistance",
                        font=dict(color=text_color, size=20),
                        x=0.5,
                        xanchor='center'
                    ),
                    template=chart_template,
                    hovermode="x unified",
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    plot_bgcolor=plot_bg_color,
                    paper_bgcolor=paper_bg_color,
                    font=dict(color=text_color),
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
        # Title and range slider on same row
        vol_title_col, vol_slider_col = st.columns([1, 2])
        with vol_title_col:
            st.markdown("#### Volatility Analysis")
        with vol_slider_col:
            if all_dates_str:
                # Inline range slider for volatility charts
                select_range = st.select_slider(
                    "Expiration Date Range",
                    options=all_dates_str,
                    value=st.session_state.get("selected_range_from_state", (all_dates_str[0], all_dates_str[-1])),
                    key="expiration_range_slider_inline",
                    help="Filter volatility charts by expiration date range.",
                    label_visibility="collapsed"
                )
                selected_range_start, selected_range_end = select_range
                st.session_state["selected_range_from_state"] = select_range

                # Filter dates available for specific selection based on range
                available_dates_in_range = [d for d in all_dates_str if selected_range_start <= d <= selected_range_end]

                # Update session state
                if 'selected_date_from_chart' not in st.session_state or st.session_state['selected_date_from_chart'] not in available_dates_in_range:
                    st.session_state['selected_date_from_chart'] = available_dates_in_range[0] if available_dates_in_range else None

        with st.expander("How to Use This Chart", expanded=False):
            st.markdown("""
            
            **3D Volatility Surface:**
            - **Red peaks = High IV:** Options are expensive (high fear/uncertainty)
            - **Valleys = Low IV:** Options are relatively cheap
            
            **2D Volatility Heatmap:**
            - **Hot colors (red/orange):** High implied volatility = expensive options
            - **Cool colors (blue/green):** Low implied volatility = cheaper options
            
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
            
            # Create two columns: 3D surface on left (50%), heatmaps on right (50%)
            col_3d, col_heatmaps = st.columns([1, 1])

            # Left column: 3D Volatility Surface
            with col_3d:
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
                            title=dict(text='IV (%)', font=dict(color=axis_title_color, size=9)),
                            tickfont=dict(color=tick_color, size=8),
                            tickformat='.0f',
                            ticksuffix='%',
                            len=0.8
                        ),
                        lighting=dict(ambient=0.5, diffuse=0.5),
                        hovertemplate='Strike: $%{x:.2f}<br>Days to Expiry: %{y:.1f}<br>IV: %{z:.1f}%<extra></extra>'
                    )])
                    fig_surf.update_layout(
                        title=dict(
                            text='3D Implied Volatility Surface',
                            font=dict(color=text_color, size=22),
                            x=0.5,
                            xanchor='center'
                        ),
                        template=chart_template,
                        height=600,
                        plot_bgcolor=plot_bg_color,
                        paper_bgcolor=paper_bg_color,
                        font=dict(color=text_color, size=9),
                        scene=dict(
                            bgcolor=plot_bg_color,
                            xaxis=dict(
                                title=dict(
                                    text='Strike ($)',
                                    font=dict(color=axis_title_color, size=9)
                                ),
                                tickfont=dict(color=tick_color, size=8),
                                gridcolor=grid_color
                            ),
                            yaxis=dict(
                                title=dict(
                                    text='Days',
                                    font=dict(color=axis_title_color, size=9)
                                ),
                                tickfont=dict(color=tick_color, size=8),
                                gridcolor=grid_color
                            ),
                            zaxis=dict(
                                title=dict(
                                    text='IV (%)',
                                    font=dict(color=axis_title_color, size=9)
                                ),
                                tickfont=dict(color=tick_color, size=8),
                                gridcolor=grid_color,
                                tickformat='.0f',
                                ticksuffix='%'
                            )
                        ),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_surf, use_container_width=True)

            # Right column: 2D heatmaps stacked
            with col_heatmaps:
                # Use global filtered data for volatility heatmaps
                if selected_range_start and selected_range_end:
                     df_greeks_vol = df_greeks_global[
                        (df_greeks_global['expiry_str'] >= selected_range_start) &
                        (df_greeks_global['expiry_str'] <= selected_range_end)
                     ].copy()
                else:
                     df_greeks_vol = df_greeks_global.copy()

                selected_expirations = sorted(df_greeks_vol['expiry'].unique())
                filtered_df = df_greeks_vol

                # 2D Heatmap for selected expiration range
                if not filtered_df.empty:
                    st.markdown(f"")
                
                # Optimized heatmap generation using pivot tables (faster than nested loops)
                df_calls = filtered_df[filtered_df['type'] == 'call']
                df_puts = filtered_df[filtered_df['type'] == 'put']
                
                pivot_calls = df_calls.pivot_table(index='strike', columns='expiry', values='iv', aggfunc='mean')
                pivot_puts = df_puts.pivot_table(index='strike', columns='expiry', values='iv', aggfunc='mean')
                
                # Create UNIFIED axes for both heatmaps so hover sync works correctly
                # Union of all strikes and expiries from both calls and puts
                all_strikes_heatmap = sorted(set(pivot_calls.index.tolist() + pivot_puts.index.tolist()))
                all_expiries_heatmap = sorted(set(pivot_calls.columns.tolist() + pivot_puts.columns.tolist()))
                
                # Reindex both pivots to use the unified axes (fills missing with NaN)
                pivot_calls = pivot_calls.reindex(index=all_strikes_heatmap, columns=all_expiries_heatmap)
                pivot_puts = pivot_puts.reindex(index=all_strikes_heatmap, columns=all_expiries_heatmap)
                
                # Extract values for Plotly (convert to % and replace NaN with None for clean hover)
                z_calls = pivot_calls.mul(100).where(~pivot_calls.isna(), None).values.tolist()
                z_puts = pivot_puts.mul(100).where(~pivot_puts.isna(), None).values.tolist()
                
                # Use unified axes for both heatmaps
                x_heatmap = [pd.to_datetime(x).strftime('%Y-%m-%d') if not isinstance(x, str) else x[:10] for x in all_expiries_heatmap]
                y_heatmap = [f"${s:.0f}" for s in all_strikes_heatmap]
                
                # Assign to both (same axes for proper hover sync)
                x_calls = x_heatmap
                x_puts = x_heatmap
                y_calls = y_heatmap
                y_puts = y_heatmap
                
                # Calculate shared color range for both heatmaps (so colors are comparable)
                # Flatten and filter out None values to find min/max
                all_iv_values = []
                for row in z_calls:
                    all_iv_values.extend([v for v in row if v is not None])
                for row in z_puts:
                    all_iv_values.extend([v for v in row if v is not None])
                
                if all_iv_values:
                    # Use percentile cap from sidebar settings for outlier handling
                    iv_min = 0  # IV can't be negative
                    iv_max = np.percentile(all_iv_values, percentile_cap) if all_iv_values else 100
                else:
                    iv_min, iv_max = 0, 100
                
                # Stack heatmaps vertically (calls on top, puts below)
                fig_heatmap_calls = go.Figure(data=go.Heatmap(
                    z=z_calls,
                    x=x_calls,
                    y=y_calls,
                    colorscale=heatmap_colorscale,
                    zmin=iv_min,
                    zmax=iv_max,
                    colorbar=dict(
                        title=dict(text='Call IV (%)', font=dict(color=axis_title_color, size=9)),
                        tickfont=dict(color=tick_color, size=8),
                        tickformat='.0f',
                        ticksuffix='%',
                        len=0.6
                    ),
                    hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>IV: %{z:.1f}%<extra></extra>'
                ))
                fig_heatmap_calls.update_layout(
                    title=dict(text='Call Implied Volatility', font=dict(color=text_color, size=22), x=0.5, xanchor='center'),
                    template=chart_template,
                    height=285,
                    yaxis_title='Strike ($)',
                    xaxis_title='',
                    plot_bgcolor=plot_bg_color,
                    paper_bgcolor=paper_bg_color,
                    font=dict(color=text_color, size=9),
                    title_font=dict(color=text_color),
                    xaxis=dict(
                        title_font=dict(color=axis_title_color, size=9),
                        tickfont=dict(color=tick_color, size=8)
                    ),
                    yaxis=dict(
                        title_font=dict(color=axis_title_color, size=9),
                        tickfont=dict(color=tick_color, size=8)
                    ),
                    margin=dict(l=50, r=10, t=30, b=20)
                )
                call_heatmap_event = st.plotly_chart(fig_heatmap_calls, use_container_width=True, key="heatmap-call-iv", on_select="rerun", selection_mode="points")

                fig_heatmap_puts = go.Figure(data=go.Heatmap(
                    z=z_puts,
                    x=x_puts,
                    y=y_puts,
                    colorscale=heatmap_colorscale,
                    zmin=iv_min,
                    zmax=iv_max,
                    colorbar=dict(
                        title=dict(text='Put IV (%)', font=dict(color=axis_title_color, size=9)),
                        tickfont=dict(color=tick_color, size=8),
                        tickformat='.0f',
                        ticksuffix='%',
                        len=0.6
                    ),
                    hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>IV: %{z:.1f}%<extra></extra>'
                ))
                fig_heatmap_puts.update_layout(
                    title=dict(text='Put Implied Volatility', font=dict(color=text_color, size=22), x=0.5, xanchor='center'),
                    template=chart_template,
                    height=285,
                    yaxis_title='Strike ($)',
                    xaxis_title='Expiration',
                    plot_bgcolor=plot_bg_color,
                    paper_bgcolor=paper_bg_color,
                    font=dict(color=text_color, size=9),
                    title_font=dict(color=text_color),
                    xaxis=dict(
                        title_font=dict(color=axis_title_color, size=9),
                        tickfont=dict(color=tick_color, size=8)
                    ),
                    yaxis=dict(
                        title_font=dict(color=axis_title_color, size=9),
                        tickfont=dict(color=tick_color, size=8)
                    ),
                    margin=dict(l=50, r=10, t=30, b=40)
                )
                put_heatmap_event = st.plotly_chart(fig_heatmap_puts, use_container_width=True, key="heatmap-put-iv", on_select="rerun", selection_mode="points")
                
                # Capture selection events from heatmaps to update selected date
                # Check for selections in call heatmap using return value
                if call_heatmap_event and hasattr(call_heatmap_event, 'selection') and call_heatmap_event.selection.points:
                    points = call_heatmap_event.selection.points
                    if points and len(points) > 0:
                        selected_x = points[0].get("x") if isinstance(points[0], dict) else getattr(points[0], "x", None)
                        if selected_x:
                            try:
                                clicked_date = pd.to_datetime(selected_x).strftime('%Y-%m-%d')
                                if clicked_date in all_dates_str and clicked_date != st.session_state.get('selected_date_from_chart'):
                                    st.session_state['selected_date_from_chart'] = clicked_date
                                    st.rerun()
                            except:
                                pass

                # Check for selections in put heatmap using return value
                if put_heatmap_event and hasattr(put_heatmap_event, 'selection') and put_heatmap_event.selection.points:
                    points = put_heatmap_event.selection.points
                    if points and len(points) > 0:
                        selected_x = points[0].get("x") if isinstance(points[0], dict) else getattr(points[0], "x", None)
                        if selected_x:
                            try:
                                clicked_date = pd.to_datetime(selected_x).strftime('%Y-%m-%d')
                                if clicked_date in all_dates_str and clicked_date != st.session_state.get('selected_date_from_chart'):
                                    st.session_state['selected_date_from_chart'] = clicked_date
                                    st.rerun()
                            except:
                                pass
                
                # Inject sync script for heatmaps (sync x-axis)
                sync_script_heatmaps = create_xaxis_sync_script(["heatmap-call-iv", "heatmap-put-iv"])
                components.html(sync_script_heatmaps, height=0)
                
                # Add specialized hover sync for heatmaps - uses pointNumber directly since axes are now unified
                heatmap_hover_sync = """
                <script>
                (function() {
                    const rootDoc = (window.parent && window.parent.document) ? window.parent.document : document;
                    let isHovering = false;
                    let hoverTimeout;
                    let setupComplete = false;
                    
                    function findAllPlotlyCharts() {
                        // Find all plotly charts and return the last two (which should be the heatmaps)
                        const allCharts = Array.from(rootDoc.querySelectorAll('.js-plotly-plot'));
                        // Filter to only heatmaps (they have z data)
                        const heatmaps = allCharts.filter(chart => {
                            return chart.data && chart.data[0] && chart.data[0].type === 'heatmap';
                        });
                        return heatmaps;
                    }
                    
                    function setupHeatmapSync() {
                        if (setupComplete) return;
                        
                        const heatmaps = findAllPlotlyCharts();
                        console.log('Found heatmaps:', heatmaps.length);
                        
                        if (heatmaps.length >= 2) {
                            setupComplete = true;
                            const callHeatmap = heatmaps[0];
                            const putHeatmap = heatmaps[1];
                            
                            console.log('Setting up heatmap hover sync between Call and Put IV heatmaps');
                            
                            // Sync Call -> Put
                            callHeatmap.on('plotly_hover', function(eventData) {
                                if (!eventData || !eventData.points || eventData.points.length === 0) return;
                                if (isHovering) return;
                                
                                clearTimeout(hoverTimeout);
                                isHovering = true;
                                
                                const point = eventData.points[0];
                                // For heatmaps, pointNumber is [xIndex, yIndex] or we can use x/y directly
                                const pointNum = point.pointNumber;
                                
                                try {
                                    const Plotly = window.parent.Plotly || window.Plotly;
                                    if (Plotly && putHeatmap.data && putHeatmap.data[0]) {
                                        // Use the same pointNumber since axes are unified
                                        Plotly.Fx.hover(putHeatmap, [{
                                            curveNumber: 0,
                                            pointNumber: pointNum
                                        }]);
                                    }
                                } catch(e) {
                                    console.error('Heatmap hover sync error (Call->Put):', e);
                                }
                                
                                setTimeout(() => { isHovering = false; }, 50);
                            });
                            
                            // Sync Put -> Call
                            putHeatmap.on('plotly_hover', function(eventData) {
                                if (!eventData || !eventData.points || eventData.points.length === 0) return;
                                if (isHovering) return;
                                
                                clearTimeout(hoverTimeout);
                                isHovering = true;
                                
                                const point = eventData.points[0];
                                const pointNum = point.pointNumber;
                                
                                try {
                                    const Plotly = window.parent.Plotly || window.Plotly;
                                    if (Plotly && callHeatmap.data && callHeatmap.data[0]) {
                                        Plotly.Fx.hover(callHeatmap, [{
                                            curveNumber: 0,
                                            pointNumber: pointNum
                                        }]);
                                    }
                                } catch(e) {
                                    console.error('Heatmap hover sync error (Put->Call):', e);
                                }
                                
                                setTimeout(() => { isHovering = false; }, 50);
                            });
                            
                            // Unhover sync
                            callHeatmap.on('plotly_unhover', function() {
                                hoverTimeout = setTimeout(() => {
                                    isHovering = false;
                                    try {
                                        const Plotly = window.parent.Plotly || window.Plotly;
                                        if (Plotly) Plotly.Fx.unhover(putHeatmap);
                                    } catch(e) {}
                                }, 100);
                            });
                            
                            putHeatmap.on('plotly_unhover', function() {
                                hoverTimeout = setTimeout(() => {
                                    isHovering = false;
                                    try {
                                        const Plotly = window.parent.Plotly || window.Plotly;
                                        if (Plotly) Plotly.Fx.unhover(callHeatmap);
                                    } catch(e) {}
                                }, 100);
                            });
                            
                            console.log('Heatmap hover sync setup complete');
                        } else {
                            // Retry
                            setTimeout(setupHeatmapSync, 300);
                        }
                    }
                    
                    // Multiple attempts to ensure charts are loaded
                    setTimeout(setupHeatmapSync, 500);
                    setTimeout(setupHeatmapSync, 1000);
                    setTimeout(setupHeatmapSync, 2000);
                    setTimeout(setupHeatmapSync, 3000);
                })();
                </script>
                """
                components.html(heatmap_hover_sync, height=0)
    
    # --- Liquidity Walls Section ---
    with st.container(border=True):
        st.markdown("#### Liquidity Walls: Open Interest & Volume")
        with st.expander("Understanding Open Interest vs Volume", expanded=False):
            st.markdown("""
            **Open Interest (OI)** and **Volume** measure different aspects of options market activity:

            **Open Interest:** Total contracts currently open at each strike.
            - **What to Look For:** High OI creates "walls" - the stock often gets pinned to these strikes near expiration due to dealer hedging.
            - **Use Case:** Identifies established support/resistance levels. Strikes with massive OI act as magnets.

            **Volume:** Contracts traded today (resets daily).
            - **What to Look For:** Spikes show where smart money is actively positioning right now.
            - **Use Case:** Identifies emerging trends and unusual institutional activity.

            **Combined Analysis:**
            - **High OI + High Volume:** Strong conviction at this level
            - **High OI + Low Volume:** Established wall, no new activity
            - **Low OI + High Volume:** Fresh positioning that could move the market
            """)
        
        if "greeks" in data and data["greeks"]:
            # Use global filtered data for Liquidity Analysis
            if selected_range_start and selected_range_end:
                 df_greeks_liq = df_greeks_global[
                    (df_greeks_global['expiry_str'] >= selected_range_start) & 
                    (df_greeks_global['expiry_str'] <= selected_range_end)
                 ].copy()
            else:
                 df_greeks_liq = df_greeks_global.copy()
            
            # Ensure expiry_date column exists (used in downstream logic)
            df_greeks_liq['expiry_date'] = df_greeks_liq['expiry_dt']
            
            # date_range derived from the filtered data
            date_range = sorted(df_greeks_liq['expiry_str'].unique())
            
            # df_range is the main dataframe used for charts below
            df_range = df_greeks_liq
            
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
                    ["Expiration Date", "Strike Price"],
                    horizontal=True,
                    key="x_axis_range_analysis",
                    index=0,
                    help=get_tooltip("X-Axis")
                )
            else:
                x_axis_option = "Expiration Date"
            
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
                
                # Open Interest and Volume Charts side-by-side
                # Width parameter depends on x-axis type (numeric vs categorical)
                bar_width = None if x_axis_option == "Expiration Date" else 1.6

                col_oi, col_vol = st.columns(2)

                with col_oi:
                    st.markdown("**Open Interest**")
                    fig_oi = go.Figure()
                    fig_oi.add_trace(go.Bar(
                        x=all_x_values,
                        y=call_oi_values,
                        name='Call OI',
                        marker_color=call_color,
                        opacity=1.0,
                        width=bar_width
                    ))
                    fig_oi.add_trace(go.Bar(
                        x=all_x_values,
                        y=[-oi for oi in put_oi_values],  # Negative for below
                        name='Put OI',
                        marker_color=put_color,
                        opacity=1.0,
                        width=bar_width
                    ))
                    fig_oi.update_layout(
                        barmode='overlay',
                        title=dict(
                            text=f"Open Interest by {x_axis_option}<br><sub>Total contracts open at each level (support/resistance walls)</sub>",
                            font=dict(color=text_color, size=20),
                            x=0.5,
                            xanchor='center'
                        ),
                        template=chart_template,
                        height=400,
                        xaxis_title=x_axis_title,
                        yaxis_title='OI (Contracts)',
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
                    oi_event = st.plotly_chart(fig_oi, use_container_width=True, key="oi-range-chart", on_select="rerun", selection_mode="points")

                with col_vol:
                    st.markdown("**Volume**")
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=all_x_values,
                        y=call_vol_values,
                        name='Call Volume',
                        marker_color=call_color,
                        opacity=1.0,
                        width=bar_width
                    ))
                    fig_vol.add_trace(go.Bar(
                        x=all_x_values,
                        y=[-vol for vol in put_vol_values],  # Negative for below
                        name='Put Volume',
                        marker_color=put_color,
                        opacity=1.0,
                        width=bar_width
                    ))
                    fig_vol.update_layout(
                        barmode='overlay',
                        title=dict(
                            text=f"Volume by {x_axis_option}<br><sub>Contracts traded today (current activity & positioning)</sub>",
                            font=dict(color=text_color, size=20),
                            x=0.5,
                            xanchor='center'
                        ),
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
                    vol_event = st.plotly_chart(fig_vol, use_container_width=True, key="vol-range-chart", on_select="rerun", selection_mode="points")
                
                # Capture selection events from Liquidity Range Charts (when X-axis is Expiration Date)
                # Use the return value from st.plotly_chart which contains the selection data
                if x_axis_option == "Expiration Date":
                    # Check OI chart selection from return value
                    if oi_event and hasattr(oi_event, 'selection') and oi_event.selection.points:
                        points = oi_event.selection.points
                        if points and len(points) > 0:
                            selected_x = points[0].get("x") if isinstance(points[0], dict) else getattr(points[0], "x", None)
                            if selected_x:
                                try:
                                    clicked_date = pd.to_datetime(selected_x).strftime('%Y-%m-%d')
                                    if clicked_date in all_dates_str and clicked_date != st.session_state.get('selected_date_from_chart'):
                                        st.session_state['selected_date_from_chart'] = clicked_date
                                        st.rerun()
                                except Exception as e:
                                    pass
                    
                    # Check Volume chart selection from return value
                    if vol_event and hasattr(vol_event, 'selection') and vol_event.selection.points:
                        points = vol_event.selection.points
                        if points and len(points) > 0:
                            selected_x = points[0].get("x") if isinstance(points[0], dict) else getattr(points[0], "x", None)
                            if selected_x:
                                try:
                                    clicked_date = pd.to_datetime(selected_x).strftime('%Y-%m-%d')
                                    if clicked_date in all_dates_str and clicked_date != st.session_state.get('selected_date_from_chart'):
                                        st.session_state['selected_date_from_chart'] = clicked_date
                                        st.rerun()
                                except Exception as e:
                                    pass
                
                # Inject sync scripts for range charts
                sync_script_range = create_xaxis_sync_script(["oi-range-chart", "vol-range-chart"])
                components.html(sync_script_range, height=0)
                
                # HARDCODED hover sync for Range Charts (oi-range-chart + vol-range-chart)
                # Rebinds after rerenders; detects first two 2-trace bar charts; click updates date when x-axis is Expiration Date
                x_axis_is_date = "true" if x_axis_option == "Expiration Date" else "false"
                range_hover_sync_script = f"""
                <script>
                (function() {{
                    const rootDoc = (window.parent && window.parent.document) ? window.parent.document : document;
                    let isHovering = false;
                    let oiChart = null, volChart = null;
                    let lastKey = "";
                    const xAxisIsDate = { "true" if x_axis_option == "Expiration Date" else "false" };

                    function findRangeCharts() {{
                        const charts = Array.from(rootDoc.querySelectorAll('.js-plotly-plot'));
                        const bars2 = charts.filter(c => c.data && c.data.length === 2 && c.data[0].type === 'bar');
                        if (bars2.length >= 2 && bars2[0].data[0].x && bars2[1].data[0].x) return {{ oi: bars2[0], vol: bars2[1] }};
                        return null;
                    }}

                    function bindIfNeeded() {{
                        const found = findRangeCharts();
                        if (!found) return;
                        const key = `${{found.oi}}_${{found.vol}}_${{(found.oi.data[0].x||[]).length}}_${{(found.vol.data[0].x||[]).length}}`;
                        if (key === lastKey) return;
                        lastKey = key;

                        oiChart = found.oi; volChart = found.vol;
                        console.log('Range hover sync rebound');

                        function sync(source, target, pointNum) {{
                            const Plotly = window.parent.Plotly || window.Plotly;
                            if (!Plotly || !target || !target.data) return;
                            const pts = [];
                            for (let i=0;i<target.data.length;i++) {{
                                if (target.data[i].x && target.data[i].x.length > pointNum) pts.push({{curveNumber:i, pointNumber:pointNum}});
                            }}
                            if (pts.length) Plotly.Fx.hover(target, pts);
                        }}

                        function handleClick(ev) {{
                            if (!xAxisIsDate || !ev?.points?.[0]) return;
                            const clickedX = ev.points[0].x;
                            const url = new URL(window.parent.location.href);
                            url.searchParams.set('clicked_date', clickedX);
                            window.parent.location.href = url.toString();
                        }}

                        oiChart.on('plotly_hover', ev => {{ if (isHovering || !ev?.points?.[0]) return; isHovering=true; sync(oiChart, volChart, ev.points[0].pointNumber); setTimeout(()=>isHovering=false,50); }});
                        volChart.on('plotly_hover', ev => {{ if (isHovering || !ev?.points?.[0]) return; isHovering=true; sync(volChart, oiChart, ev.points[0].pointNumber); setTimeout(()=>isHovering=false,50); }});

                        oiChart.on('plotly_click', handleClick);
                        volChart.on('plotly_click', handleClick);

                        oiChart.on('plotly_unhover', () => {{ try {{ (window.parent.Plotly||window.Plotly).Fx.unhover(volChart); }} catch(e){{}} }});
                        volChart.on('plotly_unhover', () => {{ try {{ (window.parent.Plotly||window.Plotly).Fx.unhover(oiChart); }} catch(e){{}} }});
                    }}

                    function tick(attempt=0) {{
                        bindIfNeeded();
                        if (attempt < 120) setTimeout(()=>tick(attempt+1), 500);
                    }}
                    setTimeout(()=>tick(0), 500);
                }})();
                </script>
                """
                components.html(range_hover_sync_script, height=0)
            
            # --- Selected Date Drill-Down Section ---
            # This section is ALWAYS strike-price based, independent of range chart x-axis setting
            # It shows detailed analysis for a single selected date across all strike prices
            if not df_range.empty:
                st.markdown("---")
                # Date is already selected globally via sidebar
                # Fallback only if somehow it's None (though sidebar logic should handle this)
                if 'selected_date' not in locals() or selected_date is None:
                     if date_range:
                         selected_date = date_range[0]
                     else:
                         selected_date = None

                if selected_date:
                    # Title and date selector on same row
                    strike_title_col, strike_date_col = st.columns([2, 1])
                    with strike_title_col:
                        st.markdown("### Strike-Level Analysis for Selected Date")
                    with strike_date_col:
                        if available_dates_in_range:
                            # Default to current selection or first date
                            default_date = selected_date if selected_date in available_dates_in_range else available_dates_in_range[0]

                            selected_date = st.select_slider(
                                "Select Date",
                                options=available_dates_in_range,
                                value=default_date,
                                key="selected_exp_date_inline",
                                help="Select specific expiration date for detailed strike-level analysis.",
                                label_visibility="collapsed"
                            )
                            # Sync back to session state
                            st.session_state['selected_date_from_chart'] = selected_date
                    
                    # Filter data for selected date - STRIKE PRICE ANALYSIS ONLY
                    selected_date_data = df_range[df_range['expiry_date'].dt.strftime('%Y-%m-%d') == selected_date]
                    
                    # Validate that we have strike price data for this date
                    if not selected_date_data.empty and 'strike' in selected_date_data.columns:
                        # Aggregate OI by strike for the selected date
                        calls_oi_by_strike = selected_date_data[selected_date_data['type'] == 'call'].groupby('strike')['openInterest'].sum()
                        puts_oi_by_strike = selected_date_data[selected_date_data['type'] == 'put'].groupby('strike')['openInterest'].sum()
                        
                        # Get all strikes for this date (ensure we have strike data)
                        all_strikes = sorted(set(list(calls_oi_by_strike.index) + list(puts_oi_by_strike.index)))
                        
                        # Validation: ensure we have valid strike data  
                        if not all_strikes:
                            st.warning(f"No strike price data available for {selected_date}. Please select a different date.")
                        
                        # Proceed with strike-based analysis
                        call_oi_values = [calls_oi_by_strike.get(strike, 0) for strike in all_strikes]
                        put_oi_values = [puts_oi_by_strike.get(strike, 0) for strike in all_strikes]
                        
                        # Aggregate Volume by strike for the selected date
                        calls_vol_by_strike = selected_date_data[selected_date_data['type'] == 'call'].groupby('strike')['volume'].sum()
                        puts_vol_by_strike = selected_date_data[selected_date_data['type'] == 'put'].groupby('strike')['volume'].sum()
                        
                        call_vol_values = [calls_vol_by_strike.get(strike, 0) for strike in all_strikes]
                        put_vol_values = [puts_vol_by_strike.get(strike, 0) for strike in all_strikes]
                        
                        # Define chart keys for synchronization (Strike-based charts only)
                        oi_selected_key = "oi-selected-chart"
                        vol_selected_key = "vol-selected-chart"
                        delta_selected_key = "delta-selected-chart"
                        gamma_selected_key = "gamma-selected-chart"
                        theta_selected_key = "theta-selected-chart"
                    
                    # Create two charts side by side
                    col_oi, col_vol = st.columns(2)
                    
                    with col_oi:
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
                        st.plotly_chart(fig_oi_by_strike, use_container_width=True, key=oi_selected_key)
                    
                    with col_vol:
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
                        st.plotly_chart(fig_vol_by_strike, use_container_width=True, key=vol_selected_key)
                    
                    # Add Greeks Slice Analysis
                    st.markdown("---")
                    st.markdown("#### Greeks Slice Analysis for Selected Date")
                    with st.expander("How to Use This Chart", expanded=False):
                        st.markdown("""
                        The Greeks measure the risk of a specific options contract at different strike prices.
                        
                        **Delta:** Measures direction risk - how much the option price changes for each `$1` move in the stock.
                        - **Range:** Calls (0 to 1), Puts (-1 to 0)
                        - **What to Look For:** Higher Delta = more stock-like behavior. At-the-money options have Delta around `0.50`.
                        
                        **Gamma:** Measures acceleration risk - how quickly Delta changes as the stock moves.
                        - **What to Look For:** Peak Gamma at-the-money means Delta changes fastest when stock is near the strike. This creates volatility in option prices.
                        
                        **Theta:** Measures time decay - how much value the option loses each day.
                        - **What to Look For:** Time decay accelerates as expiration approaches. Options with less time to expiration have higher absolute Theta values.
                        """)
                    
                    selected_type = st.radio("Option Type", ["call", "put"], horizontal=True, key="greeks_type_selected_date", help=get_tooltip("Option Type"))
                    
                    # Filter data for selected date and type
                    filtered_df = selected_date_data[selected_date_data['type'] == selected_type].sort_values('strike')
                    
                    if not filtered_df.empty:
                        logger.debug(f"Generating synced Greek charts for {selected_date}...")
                        
                        try:
                            # 1. Create Subplots: 1 Row, 3 Columns
                            # shared_xaxes=True is the magic switch that links the hover interaction
                            fig_greeks = make_subplots(
                                rows=1,
                                cols=3,
                                shared_xaxes=True,
                                subplot_titles=(
                                    "Delta<br><sub>Option price change per $1 stock move</sub>",
                                    "Gamma<br><sub>Rate of Delta change (acceleration risk)</sub>",
                                    "Theta<br><sub>Daily time decay ($/day lost)</sub>"
                                ),
                                horizontal_spacing=0.05
                            )

                            # 2. Add DELTA to Column 1 (Blue)
                            fig_greeks.add_trace(
                                go.Scatter(
                                    x=filtered_df['strike'], 
                                    y=filtered_df['delta'], 
                                    mode='lines', 
                                    name='Delta',
                                    line=dict(color='#1f77b4'),
                                    # Note the use of <extra></extra> to hide the trace name in the tooltip
                                    hovertemplate='<b>Delta</b><br>Strike: $%{x:,.0f}<br>Value: %{y:.4f}<extra></extra>'
                                ),
                                row=1, col=1
                            )

                            # 3. Add GAMMA to Column 2 (Orange)
                            fig_greeks.add_trace(
                                go.Scatter(
                                    x=filtered_df['strike'], 
                                    y=filtered_df['gamma'], 
                                    mode='lines', 
                                    name='Gamma',
                                    line=dict(color='#ff7f0e'),
                                    hovertemplate='<b>Gamma</b><br>Strike: $%{x:,.0f}<br>Value: %{y:.6f}<extra></extra>'
                                ),
                                row=1, col=2
                            )

                            # 4. Add THETA to Column 3 (Green)
                            fig_greeks.add_trace(
                                go.Bar(
                                    x=filtered_df['strike'], 
                                    y=filtered_df['theta'], 
                                    name='Theta',
                                    marker_color='#2ca02c',
                                    hovertemplate='<b>Theta</b><br>Strike: $%{x:,.0f}<br>Value: %{y:.4f}<extra></extra>'
                                ),
                                row=1, col=3
                            )

                            # 5. Native Synchronization Settings
                            fig_greeks.update_layout(
                                template=chart_template,
                                height=400,
                                # 'x unified' creates the crosshair that slices through ALL subplots
                                hovermode='x unified', 
                                showlegend=False,
                                plot_bgcolor=plot_bg_color,
                                paper_bgcolor=paper_bg_color,
                                font=dict(color=text_color),
                                title_font=dict(color=text_color),
                            )

                            # 6. Uniform Axis Labels
                            fig_greeks.update_xaxes(
                                title_text="Strike Price ($)", 
                                title_font=dict(color=axis_title_color), 
                                tickfont=dict(color=tick_color),
                                showgrid=True,
                                gridcolor=grid_color
                            )
                            # Individual y-axis titles for each subplot
                            fig_greeks.update_yaxes(
                                title_text="Delta (Price Sensitivity)",
                                title_font=dict(color=axis_title_color),
                                tickfont=dict(color=tick_color),
                                showgrid=True,
                                gridcolor=grid_color,
                                row=1, col=1
                            )
                            fig_greeks.update_yaxes(
                                title_text="Gamma (Delta Sensitivity)",
                                title_font=dict(color=axis_title_color),
                                tickfont=dict(color=tick_color),
                                showgrid=True,
                                gridcolor=grid_color,
                                row=1, col=2
                            )
                            fig_greeks.update_yaxes(
                                title_text="Theta ($/day)",
                                title_font=dict(color=axis_title_color),
                                tickfont=dict(color=tick_color),
                                showgrid=True,
                                gridcolor=grid_color,
                                row=1, col=3
                            )

                            # 7. Render
                            st.plotly_chart(fig_greeks, use_container_width=True, key="greeks_synced_chart")
                            logger.info("Greek charts rendered successfully.")

                        except Exception as e:
                            # This captures the error in your terminal so you can fix data issues
                            logger.error(f"Failed to render Greek charts: {str(e)}", exc_info=True)
                            st.error(f"Chart Error: {str(e)}")
                        
                        # HARDCODED hover sync for Selected Date OI/Volume charts
                        # Uses polling + data-based detection (3rd and 4th 2-trace bar charts)
                        selected_oi_vol_hover_sync = """
                        <script>
                        (function() {
                            const rootDoc = (window.parent && window.parent.document) ? window.parent.document : document;
                            let isHovering = false;
                            let oiChart = null;
                            let volChart = null;
                            let setupDone = false;
                            
                            function findSelectedDateBarCharts() {
                                const allCharts = Array.from(rootDoc.querySelectorAll('.js-plotly-plot'));
                                const barCharts = allCharts.filter(chart => {
                                    return chart.data && chart.data.length === 2 && chart.data[0].type === 'bar';
                                });
                                
                                if (barCharts.length >= 4) {
                                    if (barCharts[2].data[0].x && barCharts[3].data[0].x) {
                                        return { oi: barCharts[2], vol: barCharts[3] };
                                    }
                                }
                                return null;
                            }
                            
                            function syncHover(sourceChart, targetChart, pointNum) {
                                if (!targetChart || !targetChart.data) return;
                                try {
                                    const Plotly = window.parent.Plotly || window.Plotly;
                                    if (!Plotly) return;
                                    
                                    const hoverPoints = [];
                                    for (let i = 0; i < targetChart.data.length; i++) {
                                        if (targetChart.data[i].x && targetChart.data[i].x.length > pointNum) {
                                            hoverPoints.push({ curveNumber: i, pointNumber: pointNum });
                                        }
                                    }
                                    if (hoverPoints.length > 0) {
                                        Plotly.Fx.hover(targetChart, hoverPoints);
                                    }
                                } catch(e) {}
                            }
                            
                            function setup() {
                                if (setupDone) return true;
                                
                                const charts = findSelectedDateBarCharts();
                                if (!charts) return false;
                                
                                setupDone = true;
                                oiChart = charts.oi;
                                volChart = charts.vol;
                                console.log('Selected Date OI/Vol hover sync: Setup complete');
                                
                                oiChart.on('plotly_hover', function(eventData) {
                                    if (isHovering || !eventData || !eventData.points || !eventData.points[0]) return;
                                    isHovering = true;
                                    syncHover(oiChart, volChart, eventData.points[0].pointNumber);
                                    setTimeout(() => { isHovering = false; }, 50);
                                });
                                
                                volChart.on('plotly_hover', function(eventData) {
                                    if (isHovering || !eventData || !eventData.points || !eventData.points[0]) return;
                                    isHovering = true;
                                    syncHover(volChart, oiChart, eventData.points[0].pointNumber);
                                    setTimeout(() => { isHovering = false; }, 50);
                                });
                                
                                oiChart.on('plotly_unhover', function() {
                                    try {
                                        const Plotly = window.parent.Plotly || window.Plotly;
                                        if (Plotly) Plotly.Fx.unhover(volChart);
                                    } catch(e) {}
                                });
                                
                                volChart.on('plotly_unhover', function() {
                                    try {
                                        const Plotly = window.parent.Plotly || window.Plotly;
                                        if (Plotly) Plotly.Fx.unhover(oiChart);
                                    } catch(e) {}
                                });
                                
                                return true;
                            }
                            
                            // Polling only
                            let attempts = 0;
                            function trySetup() {
                                attempts++;
                                if (setup() || attempts > 60) return;
                                setTimeout(trySetup, 500);
                            }
                            setTimeout(trySetup, 1000);
                        })();
                        </script>
                        """
                        components.html(selected_oi_vol_hover_sync, height=0)
                        
                        # Hover sync for Greeks charts (Delta, Gamma, Theta) - hard-coded IDs + x-value matching
                        greeks_hover_sync = """
                        <script>
                        (function() {
                            const parentDoc = (window.parent && window.parent.document) ? window.parent.document : document;
                            const Plotly = (window.parent && window.parent.Plotly) ? window.parent.Plotly : window.Plotly;
                            if (!Plotly) return;

                            let isHovering = false;

                            const TARGET_IDS = {
                                DELTA: 'chart_delta_unique',
                                GAMMA: 'chart_gamma_unique',
                                THETA: 'chart_theta_unique'
                            };

                            function log(msg) { console.log('[HardSync] ' + msg); }

                            function findSpecificCharts() {
                                const allPlots = Array.from(parentDoc.querySelectorAll('.js-plotly-plot'));
                                const found = { delta: null, gamma: null, theta: null };
                                allPlots.forEach(plot => {
                                    const id = plot?.layout?.meta?.hard_id;
                                    if (!id) return;
                                    if (id === TARGET_IDS.DELTA) found.delta = plot;
                                    if (id === TARGET_IDS.GAMMA) found.gamma = plot;
                                    if (id === TARGET_IDS.THETA) found.theta = plot;
                                });
                                return found;
                            }

                            function findIndexByStrikePrice(chart, strikePrice) {
                                if (!chart?.data || !chart.data.length) return null;
                                const xData = chart.data[0].x || [];
                                let idx = xData.findIndex(val => Math.abs(+val - +strikePrice) < 0.01);
                                if (idx === -1) {
                                    // nearest fallback
                                    let best = -1, diff = Infinity;
                                    xData.forEach((v, i) => {
                                        const d = Math.abs(+v - +strikePrice);
                                        if (!Number.isNaN(d) && d < diff) { diff = d; best = i; }
                                    });
                                    idx = best;
                                }
                                return idx >= 0 ? idx : null;
                            }

                            function executeSync(sourceChart, eventData) {
                                if (isHovering || !eventData?.points?.length) return;
                                isHovering = true;
                                const strikePrice = eventData.points[0].x;
                                const charts = findSpecificCharts();
                                const targets = [charts.delta, charts.gamma, charts.theta].filter(c => c && c !== sourceChart);
                                targets.forEach(target => {
                                    const targetIdx = findIndexByStrikePrice(target, strikePrice);
                                    if (targetIdx !== null) {
                                        try {
                                            Plotly.Fx.hover(target, [{ curveNumber: 0, pointNumber: targetIdx }]);
                                        } catch(e) { console.error(e); }
                                    }
                                });
                                setTimeout(() => { isHovering = false; }, 20);
                            }

                            function executeUnhover(sourceChart) {
                                const charts = findSpecificCharts();
                                const targets = [charts.delta, charts.gamma, charts.theta].filter(c => c && c !== sourceChart);
                                targets.forEach(target => {
                                    try { Plotly.Fx.unhover(target); } catch(e) {}
                                });
                            }

                            function bindListeners() {
                                const charts = findSpecificCharts();
                                const chartList = [charts.delta, charts.gamma, charts.theta].filter(c => c);
                                if (!chartList.length) return;
                                chartList.forEach(chart => {
                                    if (chart._hardSyncBound) return;
                                    if (chart.removeAllListeners) {
                                        chart.removeAllListeners('plotly_hover');
                                        chart.removeAllListeners('plotly_unhover');
                                    }
                                    log('Binding listener to: ' + (chart?.layout?.meta?.hard_id || 'unknown'));
                                    chart.on('plotly_hover', data => executeSync(chart, data));
                                    chart.on('plotly_unhover', () => executeUnhover(chart));
                                    chart._hardSyncBound = true;
                                });
                            }

                            function startMonitoring() {
                                log('Starting Hard-Sync Monitor');
                                bindListeners();
                                const observer = new MutationObserver(() => { bindListeners(); });
                                observer.observe(parentDoc.body, { childList: true, subtree: true });
                            }

                            setTimeout(startMonitoring, 1000);
                        })();
                        </script>
                        """
                        components.html(greeks_hover_sync, height=0)

else:
    st.info("Enter a ticker symbol and click 'Analyze' to begin.")

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm

# --- HELPER FUNCTIONS ---
def calculate_gex_profile(df_greeks, spot_price):
    """
    Calculates Gamma Exposure (GEX) Profile per strike.
    GEX = Gamma * Open Interest * 100 * Spot^2 * 0.01
    Direction: Call GEX is positive, Put GEX is negative.
    """
    if df_greeks.empty or spot_price is None:
        return pd.DataFrame()
        
    # Ensure numeric types
    df = df_greeks.copy()
    df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce').fillna(0)
    df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').fillna(0)
    
    # Calculate Notional GEX per contract
    # Factor 0.01 is to normalize for 1% move
    const_factor = 100 * (spot_price**2) * 0.01
    
    # Call GEX (Positive)
    df.loc[df['type'] == 'call', 'gex'] = df['gamma'] * df['openInterest'] * const_factor
    
    # Put GEX (Negative)
    df.loc[df['type'] == 'put', 'gex'] = df['gamma'] * df['openInterest'] * const_factor * -1
    
    # Aggregate by strike
    gex_profile = df.groupby('strike')['gex'].sum().reset_index()
    return gex_profile

def calculate_zero_gamma(df_greeks, current_spot):
    """
    Estimates the Zero Gamma Flip Level by simulating spot price moves.
    """
    if df_greeks.empty or current_spot is None:
        return None
        
    gex_profile = calculate_gex_profile(df_greeks, current_spot)
    if gex_profile.empty:
        return None
        
    # Sort by strike
    gex_profile = gex_profile.sort_values('strike')
    
    # Calculate cumulative sum
    gex_profile['cum_gex'] = gex_profile['gex'].cumsum()
    
    # Find where sign flips from positive to negative or vice versa
    # Iterate to find crossing
    flip_price = None
    for i in range(1, len(gex_profile)):
        prev_gex = gex_profile.iloc[i-1]['cum_gex']
        curr_gex = gex_profile.iloc[i]['cum_gex']
        
        if (prev_gex > 0 and curr_gex < 0) or (prev_gex < 0 and curr_gex > 0):
            # Linear interpolation for the crossing strike
            strike_prev = gex_profile.iloc[i-1]['strike']
            strike_curr = gex_profile.iloc[i]['strike']
            
            # Simple average for now
            flip_price = (strike_prev + strike_curr) / 2
            break
            
    return flip_price

def generate_probability_cone_data(spot_price, current_iv, start_date, days=60):
    """
    Generates data for Probability Cone (1SD and 2SD), starting from a specific date.
    """
    days_future = np.arange(1, days)
    
    # Anchor the start of the cone to the last known price
    cone_anchor_price = spot_price

    upper_1sd = cone_anchor_price * (1 + current_iv * np.sqrt(days_future/365))
    lower_1sd = cone_anchor_price * (1 - current_iv * np.sqrt(days_future/365))
    
    upper_2sd = cone_anchor_price * (1 + 2 * current_iv * np.sqrt(days_future/365))
    lower_2sd = cone_anchor_price * (1 - 2 * current_iv * np.sqrt(days_future/365))
    
    # Start dates from the day after the last historical date
    dates_future = [start_date + timedelta(days=int(d)) for d in days_future]
    
    # Prepend the anchor point to the cone data arrays to ensure visual connection
    dates_future.insert(0, start_date)
    upper_1sd = np.insert(upper_1sd, 0, cone_anchor_price)
    lower_1sd = np.insert(lower_1sd, 0, cone_anchor_price)
    upper_2sd = np.insert(upper_2sd, 0, cone_anchor_price)
    lower_2sd = np.insert(lower_2sd, 0, cone_anchor_price)

    return dates_future, upper_1sd, lower_1sd, upper_2sd, lower_2sd

# Config
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Options Viz", layout="wide")
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
            </style>
        """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()

# Sidebar Visualization Settings (New)
with st.sidebar.expander("🎨 Visualization Settings", expanded=False):
    st.session_state["dark_mode"] = st.checkbox("Dark Mode (Pure Black)", value=st.session_state["dark_mode"])
    
    st.caption("Heatmap Outlier Handling")
    use_log_scale = st.checkbox("Use Log Scale (Open Interest)", value=True, help="Compresses massive spikes to make smaller values visible.")
    percentile_cap = st.slider("Color Cap (Percentile)", 50, 100, 99, help="Cap colors at this percentile to ignore extreme outliers.")

apply_theme() # Apply CSS based on state

# Sidebar Educational Tooltips
with st.sidebar.expander("📚 What are these metrics?"):
    st.markdown(r"""
    **Implied Volatility (IV):**
    A measure of the market's expected future volatility. High IV means options are expensive (high fear/uncertainty).
    
    **Delta ($\Delta$):**
    How much an option's price changes for a $1 move in the stock.
    *   Calls: 0 to 1
    *   Puts: -1 to 0
    
    **Gamma ($\Gamma$):**
    The rate of change of Delta. High Gamma means your Delta changes rapidly (risky!).
    
    **Theta ($\Theta$):**
    Time decay. How much value an option loses *every day* just by holding it.
    """)
    
    st.info("Tip: Use the 'Strategy Simulator' tab to see how Time Decay (Theta) affects your profit over time.")

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
                st.session_state['historical_price'] = pd.DataFrame(resp_hist.json())
                
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# --- MAIN CONTENT ---
if st.session_state['data']:
    data = st.session_state['data']
    chart_template = "plotly_dark" if st.session_state["dark_mode"] else "plotly"
    
    # --- TABS STRUCTURE ---
    tab_main, tab_advanced = st.tabs(["Main Dashboard", "Advanced Charts"])
    
    with tab_main:
        # --- Top Row Metrics: Comparative Market Pulse ---
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown(f"**📡 Market Pulse: {ticker}**")
                if st.session_state['summary']:
                    summ = st.session_state['summary']
                    m1, m2, m3 = st.columns(3)
                    pcr = summ['putCallRatio']
                    m1.metric("Put/Call Ratio", pcr, delta="Bearish" if pcr > 1.0 else ("Bullish" if pcr < 0.7 else "Neutral"), delta_color="inverse")
                    avg_iv = summ['averageIV']
                    m2.metric("Average IV", f"{avg_iv:.2%}", "Fear Gauge")
                    vol = summ['totalVolume']
                    m3.metric("Total Volume", f"{vol:,}")
        
        with c2:
            with st.container(border=True):
                st.markdown("**📡 Market Pulse: SPY**")
                if st.session_state['spy_summary']:
                    spy_summ = st.session_state['spy_summary']
                    m1, m2, m3 = st.columns(3)
                    spy_pcr = spy_summ['putCallRatio']
                    m1.metric("Put/Call Ratio", spy_pcr, delta="Bearish" if spy_pcr > 1.0 else ("Bullish" if spy_pcr < 0.7 else "Neutral"), delta_color="inverse")
                    spy_avg_iv = spy_summ['averageIV']
                    m2.metric("Average IV", f"{spy_avg_iv:.2%}", "Fear Gauge")
                    spy_vol = spy_summ['totalVolume']
                    m3.metric("Total Volume", f"{spy_vol:,}")
                else:
                    st.info("SPY comparison data not available")
        
        # --- Main Chart: Price Action with Support & Resistance ---
        with st.container(border=True):
            st.markdown("#### 📈 Price Action vs. Support & Resistance Levels")
            with st.expander("How to Use This Chart"):
                st.markdown("""
                This chart shows the stock's price action with key support and resistance levels derived from options market data.
                - **Support (Put Wall):** The strike price with the highest put open interest, often acting as a floor.
                - **Resistance (Call Wall):** The strike price with the highest call open interest, often acting as a ceiling.
                - **What to Look For:** Price behavior around these levels can indicate potential reversals or breakouts.
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

                # Add historical data trace
                fig_map.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['price'], mode='lines', name='Historical Price', line=dict(color='white', width=2)))

                # Add Support and Resistance Lines
                if call_wall is not None:
                    fig_map.add_hline(y=call_wall, line_dash="dot", line_color="red", annotation_text=f"Resistance (Call Wall): ${call_wall:,.0f}")
                if put_wall is not None:
                    fig_map.add_hline(y=put_wall, line_dash="dot", line_color="green", annotation_text=f"Support (Put Wall): ${put_wall:,.0f}")
                
                fig_map.update_layout(title="Historical Price vs. Options-Derived Support & Resistance", template=chart_template, hovermode="x unified", height=500)
                st.plotly_chart(fig_map, use_container_width=True)
        
        # --- Volatility Analysis Section ---
        with st.container(border=True):
            st.markdown("#### 📊 Volatility Analysis")
            if "greeks" in data and data["greeks"]:
                df_greeks = pd.DataFrame(data["greeks"])
                spot_price = float(data['raw_x'][0]) if data['raw_x'] else 400.0
                expirations = sorted(df_greeks['expiry'].unique())
                
                # Expiration range selector for 2D heatmap
                col_vol1, col_vol2 = st.columns(2)
                with col_vol1:
                    start_exp_idx = st.selectbox("Start Expiration", range(len(expirations)), format_func=lambda x: expirations[x], key="vol_start_exp", index=0)
                with col_vol2:
                    end_exp_idx = st.selectbox("End Expiration", range(len(expirations)), format_func=lambda x: expirations[x], key="vol_end_exp", index=min(4, len(expirations)-1))
                
                selected_expirations = expirations[start_exp_idx:end_exp_idx+1]
                
                # 3D Volatility Surface
                if "mesh_z" in data and data["mesh_z"]:
                    st.markdown("**3D Volatility Surface**")
                    fig_surf = go.Figure(data=[go.Surface(z=data['mesh_z'], x=data['mesh_x'], y=data['mesh_y'], colorscale='Reds', cmin=0, opacity=0.9, colorbar=dict(title='IV'), lighting=dict(ambient=0.5, diffuse=0.5))])
                    fig_surf.update_layout(title='3D Volatility Surface: Find Cheap Options in Valleys', template=chart_template, height=600, scene=dict(xaxis_title='Strike', yaxis_title='Days to Expiry', zaxis_title='IV'))
                    st.plotly_chart(fig_surf, use_container_width=True)
                
                # 2D Heatmap for selected expiration range
                st.markdown(f"**2D Volatility Heatmap (Range: {selected_expirations[0]} to {selected_expirations[-1]})**")
                filtered_df = df_greeks[df_greeks['expiry'].isin(selected_expirations)].copy()
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
                            
                            row_calls.append(call_iv if call_iv else None)
                            row_puts.append(put_iv if put_iv else None)
                        
                        iv_matrix_calls.append(row_calls)
                        iv_matrix_puts.append(row_puts)
                    
                    # Create two heatmaps side by side
                    heatmap_col1, heatmap_col2 = st.columns(2)
                    
                    with heatmap_col1:
                        fig_heatmap_calls = go.Figure(data=go.Heatmap(
                            z=iv_matrix_calls,
                            x=[exp[:10] for exp in all_expirations],  # Shorten date format
                            y=[f"${s:.0f}" for s in all_strikes],
                            colorscale='Reds',
                            colorbar=dict(title='Call IV'),
                            text=[[f"{val:.2%}" if val else "" for val in row] for row in iv_matrix_calls],
                            texttemplate='%{text}',
                            textfont={"size": 8},
                            hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>IV: %{z:.2%}<extra></extra>'
                        ))
                        fig_heatmap_calls.update_layout(
                            title='Call IV Heatmap',
                            template=chart_template,
                            height=600,
                            yaxis_title='Strike Price',
                            xaxis_title='Expiration Date'
                        )
                        st.plotly_chart(fig_heatmap_calls, use_container_width=True)
                    
                    with heatmap_col2:
                        fig_heatmap_puts = go.Figure(data=go.Heatmap(
                            z=iv_matrix_puts,
                            x=[exp[:10] for exp in all_expirations],
                            y=[f"${s:.0f}" for s in all_strikes],
                            colorscale='Reds',
                            colorbar=dict(title='Put IV'),
                            text=[[f"{val:.2%}" if val else "" for val in row] for row in iv_matrix_puts],
                            texttemplate='%{text}',
                            textfont={"size": 8},
                            hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>IV: %{z:.2%}<extra></extra>'
                        ))
                        fig_heatmap_puts.update_layout(
                            title='Put IV Heatmap',
                            template=chart_template,
                            height=600,
                            yaxis_title='Strike Price',
                            xaxis_title='Expiration Date'
                        )
                        st.plotly_chart(fig_heatmap_puts, use_container_width=True)
        
        # --- Liquidity Walls Section ---
        with st.container(border=True):
            st.markdown("#### 💧 Liquidity Walls: Open Interest & Volume")
            with st.expander("📚 Understanding Open Interest vs Volume", expanded=True):
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
                """)
            
            if "greeks" in data and data["greeks"]:
                df_greeks = pd.DataFrame(data["greeks"])
                expirations = sorted(df_greeks['expiry'].unique())
                
                # Expiration selector
                selected_exp_oi = st.selectbox("Select Expiration", expirations, key="oi_expiry")
                
                # Fetch OI data for selected expiration
                if st.session_state['summary'] and 'availableExpirations' in st.session_state['summary']:
                    try:
                        resp_oi = requests.get(f"{API_URL}/ticker/{ticker}/open-interest", params={"expiration_date": selected_exp_oi})
                        if resp_oi.status_code == 200:
                            oi_json = resp_oi.json()
                            oi_data = pd.DataFrame(oi_json['data'])
                        else:
                            oi_data = st.session_state.get('oi_data', pd.DataFrame())
                    except:
                        oi_data = st.session_state.get('oi_data', pd.DataFrame())
                else:
                    oi_data = st.session_state.get('oi_data', pd.DataFrame())
                
                if oi_data is not None and not oi_data.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Open Interest by Strike**")
                        fig_oi = go.Figure()
                        fig_oi.add_trace(go.Bar(x=oi_data['strike'], y=oi_data['callOpenInterest'], name='Call OI', marker_color='green'))
                        fig_oi.add_trace(go.Bar(x=oi_data['strike'], y=-oi_data['putOpenInterest'], name='Put OI', marker_color='red'))
                        fig_oi.update_layout(barmode='overlay', title=f"Open Interest Walls ({selected_exp_oi})", template=chart_template, height=400, yaxis_title='Open Interest')
                        st.plotly_chart(fig_oi, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Volume by Strike**")
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Bar(x=oi_data['strike'], y=oi_data.get('callVolume', [0]*len(oi_data)), name='Call Volume', marker_color='lightgreen'))
                        fig_vol.add_trace(go.Bar(x=oi_data['strike'], y=-oi_data.get('putVolume', [0]*len(oi_data)), name='Put Volume', marker_color='lightcoral'))
                        fig_vol.update_layout(barmode='overlay', title=f"Volume ({selected_exp_oi})", template=chart_template, height=400, yaxis_title='Volume')
                        st.plotly_chart(fig_vol, use_container_width=True)
    
    # ==============================================================================
    # ADVANCED CHARTS TAB
    # ==============================================================================
    with tab_advanced:
        st.subheader("🔬 Advanced Analytics")
        st.info("Specialized charts for deep analysis. Use these tools to understand risks and dealer positioning.")
        
        if "greeks" in data and data["greeks"]:
            df_greeks = pd.DataFrame(data["greeks"])
            spot_price = float(data['raw_x'][0]) if data['raw_x'] else 400.0
            
            # --- Greeks Lab ---
            with st.container(border=True):
                st.markdown("#### 🔬 Greeks Slice Analysis")
                with st.expander("How to Use This Chart"):
                    st.markdown("""
                    The Greeks measure the risk of a specific options contract.
                    - **Delta:** Measures direction risk. A Delta of 0.50 means the option price will move $0.50 for every $1.00 the stock moves.
                    - **Gamma:** Measures acceleration risk. High Gamma means your Delta will change very quickly, making the position's value swing violently.
                    - **Theta:** Measures time risk. This is how much money the option loses each day due to time decay. If you buy an option, you want Theta to be as low as possible.
                    """)
                c1, c2 = st.columns(2)
                with c1:
                    expirations = sorted(df_greeks['expiry'].unique())
                    selected_expiry = st.selectbox("Select Expiration", expirations, key="greeks_expiry")
                with c2:
                    selected_type = st.radio("Option Type", ["call", "put"], horizontal=True, key="greeks_type")
                
                filtered_df = df_greeks[(df_greeks['expiry'] == selected_expiry) & (df_greeks['type'] == selected_type)].sort_values('strike')
                
                if not filtered_df.empty:
                    col_g1, col_g2, col_g3 = st.columns(3)
                    with col_g1:
                        fig_delta = go.Figure(go.Scatter(x=filtered_df['strike'], y=filtered_df['delta'], mode='lines', name='Delta'))
                        fig_delta.update_layout(title="Delta", template=chart_template)
                        st.plotly_chart(fig_delta, use_container_width=True)
                    with col_g2:
                        fig_gamma = go.Figure(go.Scatter(x=filtered_df['strike'], y=filtered_df['gamma'], mode='lines', name='Gamma'))
                        fig_gamma.update_layout(title="Gamma", template=chart_template)
                        st.plotly_chart(fig_gamma, use_container_width=True)
                    with col_g3:
                        fig_theta = go.Figure(go.Bar(x=filtered_df['strike'], y=filtered_df['theta'], name='Theta'))
                        fig_theta.update_layout(title="Theta", template=chart_template)
                        st.plotly_chart(fig_theta, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            # --- P&L Simulator ---
            with st.container(border=True):
                st.markdown("#### 📈 P&L Simulator for a *Hypothetical* Trade")
                with st.expander("How to Use This Chart"):
                    st.markdown("""
                    This is your final sanity check. Model a trade to see its potential profit or loss under different scenarios.
                    - **What to Look For:** The difference between the **curved line (P&L Today)** and the **straight line (P&L at Expiration)**.
                    - **Key Insight:** The gap between the lines represents "Time Value." Your goal is for the stock to move enough for the curved line to get above zero *before* it decays and sinks towards the straight line. If a small stock move still results in a loss today, you might need a bigger move or a contract with less time decay.
                    """)
                
                col_strat1, col_strat2 = st.columns(2)
                with col_strat1:
                    strat_type = st.selectbox("Strategy Type", ["Long Call", "Long Put"])
                    strike_price = st.number_input("Strike Price", value=spot_price)
                    contract_cost = st.number_input("Option Premium ($)", value=5.0)
                
                with col_strat2:
                    expirations_sim = sorted(df_greeks['expiry'].unique())
                    selected_expiry_sim = st.selectbox("Select Expiration Date", expirations_sim, key="sim_expiry")
                    days_to_go = (datetime.strptime(selected_expiry_sim, '%Y-%m-%d') - datetime.now()).days
                    st.metric("Days to Expiration", f"{days_to_go} days")
                    current_iv = df_greeks.iloc[(df_greeks['strike'] - spot_price).abs().argsort()[:1]]['iv'].values[0]
                    iv_sim = st.slider("Implied Volatility (%)", 10, 100, int(current_iv * 100)) / 100.0
                
                if strat_type == "Long Call":
                    breakeven = strike_price + contract_cost
                    st.metric("Breakeven Price at Expiration", f"${breakeven:.2f}", f"Stock must be above this.")
                else:
                    breakeven = strike_price - contract_cost
                    st.metric("Breakeven Price at Expiration", f"${breakeven:.2f}", f"Stock must be below this.")

                def bsm_price(S, K, T, r, sigma, type='call'):
                    if T <= 0: return max(0, S-K) if type == 'call' else max(0, K-S)
                    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma*np.sqrt(T)
                    if type == 'call': return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
                    else: return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

                spot_range = np.linspace(strike_price * 0.8, strike_price * 1.2, 100)
                pnl_expiration, pnl_today = [], []
                r_rate = 0.045
                t_years = max(days_to_go, 0) / 365.0
                
                for s in spot_range:
                    if strat_type == "Long Call":
                        val_exp = max(0, s - strike_price) - contract_cost
                        val_today = bsm_price(s, strike_price, t_years, r_rate, iv_sim, 'call') - contract_cost
                    else:
                        val_exp = max(0, strike_price - s) - contract_cost
                        val_today = bsm_price(s, strike_price, t_years, r_rate, iv_sim, 'put') - contract_cost
                    pnl_expiration.append(val_exp)
                    pnl_today.append(val_today)
                    
                fig_strat = go.Figure()
                fig_strat.add_trace(go.Scatter(x=spot_range, y=pnl_expiration, name='At Expiration', line=dict(color='rgb(31, 119, 180)', width=2), fill='tozeroy'))
                fig_strat.add_trace(go.Scatter(x=spot_range, y=pnl_today, name='Today (T+0)', line=dict(color='rgb(255, 127, 14)', width=2, dash='dot')))
                fig_strat.add_hline(y=0, line_dash="solid", line_color="white" if st.session_state["dark_mode"] else "black", line_width=1)
                fig_strat.update_layout(title=f"P&L: {strat_type} @ ${strike_price} (Exp: {selected_expiry_sim})", xaxis_title="Stock Price", yaxis_title="P&L ($)", hovermode="x unified", template=chart_template)
                st.plotly_chart(fig_strat, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- GEX Profile ---
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("#### 🏗️ Full Net GEX Profile")
                    with st.expander("How to Use This Chart"):
                        st.markdown("""
                        This chart shows the total Gamma Exposure held by dealers at each strike. It reveals how market makers are positioned and how they are likely to hedge.
                        - **What to Look For:** The "Call Wall" (largest green bar) and "Put Wall" (largest red bar), and the "Zero Gamma" level where the net exposure flips.
                        - **Interpretation:** When the market is in a **Positive Gamma** regime (net GEX is positive), dealers hedge by buying dips and selling rips, which dampens volatility. In a **Negative Gamma** regime, they do the opposite, which accelerates moves and increases volatility. The Call/Put walls often act as major resistance/support.
                        """)
                    gex_profile = calculate_gex_profile(df_greeks, spot_price)
                    if not gex_profile.empty:
                        colors = ['green' if x > 0 else 'red' for x in gex_profile['gex']]
                        fig_gex = go.Figure(go.Bar(x=gex_profile['gex'], y=gex_profile['strike'], orientation='h', marker_color=colors, name='Net GEX'))
                        fig_gex.add_vline(x=0, line_color="white" if st.session_state["dark_mode"] else "black")
                        call_wall_idx = gex_profile['gex'].idxmax()
                        put_wall_idx = gex_profile['gex'].idxmin()
                        fig_gex.add_hline(y=gex_profile.loc[call_wall_idx, 'strike'], line_dash="dot", line_color="green", annotation_text="Call Wall")
                        fig_gex.add_hline(y=gex_profile.loc[put_wall_idx, 'strike'], line_dash="dot", line_color="red", annotation_text="Put Wall")
                        fig_gex.update_layout(title="Dealer Positioning", height=600, template=chart_template)
                        st.plotly_chart(fig_gex, use_container_width=True)

            with c2:
                with st.container(border=True):
                    st.markdown("#### 🌊 Open Interest Walls")
                    with st.expander("How to Use This Chart"):
                        st.markdown("""
                        This chart shows the raw Open Interest for the nearest expiration date, indicating where the most contracts are currently open.
                        - **What to Look For:** The largest Call (green) and Put (red) bars.
                        - **Interpretation:** These high OI strikes act as significant support and resistance levels, often pinning the price as expiration approaches, a phenomenon known as "max pain."
                        """)
                    if st.session_state['oi_data'] is not None and not st.session_state['oi_data'].empty:
                        oi_data = st.session_state['oi_data']
                        fig_oi = go.Figure()
                        fig_oi.add_trace(go.Bar(x=oi_data['strike'], y=oi_data['callOpenInterest'], name='Calls', marker_color='green'))
                        fig_oi.add_trace(go.Bar(x=oi_data['strike'], y=-oi_data['putOpenInterest'], name='Puts', marker_color='red'))
                        fig_oi.update_layout(barmode='overlay', title="Liquidity Walls (Nearest Expiration)", height=600, template=chart_template)
                        st.plotly_chart(fig_oi, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- Vanna & Charm Profiles ---
            c3, c4 = st.columns(2)
            with c3:
                with st.container(border=True):
                    st.markdown("#### ✨ Vanna Profile (Volatility Flow)")
                    with st.expander("How to Use This Chart"):
                        st.markdown("""
                        Vanna measures how much dealers have to buy or sell stock as **Implied Volatility (IV)** changes. It's a key risk for dealers.
                        - **What to Look For:** Large positive or negative bars, especially away from the current stock price.
                        - **Interpretation:** If there is large negative Vanna exposure, and IV drops (e.g., after an earnings event), dealers will be forced to buy back their hedges, pushing the stock price up. This is known as "Vanna Charm."
                        """)
                    df_greeks_temp = df_greeks.copy()
                    df_greeks_temp['vanna'] = pd.to_numeric(df_greeks_temp['vanna'], errors='coerce').fillna(0)
                    df_greeks_temp['vanna_ex'] = df_greeks_temp['vanna'] * df_greeks_temp['openInterest'] * 100
                    vanna_profile = df_greeks_temp.groupby('strike')['vanna_ex'].sum().reset_index()
                    fig_vanna = go.Figure(go.Bar(x=vanna_profile['strike'], y=vanna_profile['vanna_ex'], marker_color='purple', name='Net Vanna'))
                    fig_vanna.update_layout(title="Impact of IV Changes on Dealer Hedging", template=chart_template, height=400)
                    st.plotly_chart(fig_vanna, use_container_width=True)
            
            with c4:
                with st.container(border=True):
                    st.markdown("#### ⏳ Charm Profile (Time Flow)")
                    with st.expander("How to Use This Chart"):
                        st.markdown("""
                        Charm measures how much dealers have to buy or sell stock as **time passes**. It's the "time decay of Delta."
                        - **What to Look For:** Whether the net Charm is positive or negative.
                        - **Interpretation:** If Charm is highly positive, dealers will have to buy stock as the day goes on simply due to time decay, creating a natural upward drift into the market close. This is often called "Charm pinning."
                        """)
                    df_greeks_temp2 = df_greeks.copy()
                    df_greeks_temp2['charm'] = pd.to_numeric(df_greeks_temp2['charm'], errors='coerce').fillna(0)
                    df_greeks_temp2['charm_ex'] = df_greeks_temp2['charm'] * df_greeks_temp2['openInterest'] * 100
                    charm_profile = df_greeks_temp2.groupby('strike')['charm_ex'].sum().reset_index()
                    fig_charm = go.Figure(go.Bar(x=charm_profile['strike'], y=charm_profile['charm_ex'], marker_color='teal', name='Net Charm'))
                    fig_charm.update_layout(title="Impact of Time Decay on Dealer Hedging", template=chart_template, height=400)
                    st.plotly_chart(fig_charm, use_container_width=True)

else:
    st.info("Please enter a ticker symbol and click 'Analyze' to begin.")

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

def generate_probability_cone_data(spot_price, current_iv, days=60):
    """
    Generates data for Probability Cone (1SD and 2SD).
    """
    days_future = np.arange(1, days)
    
    upper_1sd = spot_price * (1 + current_iv * np.sqrt(days_future/365))
    lower_1sd = spot_price * (1 - current_iv * np.sqrt(days_future/365))
    
    upper_2sd = spot_price * (1 + 2 * current_iv * np.sqrt(days_future/365))
    lower_2sd = spot_price * (1 - 2 * current_iv * np.sqrt(days_future/365))
    
    dates_future = [datetime.now() + timedelta(days=int(d)) for d in days_future]
    
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
                
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# --- MAIN CONTENT ---
if st.session_state['data']:
    data = st.session_state['data']
    chart_template = "plotly_dark" if st.session_state["dark_mode"] else "plotly"
    
    # --- NAVIGATION (Martini Glass Structure) ---
    view_options = [
        "1. Market Command Center",
        "2. Deep Dive: Structure & Flow",
        "3. Deep Dive: Volatility Dynamics",
        "4. Strategy Workbench"
    ]
    
    # Initialize view state if not present or invalid
    if "current_view" not in st.session_state or st.session_state["current_view"] not in view_options:
        st.session_state["current_view"] = view_options[0]
        
    selected_view = st.radio(
        "Analysis Module", 
        view_options, 
        horizontal=True, 
        label_visibility="collapsed",
        key="current_view"
    )
    
    st.divider()
    
    # ==============================================================================
    # MODULE 1: MARKET COMMAND CENTER (Overview)
    # ==============================================================================
    if selected_view == "1. Market Command Center":
        st.subheader("🚀 Market Command Center")
        
        # --- SECTION A: MARKET PULSE (Sentiment) ---
        if st.session_state['summary']:
            summ = st.session_state['summary']
            with st.container(border=True):
                st.markdown("**📡 Market Pulse**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    pcr = summ['putCallRatio']
                    st.metric("Put/Call Ratio", pcr, 
                            delta="Bearish" if pcr > 1.0 else ("Bullish" if pcr < 0.7 else "Neutral"),
                            delta_color="inverse")
                with c2:
                    avg_iv = summ['averageIV']
                    st.metric("Average IV", f"{avg_iv:.2%}", "Volatility Level")
                with c3:
                    vol = summ['totalVolume']
                    st.metric("Total Volume", f"{vol:,}", "Contracts Traded")

        if "greeks" in data and data["greeks"]:
            df_greeks = pd.DataFrame(data["greeks"])
            spot_price = float(data['raw_x'][0]) if data['raw_x'] else 400.0
            
            # --- SECTION B: REGIME (Structure) ---
            gex_profile = calculate_gex_profile(df_greeks, spot_price)
            flip_level = calculate_zero_gamma(df_greeks, spot_price)
            total_gex = gex_profile['gex'].sum() if not gex_profile.empty else 0
            
            if not gex_profile.empty:
                call_wall_idx = gex_profile['gex'].idxmax()
                put_wall_idx = gex_profile['gex'].idxmin()
                call_wall = gex_profile.loc[call_wall_idx, 'strike']
                put_wall = gex_profile.loc[put_wall_idx, 'strike']
            else:
                call_wall, put_wall = None, None

            with st.container(border=True):
                st.markdown("**🛡️ Gamma Regime & Key Levels**")
                m1, m2, m3, m4 = st.columns(4)
                
                if total_gex > 0:
                    regime_color = "normal" # Green
                    regime_label = "Long Gamma (Buy Dips)"
                else:
                    regime_color = "inverse" # Red
                    regime_label = "Short Gamma (Sell Rallies)"
                    
                m1.metric("Market Regime", "Positive" if total_gex > 0 else "Negative", regime_label, delta_color=regime_color)
                m2.metric("Zero Gamma Flip", f"${flip_level:.2f}" if flip_level else "N/A", "Risk Level")
                m3.metric("Net GEX", f"${total_gex/1e6:.1f}M", "Dealer Exposure")
                m4.metric("Trading Range", f"${put_wall:.0f} - ${call_wall:.0f}", "Put/Call Walls")

            # --- SECTION C: STRUCTURAL MAP (Forecast + Walls) ---
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.markdown("#### 🔮 Price Forecast & Structural Walls")
                
                # Get IV
                if not df_greeks.empty:
                     nearest_exp = sorted(df_greeks['expiry'].unique())[0]
                     nearest_df = df_greeks[df_greeks['expiry'] == nearest_exp]
                     atm_strike = nearest_df.iloc[(nearest_df['strike'] - spot_price).abs().argsort()[:1]]
                     current_iv = atm_strike['iv'].values[0] if not atm_strike.empty else 0.20
                else:
                     current_iv = 0.20

                # Generate Cone
                dates_f, u1, l1, u2, l2 = generate_probability_cone_data(spot_price, current_iv)
                
                fig_map = go.Figure()
                
                # 2SD
                fig_map.add_trace(go.Scatter(x=dates_f + dates_f[::-1], y=np.concatenate([u2, l2[::-1]]), fill='toself', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(color='rgba(255,0,0,0)'), name='95% Risk Zone'))
                # 1SD
                fig_map.add_trace(go.Scatter(x=dates_f + dates_f[::-1], y=np.concatenate([u1, l1[::-1]]), fill='toself', fillcolor='rgba(0, 255, 0, 0.2)', line=dict(color='rgba(0,255,0,0)'), name='68% Noise Zone'))
                # Spot
                fig_map.add_trace(go.Scatter(x=[dates_f[0]], y=[spot_price], mode='markers', marker=dict(color='white', size=10), name='Spot'))
                
                # Add Walls
                if call_wall:
                    fig_map.add_hline(y=call_wall, line_dash="dash", line_color="green", annotation_text="Call Wall (Res)")
                if put_wall:
                    fig_map.add_hline(y=put_wall, line_dash="dash", line_color="red", annotation_text="Put Wall (Sup)")
                
                fig_map.update_layout(title="Projected Path vs Dealer Walls", template=chart_template, hovermode="x unified", height=500)
                st.plotly_chart(fig_map, width="stretch" if True else "content")

            with c2:
                st.markdown("#### 🏗️ Net GEX Profile")
                # Render GEX Histogram (Condensed)
                if not gex_profile.empty:
                    colors = ['green' if x > 0 else 'red' for x in gex_profile['gex']]
                    fig_gex_mini = go.Figure(go.Bar(
                        x=gex_profile['gex'],
                        y=gex_profile['strike'],
                        orientation='h',
                        marker_color=colors
                    ))
                    fig_gex_mini.update_layout(
                        title="Net Gamma Exposure", 
                        xaxis_title="Net GEX", 
                        template=chart_template, 
                        height=500,
                        yaxis=dict(range=[put_wall*0.9, call_wall*1.1]) if put_wall and call_wall else None
                    )
                    st.plotly_chart(fig_gex_mini, width="stretch" if True else "content")

            # --- SECTION D: FLOW RADAR (Clusters) ---
            st.markdown("#### 📡 Order Flow Radar")
            df_vol = df_greeks[df_greeks['volume'] > 0]
            if not df_vol.empty:
                 fig_radar = go.Figure()
                 for opt_type, color in [('call', 'green'), ('put', 'red')]:
                     subset = df_vol[df_vol['type'] == opt_type]
                     if not subset.empty:
                         fig_radar.add_trace(go.Scatter(
                             x=subset['expiry'], y=subset['strike'], mode='markers', name=f'{opt_type.title()}s',
                             marker=dict(size=subset['volume'], sizemode='area', sizeref=2.0 * max(df_vol['volume']) / (20**2), sizemin=2, color=color, opacity=0.6),
                             customdata=subset['volume'], hovertemplate='<b>Date</b>: %{x}<br><b>Strike</b>: $%{y}<br><b>Vol</b>: %{customdata:,}<extra></extra>'
                         ))
                 fig_radar.update_layout(title="Real-time Kinetic Flow (Volume Clusters)", xaxis_title="Expiration", yaxis_title="Strike", height=400, template=chart_template)
                 st.plotly_chart(fig_radar, width="stretch" if True else "content")
            else:
                st.info("No volume data available.")

    # ==============================================================================
    # MODULE 2: DEEP DIVE - STRUCTURE & FLOW
    # ==============================================================================
    elif selected_view == "2. Deep Dive: Structure & Flow":
        st.subheader("🔍 Deep Dive: Market Structure & Order Flow")
        st.caption("Analyze Dealer Positioning (GEX), Liquidity (OI), and Hedging Flows (Vanna/Charm).")
        
        if "greeks" in data and data["greeks"]:
            df_greeks = pd.DataFrame(data["greeks"])
            spot_price = float(data['raw_x'][0]) if data['raw_x'] else 400.0
            
            # --- ROW 1: POSITIONING (GEX & OI) ---
            c1, c2 = st.columns([3, 2])
            
            with c1:
                st.markdown("#### 🏗️ Net Gamma Exposure (GEX)")
                gex_profile = calculate_gex_profile(df_greeks, spot_price)
                if not gex_profile.empty:
                    colors = ['green' if x > 0 else 'red' for x in gex_profile['gex']]
                    fig_gex = go.Figure()
                    fig_gex.add_trace(go.Bar(x=gex_profile['gex'], y=gex_profile['strike'], orientation='h', marker_color=colors, name='Net GEX'))
                    fig_gex.add_vline(x=0, line_color="white" if st.session_state["dark_mode"] else "black")
                    
                    # Walls annotations
                    call_wall_idx = gex_profile['gex'].idxmax()
                    put_wall_idx = gex_profile['gex'].idxmin()
                    
                    fig_gex.add_hline(y=gex_profile.loc[call_wall_idx, 'strike'], line_dash="dot", line_color="green", annotation_text="Call Wall")
                    fig_gex.add_hline(y=gex_profile.loc[put_wall_idx, 'strike'], line_dash="dot", line_color="red", annotation_text="Put Wall")
                    
                    fig_gex.update_layout(title="Net GEX Profile", xaxis_title="Net GEX ($)", yaxis_title="Strike", height=600, template=chart_template)
                    st.plotly_chart(fig_gex, width="stretch" if True else "content")
            
            with c2:
                st.markdown("#### 🌊 Open Interest Walls")
                if st.session_state['oi_data'] is not None and not st.session_state['oi_data'].empty:
                    oi_data = st.session_state['oi_data']
                    fig_oi = go.Figure()
                    fig_oi.add_trace(go.Bar(x=oi_data['strike'], y=oi_data['callOpenInterest'], name='Calls OI', marker_color='green'))
                    fig_oi.add_trace(go.Bar(x=oi_data['strike'], y=-oi_data['putOpenInterest'], name='Puts OI', marker_color='red'))
                    fig_oi.update_layout(barmode='overlay', title="Open Interest Distribution (Nearest)", xaxis_title="Strike", yaxis_title="OI", template=chart_template, height=600)
                    st.plotly_chart(fig_oi, width="stretch" if True else "content")
            
            st.divider()
            
            # --- ROW 2: ADVANCED GREEKS (FLOWS) ---
            st.markdown("#### 🌊 Second-Order Flows (Vanna & Charm)")
            st.caption("Predict dealer hedging pressure. **Vanna** = Volatility Sensitivity. **Charm** = Time Decay Sensitivity.")
            
            t1, t2 = st.columns(2)
            
            with t1:
                # Vanna Profile
                df_greeks['vanna'] = pd.to_numeric(df_greeks['vanna'], errors='coerce').fillna(0)
                df_greeks['vanna_ex'] = df_greeks['vanna'] * df_greeks['openInterest'] * 100 * spot_price * 0.01 
                vanna_profile = df_greeks.groupby('strike')['vanna_ex'].sum().reset_index()
                
                fig_vanna = go.Figure(go.Bar(x=vanna_profile['strike'], y=vanna_profile['vanna_ex'], marker_color='purple', name='Net Vanna'))
                fig_vanna.update_layout(title="Net Vanna Exposure (Vol Flows)", xaxis_title="Strike", yaxis_title="Net Vanna ($)", template=chart_template)
                st.plotly_chart(fig_vanna, width="stretch" if True else "content")
                
            with t2:
                # Charm Heatmap
                df_greeks['charm'] = pd.to_numeric(df_greeks['charm'], errors='coerce').fillna(0)
                fig_charm = go.Figure(data=go.Heatmap(
                    z=df_greeks['charm'], x=df_greeks['expiry'], y=df_greeks['strike'],
                    colorscale='RdBu', zmid=0, colorbar=dict(title="Charm")
                ))
                fig_charm.update_layout(title="Charm Exposure (Time Flows)", xaxis_title="Expiry", yaxis_title="Strike", template=chart_template)
                st.plotly_chart(fig_charm, width="stretch" if True else "content")

            # --- ROW 3: LIQUIDITY MAP ---
            st.divider()
            st.markdown("#### 💧 Liquidity Heatmap")
            
            # Filter out zero OI
            df_greeks_oi = df_greeks[df_greeks['openInterest'] > 0]
            
            if use_log_scale:
                color_values = np.log1p(df_greeks_oi['openInterest'])
                colorbar_title = "OI (Log)"
            else:
                color_values = df_greeks_oi['openInterest']
                colorbar_title = "OI"
                
            sizeref = 2.0 * max(color_values) / (30**2)
            
            fig_heat = go.Figure(go.Scatter(
                x=df_greeks_oi['expiry'], y=df_greeks_oi['strike'], mode='markers',
                marker=dict(
                    size=color_values, sizemode='area', sizeref=sizeref, sizemin=2,
                    color=color_values, colorscale='Cividis', showscale=True, colorbar=dict(title=colorbar_title), opacity=0.8
                ),
                customdata=df_greeks_oi['openInterest'],
                hovertemplate='<b>Date</b>: %{x}<br><b>Strike</b>: $%{y}<br><b>OI</b>: %{customdata:,}<extra></extra>'
            ))
            fig_heat.update_layout(title='Open Interest Liquidity Map', xaxis_title='Expiry', yaxis_title='Strike', height=600, template=chart_template)
            st.plotly_chart(fig_heat, width="stretch" if True else "content")

    # ==============================================================================
    # MODULE 3: DEEP DIVE - VOLATILITY DYNAMICS
    # ==============================================================================
    elif selected_view == "3. Deep Dive: Volatility Dynamics":
        st.subheader("📉 Deep Dive: Volatility Surface & Greeks")
        
        if "mesh_z" in data and data["mesh_z"]:
            # 1. Surface
            fig_surf = go.Figure(data=[go.Surface(
                z=data['mesh_z'], x=data['mesh_x'], y=data['mesh_y'],
                colorscale='Reds', cmin=0, opacity=0.9,
                colorbar=dict(title='Implied Volatility'), lighting=dict(ambient=0.5, diffuse=0.5)
            )])
            fig_surf.update_layout(
                title='Implied Volatility Surface', template=chart_template, height=700,
                scene=dict(xaxis_title='Strike', yaxis_title='Days to Expiry', zaxis_title='IV')
            )
            st.plotly_chart(fig_surf, width="stretch" if True else "content")
            
            st.divider()
            
            # 2. Greeks Lab
            st.markdown("#### 🔬 Greeks Slice Analysis")
            if "greeks" in data and data["greeks"]:
                df_greeks = pd.DataFrame(data["greeks"])
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
                        fig_delta = go.Figure(go.Scatter(x=filtered_df['strike'], y=filtered_df['delta'], mode='lines', name='Delta', line=dict(color='blue')))
                        fig_delta.update_layout(title=f"Delta ({selected_expiry})", xaxis_title="Strike", yaxis_title="Delta", template=chart_template)
                        st.plotly_chart(fig_delta, width="stretch" if True else "content")
                    with col_g2:
                        fig_gamma = go.Figure(go.Scatter(x=filtered_df['strike'], y=filtered_df['gamma'], mode='lines', name='Gamma', line=dict(color='orange')))
                        fig_gamma.update_layout(title=f"Gamma ({selected_expiry})", xaxis_title="Strike", yaxis_title="Gamma", template=chart_template)
                        st.plotly_chart(fig_gamma, width="stretch" if True else "content")
                    with col_g3:
                        fig_theta = go.Figure(go.Bar(x=filtered_df['strike'], y=filtered_df['theta'], name='Theta', marker_color='red'))
                        fig_theta.update_layout(title=f"Theta ({selected_expiry})", xaxis_title="Strike", yaxis_title="Theta", template=chart_template)
                        st.plotly_chart(fig_theta, width="stretch" if True else "content")

    # ==============================================================================
    # MODULE 4: STRATEGY WORKBENCH (Action)
    # ==============================================================================
    elif selected_view == "4. Strategy Workbench":
        st.subheader("🛠️ Strategy Workbench")
        st.caption("Plan your trade: Forecast price bounds and simulate strategy P&L.")
        
        if data and 'raw_x' in data:
            spot_price = float(data['raw_x'][0]) if data['raw_x'] else 400.0
            
            # 1. Forecast Cone
            st.markdown("#### 1. Market Forecast (Probability Cone)")
            
            # Estimate IV
            if 'greeks' in data:
                 df_greeks = pd.DataFrame(data["greeks"])
                 nearest_exp = sorted(df_greeks['expiry'].unique())[0]
                 nearest_df = df_greeks[df_greeks['expiry'] == nearest_exp]
                 atm_strike = nearest_df.iloc[(nearest_df['strike'] - spot_price).abs().argsort()[:1]]
                 current_iv = atm_strike['iv'].values[0] if not atm_strike.empty else 0.20
            else:
                 current_iv = 0.20

            dates_f, u1, l1, u2, l2 = generate_probability_cone_data(spot_price, current_iv)
            
            fig_cone = go.Figure()
            fig_cone.add_trace(go.Scatter(x=dates_f + dates_f[::-1], y=np.concatenate([u2, l2[::-1]]), fill='toself', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(color='rgba(255,0,0,0)'), name='95% (2SD)'))
            fig_cone.add_trace(go.Scatter(x=dates_f + dates_f[::-1], y=np.concatenate([u1, l1[::-1]]), fill='toself', fillcolor='rgba(0, 255, 0, 0.2)', line=dict(color='rgba(0,255,0,0)'), name='68% (1SD)'))
            fig_cone.add_trace(go.Scatter(x=[dates_f[0]], y=[spot_price], mode='markers', marker=dict(color='white', size=10), name='Spot'))
            fig_cone.update_layout(title="Price Forecast (60 Days)", xaxis_title="Date", yaxis_title="Price", template=chart_template, hovermode="x unified", height=400)
            st.plotly_chart(fig_cone, width="stretch" if True else "content")
            
            st.divider()
            
            # 2. Simulator
            st.markdown("#### 2. P&L Simulator")
            
            col_strat1, col_strat2 = st.columns(2)
            with col_strat1:
                strat_type = st.selectbox("Strategy Type", ["Long Call", "Long Put"])
                strike_price = st.number_input("Strike Price", value=spot_price)
                contract_cost = st.number_input("Option Premium ($)", value=5.0)
            
            with col_strat2:
                days_to_go = st.slider("Days to Expiration", 1, 90, 30)
                iv_sim = st.slider("Implied Volatility (%)", 10, 100, 20) / 100.0
            
            # BSM Logic
            def bsm_price(S, K, T, r, sigma, type='call'):
                if T <= 0: return max(0, S-K) if type == 'call' else max(0, K-S)
                d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                if type == 'call':
                    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
                else:
                    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            spot_range = np.linspace(strike_price * 0.8, strike_price * 1.2, 100)
            pnl_expiration = []
            pnl_today = []
            r_rate = 0.045
            t_years = days_to_go / 365.0
            
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
            fig_strat.update_layout(title=f"P&L: {strat_type} @ ${strike_price}", xaxis_title="Stock Price", yaxis_title="P&L ($)", hovermode="x unified", template=chart_template)
            st.plotly_chart(fig_strat, width="stretch" if True else "content")

else:
    st.info("Please enter a ticker symbol and click 'Analyze' to begin.")

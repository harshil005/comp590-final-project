import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Config
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Options Viz", layout="wide")
st.title("Stock Options Visualization Engine")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
analyze_btn = st.sidebar.button("Analyze")

if analyze_btn:
    with st.spinner(f"Fetching data for {ticker}..."):
        
        # 1. Volatility Surface
        try:
            response = requests.get(f"{API_URL}/ticker/{ticker}/volatility-surface")
            if response.status_code == 200:
                data = response.json()
                
                if "mesh_z" in data and data["mesh_z"]:
                    st.subheader(f"Volatility Surface: {ticker}")
                    
                    # Create 3D Surface Plot
                    fig = go.Figure(data=[go.Surface(
                        z=data['mesh_z'],
                        x=data['mesh_x'],
                        y=data['mesh_y'],
                        colorscale='Viridis',
                        opacity=0.9,
                        colorbar=dict(title='Implied Volatility')
                    )])
                    
                    fig.update_layout(
                        title='Implied Volatility Surface',
                        scene=dict(
                            xaxis_title='Strike Price',
                            yaxis_title='Days to Expiration',
                            zaxis_title='Implied Volatility'
                        ),
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data to generate volatility surface. (Check if market is open or ticker is valid)")
            else:
                st.error(f"Failed to fetch volatility data. Status: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

        # 2. Open Interest
        try:
            resp_oi = requests.get(f"{API_URL}/ticker/{ticker}/open-interest")
            if resp_oi.status_code == 200:
                oi_data = pd.DataFrame(resp_oi.json())
                
                if not oi_data.empty:
                    st.subheader("Open Interest Walls (Nearest Expiration)")
                    fig_oi = go.Figure()
                    fig_oi.add_trace(go.Bar(
                        x=oi_data['strike'], 
                        y=oi_data['callOpenInterest'], 
                        name='Calls OI',
                        marker_color='green'
                    ))
                    fig_oi.add_trace(go.Bar(
                        x=oi_data['strike'], 
                        y=-oi_data['putOpenInterest'], 
                        name='Puts OI',
                        marker_color='red'
                    ))
                    fig_oi.update_layout(
                        barmode='overlay', 
                        title="Open Interest Distribution",
                        xaxis_title="Strike Price",
                        yaxis_title="Open Interest",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_oi, use_container_width=True)
                else:
                    st.info("No open interest data found.")
            else:
                st.warning(f"Could not fetch open interest data. Status: {resp_oi.status_code}")
        except Exception as e:
            st.error(f"Error fetching OI data: {e}")




"""
The Market Sentiment Radar
A Senior Streamlit Dashboard with Top-Heavy Layout & Accessibility
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="Market Radar",
    page_icon="📊",  # Keep page_icon for browser tab
    initial_sidebar_state="collapsed"
)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    """Load sentiment data from parquet file with caching"""
    try:
        df = pd.read_parquet('sentiment_data.parquet')
        # Ensure datetime column is properly formatted
        if 'QUOTE_DATE' in df.columns:
            df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
        return df
    except FileNotFoundError:
        st.error("❌ sentiment_data.parquet not found. Please ensure the file exists in the project directory.")
        st.stop()
        return None  # This line won't be reached, but keeps linters happy
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
        return None  # This line won't be reached, but keeps linters happy

# ============================================================================
# MARKET MATH - ADVANCED CALCULATIONS
# ============================================================================
class MarketMath:
    """
    Advanced market analytics calculator for options data

    Handles:
    - Support/Resistance levels (Call/Put Walls)
    - Gamma Exposure (GEX) proxy calculations
    - Volatility-based price projection cones
    """

    def __init__(self, day_df):
        """
        Initialize MarketMath with filtered daily data

        Args:
            day_df (pd.DataFrame): Filtered dataframe for a specific date
        """
        self.df = day_df.copy() if len(day_df) > 0 else pd.DataFrame()
        self.is_valid = len(self.df) > 0

    def get_walls(self):
        """
        Calculate support and resistance levels based on volume concentration

        Returns:
            dict: {
                'call_wall': float,  # Strike with max call volume (ceiling)
                'put_wall': float,   # Strike with max put volume (floor)
                'call_wall_volume': float,
                'put_wall_volume': float
            }
        """
        if not self.is_valid:
            return {
                'call_wall': None,
                'put_wall': None,
                'call_wall_volume': 0,
                'put_wall_volume': 0
            }

        try:
            # Ensure volume columns are numeric
            self.df['C_VOLUME'] = pd.to_numeric(self.df['C_VOLUME'], errors='coerce').fillna(0)
            self.df['P_VOLUME'] = pd.to_numeric(self.df['P_VOLUME'], errors='coerce').fillna(0)

            # Group by strike and sum volumes
            volume_by_strike = self.df.groupby('STRIKE').agg({
                'C_VOLUME': 'sum',
                'P_VOLUME': 'sum'
            }).reset_index()

            # Find call wall (maximum call volume = resistance/ceiling)
            call_wall_idx = volume_by_strike['C_VOLUME'].idxmax()
            call_wall = volume_by_strike.loc[call_wall_idx, 'STRIKE']
            call_wall_volume = volume_by_strike.loc[call_wall_idx, 'C_VOLUME']

            # Find put wall (maximum put volume = support/floor)
            put_wall_idx = volume_by_strike['P_VOLUME'].idxmax()
            put_wall = volume_by_strike.loc[put_wall_idx, 'STRIKE']
            put_wall_volume = volume_by_strike.loc[put_wall_idx, 'P_VOLUME']

            return {
                'call_wall': float(call_wall),
                'put_wall': float(put_wall),
                'call_wall_volume': float(call_wall_volume),
                'put_wall_volume': float(put_wall_volume)
            }

        except Exception as e:
            return {
                'call_wall': None,
                'put_wall': None,
                'call_wall_volume': 0,
                'put_wall_volume': 0,
                'error': str(e)
            }

    def calculate_gex_proxy(self):
        """
        Calculate Gamma Exposure (GEX) proxy
        Net_GEX = C_VOLUME - P_VOLUME for each strike

        Returns:
            pd.DataFrame: DataFrame with columns [STRIKE, NET_GEX, C_VOLUME, P_VOLUME]
        """
        if not self.is_valid:
            return pd.DataFrame(columns=['STRIKE', 'NET_GEX', 'C_VOLUME', 'P_VOLUME'])

        try:
            # Ensure volume columns are numeric
            self.df['C_VOLUME'] = pd.to_numeric(self.df['C_VOLUME'], errors='coerce').fillna(0)
            self.df['P_VOLUME'] = pd.to_numeric(self.df['P_VOLUME'], errors='coerce').fillna(0)

            # Group by strike and calculate net GEX
            gex_df = self.df.groupby('STRIKE').agg({
                'C_VOLUME': 'sum',
                'P_VOLUME': 'sum'
            }).reset_index()

            # Net GEX = Calls - Puts
            gex_df['NET_GEX'] = gex_df['C_VOLUME'] - gex_df['P_VOLUME']

            # Sort by strike
            gex_df = gex_df.sort_values('STRIKE').reset_index(drop=True)

            return gex_df

        except Exception as e:
            return pd.DataFrame(columns=['STRIKE', 'NET_GEX', 'C_VOLUME', 'P_VOLUME'])

    def get_flip_level(self):
        """
        Identify the "Flip Level" where Net_GEX changes from negative to positive
        This indicates a critical price level where market dynamics shift

        Returns:
            dict: {
                'flip_strike': float or None,
                'flip_gex': float,
                'interpretation': str
            }
        """
        gex_df = self.calculate_gex_proxy()

        if len(gex_df) == 0:
            return {
                'flip_strike': None,
                'flip_gex': 0,
                'interpretation': 'No data available'
            }

        try:
            # Find where NET_GEX crosses zero
            negative_gex = gex_df[gex_df['NET_GEX'] < 0]
            positive_gex = gex_df[gex_df['NET_GEX'] > 0]

            if len(negative_gex) == 0:
                # All positive - bullish
                return {
                    'flip_strike': gex_df['STRIKE'].min(),
                    'flip_gex': gex_df['NET_GEX'].min(),
                    'interpretation': 'Bullish (all strikes have positive GEX)'
                }

            if len(positive_gex) == 0:
                # All negative - bearish
                return {
                    'flip_strike': gex_df['STRIKE'].max(),
                    'flip_gex': gex_df['NET_GEX'].max(),
                    'interpretation': 'Bearish (all strikes have negative GEX)'
                }

            # Find the transition point
            max_negative_strike = negative_gex['STRIKE'].max()
            min_positive_strike = positive_gex['STRIKE'].min()

            # Use the midpoint as flip level
            flip_strike = (max_negative_strike + min_positive_strike) / 2

            # Find the GEX value closest to the flip strike
            closest_idx = (gex_df['STRIKE'] - flip_strike).abs().idxmin()
            flip_gex = gex_df.loc[closest_idx, 'NET_GEX']

            return {
                'flip_strike': float(flip_strike),
                'flip_gex': float(flip_gex),
                'interpretation': f'GEX flips at ${flip_strike:.2f} (market neutral zone)'
            }

        except Exception as e:
            return {
                'flip_strike': None,
                'flip_gex': 0,
                'interpretation': f'Error: {str(e)}'
            }

    def get_projection_cone(self, current_price, iv, days=60):
        """
        Calculate volatility-based price projection cone

        Uses standard deviation formula:
        σ = Price × IV × sqrt(T/365)

        Args:
            current_price (float): Current stock price
            iv (float): Implied volatility (as decimal, e.g., 0.25 for 25%)
            days (int): Number of days to project (default 60)

        Returns:
            dict: {
                'days': list,  # [0, 1, 2, ..., days]
                'upper_band': list,  # +1 SD projection
                'lower_band': list,  # -1 SD projection
                'center_line': list  # Current price (no drift assumption)
            }
        """
        if current_price <= 0 or iv <= 0:
            return {
                'days': [],
                'upper_band': [],
                'lower_band': [],
                'center_line': []
            }

        try:
            # Generate day array
            day_array = list(range(days + 1))

            # Initialize arrays
            upper_band = []
            lower_band = []
            center_line = []

            for day in day_array:
                # Time fraction in years
                t = day / 365.0

                # Standard deviation calculation
                # σ = Price × IV × sqrt(T)
                sigma = current_price * iv * (t ** 0.5)

                # Center line (no drift assumption)
                center = current_price

                # Upper and lower bands (±1 standard deviation)
                upper = center + sigma
                lower = max(0, center - sigma)  # Price can't go negative

                upper_band.append(upper)
                lower_band.append(lower)
                center_line.append(center)

            return {
                'days': day_array,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'center_line': center_line
            }

        except Exception as e:
            return {
                'days': [],
                'upper_band': [],
                'lower_band': [],
                'center_line': [],
                'error': str(e)
            }

    def get_summary_stats(self):
        """
        Calculate comprehensive market summary statistics

        Returns:
            dict: Summary of all key metrics
        """
        if not self.is_valid:
            return {
                'status': 'No data available',
                'record_count': 0
            }

        walls = self.get_walls()
        flip = self.get_flip_level()
        gex_df = self.calculate_gex_proxy()

        current_price = self.df['UNDERLYING_LAST'].iloc[0] if 'UNDERLYING_LAST' in self.df.columns else 0
        avg_iv = self.df['P_IV'].mean() if 'P_IV' in self.df.columns else 0

        return {
            'status': 'OK',
            'record_count': len(self.df),
            'current_price': float(current_price),
            'avg_iv': float(avg_iv),
            'call_wall': walls['call_wall'],
            'put_wall': walls['put_wall'],
            'flip_level': flip['flip_strike'],
            'flip_interpretation': flip['interpretation'],
            'total_call_volume': float(self.df['C_VOLUME'].sum()) if 'C_VOLUME' in self.df.columns else 0,
            'total_put_volume': float(self.df['P_VOLUME'].sum()) if 'P_VOLUME' in self.df.columns else 0,
            'net_gex_total': float(gex_df['NET_GEX'].sum()) if len(gex_df) > 0 else 0
        }

# ============================================================================
# ACCESSIBILITY ENGINE
# ============================================================================
def get_theme(mode):
    """
    Returns color dictionary based on selected accessibility mode

    Args:
        mode (str): One of 'Dark Mode', 'Light Mode', 'Colorblind Safe'

    Returns:
        dict: Color configuration with keys: bg, text, call, put, heatmap
    """
    themes = {
        'Dark Mode': {
            'bg': '#0E1117',
            'text': '#FAFAFA',
            'call': '#00cea9',
            'put': '#ff4b4b',
            'heatmap': 'RdBu_r'
        },
        'Light Mode': {
            'bg': '#FFFFFF',
            'text': '#000000',
            'call': '#008000',
            'put': '#d62728',
            'heatmap': 'RdBu_r'
        },
        'Colorblind Safe': {
            'bg': '#FFFFFF',
            'text': '#000000',
            'call': '#0072B2',  # Blue (Okabe-Ito palette)
            'put': '#D55E00',   # Vermilion (Okabe-Ito palette)
            'heatmap': 'Viridis'
        }
    }
    return themes.get(mode, themes['Dark Mode'])

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_heatmap(df, theme):
    """
    Create an implied volatility heatmap with current price annotation

    Args:
        df (pd.DataFrame): Options data with columns: DTE, STRIKE, P_IV, UNDERLYING_LAST
        theme (dict): Theme dictionary with 'heatmap' and 'text' keys

    Returns:
        go.Figure: Plotly heatmap figure
    """
    # Validate required columns
    required_cols = ['DTE', 'STRIKE', 'P_IV', 'UNDERLYING_LAST']
    if not all(col in df.columns for col in required_cols):
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required columns: DTE, STRIKE, P_IV, UNDERLYING_LAST",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

    # Create pivot table for heatmap
    try:
        pivot_df = df.pivot_table(
            values='P_IV',
            index='STRIKE',
            columns='DTE',
            aggfunc='mean'
        )
        pivot_df = pivot_df.sort_index(ascending=False)  # Higher strikes on top
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating pivot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Get current stock price
    current_price = df['UNDERLYING_LAST'].iloc[0] if len(df) > 0 else None

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=theme['heatmap'],
        hovertemplate='DTE: %{x}<br>Strike: %{y}<br>Put IV: %{z:.2%}<extra></extra>',
        colorbar=dict(
            title=dict(text="Put IV", side='right'),
            tickformat='.0%'
        )
    ))

    # Add horizontal line at current price if available
    if current_price is not None:
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color=theme['text'],
            line_width=2,
            annotation_text="Current Price",
            annotation_position="right",
            annotation=dict(
                font_size=12,
                font_color=theme['text']
            )
        )

    # Update layout
    fig.update_layout(
        title="Put Implied Volatility Heatmap",
        xaxis_title="Days to Expiration (DTE)",
        yaxis_title="Strike Price",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=theme['text']),
        height=500,
        margin=dict(l=80, r=80, t=60, b=60)
    )

    return fig


def plot_crowd_bets(df, theme):
    """
    Create diverging bar chart showing call vs put volume by strike

    Args:
        df (pd.DataFrame): Options data with columns: STRIKE, C_VOLUME, P_VOLUME
        theme (dict): Theme dictionary with 'call', 'put', and 'text' keys

    Returns:
        go.Figure: Plotly diverging bar chart
    """
    # Validate required columns
    required_cols = ['STRIKE', 'C_VOLUME', 'P_VOLUME']
    if not all(col in df.columns for col in required_cols):
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required columns: STRIKE, C_VOLUME, P_VOLUME",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

    # Group by strike and sum volumes
    try:
        # Ensure numeric types before aggregation
        df['C_VOLUME'] = pd.to_numeric(df['C_VOLUME'], errors='coerce').fillna(0)
        df['P_VOLUME'] = pd.to_numeric(df['P_VOLUME'], errors='coerce').fillna(0)

        volume_by_strike = df.groupby('STRIKE').agg({
            'C_VOLUME': 'sum',
            'P_VOLUME': 'sum'
        }).reset_index()

        # Sort by strike price
        volume_by_strike = volume_by_strike.sort_values('STRIKE')

        # Make put volume negative for diverging effect
        volume_by_strike['P_VOLUME_NEG'] = -volume_by_strike['P_VOLUME']

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error processing data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Create figure
    fig = go.Figure()

    # Add Call Volume trace (positive)
    fig.add_trace(go.Bar(
        x=volume_by_strike['C_VOLUME'],
        y=volume_by_strike['STRIKE'],
        orientation='h',
        name='Calls',
        marker=dict(color=theme['call']),
        hovertemplate='Strike: %{y}<br>Call Volume: %{x:,.0f}<extra></extra>'
    ))

    # Add Put Volume trace (negative)
    fig.add_trace(go.Bar(
        x=volume_by_strike['P_VOLUME_NEG'],
        y=volume_by_strike['STRIKE'],
        orientation='h',
        name='Puts',
        marker=dict(color=theme['put']),
        hovertemplate='Strike: %{y}<br>Put Volume: %{customdata:,.0f}<extra></extra>',
        customdata=volume_by_strike['P_VOLUME']  # Show positive value in hover
    ))

    # Update layout
    fig.update_layout(
        title="Crowd Bets: Call vs Put Volume by Strike",
        xaxis_title="Volume (Calls → | ← Puts)",
        yaxis_title="Strike Price",
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=theme['text']),
        height=600,
        xaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=theme['text']
        ),
        yaxis=dict(
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=80, r=80, t=80, b=60)
    )

    return fig


def plot_gauge(df, theme):
    """
    Create gauge chart showing average ATM put implied volatility

    Args:
        df (pd.DataFrame): Options data with columns: STRIKE, UNDERLYING_LAST, P_IV
        theme (dict): Theme dictionary with 'call', 'put', 'text' keys

    Returns:
        go.Figure: Plotly gauge indicator
    """
    # Validate required columns
    required_cols = ['STRIKE', 'UNDERLYING_LAST', 'P_IV']
    if not all(col in df.columns for col in required_cols):
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required columns: STRIKE, UNDERLYING_LAST, P_IV",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

    # Calculate ATM IV (strikes within 5% of current price)
    try:
        current_price = df['UNDERLYING_LAST'].iloc[0] if len(df) > 0 else 0
        lower_bound = current_price * 0.95
        upper_bound = current_price * 1.05

        atm_options = df[
            (df['STRIKE'] >= lower_bound) &
            (df['STRIKE'] <= upper_bound)
        ]

        if len(atm_options) == 0:
            avg_iv = 0
        else:
            avg_iv = atm_options['P_IV'].mean() * 100  # Scale by 100

    except Exception as e:
        avg_iv = 0

    # Determine color steps based on theme
    if 'Colorblind' in str(theme.get('heatmap', '')):
        # Colorblind Safe: Blue (low) → Orange (high)
        gauge_colors = [
            [0, '#0072B2'],      # Blue
            [0.5, '#F0E442'],    # Yellow
            [1, '#D55E00']       # Vermilion/Orange
        ]
    else:
        # Standard: Green (low) → Yellow (medium) → Red (high)
        gauge_colors = [
            [0, '#00CC00'],      # Green
            [0.5, '#FFCC00'],    # Yellow
            [1, '#FF0000']       # Red
        ]

    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_iv,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ATM Put IV (%)", 'font': {'size': 20, 'color': theme['text']}},
        number={'suffix': "%", 'font': {'size': 40, 'color': theme['text']}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': theme['text'],
                'ticksuffix': '%'
            },
            'bar': {'color': theme['put']},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 2,
            'bordercolor': theme['text'],
            'steps': [
                {'range': [0, 30], 'color': gauge_colors[0][1]},
                {'range': [30, 60], 'color': gauge_colors[1][1]},
                {'range': [60, 100], 'color': gauge_colors[2][1]}
            ],
            'threshold': {
                'line': {'color': theme['text'], 'width': 4},
                'thickness': 0.75,
                'value': avg_iv
            }
        }
    ))

    # Update layout
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme['text'], 'family': "Arial"},
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def plot_gex_profile(df, theme):
    """
    Create GEX (Gamma Exposure) profile chart - "The Battlefield"
    Shows net GEX by strike with call/put walls

    Args:
        df (pd.DataFrame): Options data with columns: STRIKE, C_VOLUME, P_VOLUME
        theme (dict): Theme dictionary with 'call', 'put', 'text' keys

    Returns:
        go.Figure: Plotly horizontal bar chart
    """
    # Validate required columns
    required_cols = ['STRIKE', 'C_VOLUME', 'P_VOLUME']
    if not all(col in df.columns for col in required_cols):
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required columns: STRIKE, C_VOLUME, P_VOLUME",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

    try:
        # Initialize MarketMath to get GEX and walls
        market = MarketMath(df)
        gex_df = market.calculate_gex_proxy()
        walls = market.get_walls()

        if len(gex_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No GEX data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Separate positive and negative GEX
        positive_gex = gex_df[gex_df['NET_GEX'] >= 0].copy()
        negative_gex = gex_df[gex_df['NET_GEX'] < 0].copy()

        # Create figure
        fig = go.Figure()

        # Add positive GEX bars (calls dominate)
        if len(positive_gex) > 0:
            fig.add_trace(go.Bar(
                x=positive_gex['NET_GEX'],
                y=positive_gex['STRIKE'],
                orientation='h',
                name='Call Dominance',
                marker=dict(color=theme['call']),
                hovertemplate='Strike: %{y}<br>Net GEX: %{x:,.0f}<extra></extra>'
            ))

        # Add negative GEX bars (puts dominate)
        if len(negative_gex) > 0:
            fig.add_trace(go.Bar(
                x=negative_gex['NET_GEX'],
                y=negative_gex['STRIKE'],
                orientation='h',
                name='Put Dominance',
                marker=dict(color=theme['put']),
                hovertemplate='Strike: %{y}<br>Net GEX: %{x:,.0f}<extra></extra>'
            ))

        # Add call wall line
        if walls['call_wall'] is not None:
            fig.add_hline(
                y=walls['call_wall'],
                line_dash="dash",
                line_color=theme['call'],
                line_width=2,
                annotation_text=f"Call Wall: ${walls['call_wall']:.0f}",
                annotation_position="top right",
                annotation=dict(font_size=10, font_color=theme['call'])
            )

        # Add put wall line
        if walls['put_wall'] is not None:
            fig.add_hline(
                y=walls['put_wall'],
                line_dash="dash",
                line_color=theme['put'],
                line_width=2,
                annotation_text=f"Put Wall: ${walls['put_wall']:.0f}",
                annotation_position="bottom right",
                annotation=dict(font_size=10, font_color=theme['put'])
            )

        # Update layout
        fig.update_layout(
            title="GEX Profile: The Battlefield",
            xaxis_title="Net Gamma Exposure (Calls - Puts)",
            yaxis_title="Strike Price",
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme['text']),
            height=600,
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinewidth=3,
                zerolinecolor=theme['text']
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=80, r=80, t=80, b=60)
        )

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating GEX profile: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig


def plot_forecast_cone(current_price, upper, lower, days, theme):
    """
    Create volatility forecast cone - "The Hurricane"
    Shows projected price range based on implied volatility

    Args:
        current_price (float): Current stock price
        upper (list): Upper band values (+1 SD)
        lower (list): Lower band values (-1 SD)
        days (list): Array of days [0, 1, 2, ..., N]
        theme (dict): Theme dictionary with 'call', 'put', 'text' keys

    Returns:
        go.Figure: Plotly line chart with filled area
    """
    if len(days) == 0 or len(upper) == 0 or len(lower) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No forecast data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    try:
        fig = go.Figure()

        # Add upper band
        fig.add_trace(go.Scatter(
            x=days,
            y=upper,
            mode='lines',
            name='Upper Band (+1σ)',
            line=dict(color=theme['call'], width=2, dash='dot'),
            hovertemplate='Day %{x}<br>Upper: $%{y:.2f}<extra></extra>'
        ))

        # Add lower band with fill
        fig.add_trace(go.Scatter(
            x=days,
            y=lower,
            mode='lines',
            name='Lower Band (-1σ)',
            line=dict(color=theme['put'], width=2, dash='dot'),
            fill='tonexty',
            fillcolor=f"rgba({int(theme['put'][1:3], 16)}, {int(theme['put'][3:5], 16)}, {int(theme['put'][5:7], 16)}, 0.1)",
            hovertemplate='Day %{x}<br>Lower: $%{y:.2f}<extra></extra>'
        ))

        # Add center line (current price)
        center_line = [current_price] * len(days)
        fig.add_trace(go.Scatter(
            x=days,
            y=center_line,
            mode='lines',
            name='Current Price',
            line=dict(color=theme['text'], width=3, dash='dot'),
            hovertemplate='Day %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title="Volatility Forecast Cone: The Hurricane",
            xaxis_title="Days Ahead",
            yaxis_title="Projected Price ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme['text']),
            height=500,
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=80, r=80, t=80, b=60)
        )

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating forecast cone: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig


def plot_3d_surface(df, theme):
    """
    Create 3D surface plot of implied volatility - "The Landscape"
    Shows IV across strike and DTE dimensions

    Args:
        df (pd.DataFrame): Options data with columns: DTE, STRIKE, P_IV
        theme (dict): Theme dictionary with 'heatmap' and 'text' keys

    Returns:
        go.Figure: Plotly 3D surface plot
    """
    # Validate required columns
    required_cols = ['DTE', 'STRIKE', 'P_IV']
    if not all(col in df.columns for col in required_cols):
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required columns: DTE, STRIKE, P_IV",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

    try:
        # Create pivot table for surface
        pivot_df = df.pivot_table(
            values='P_IV',
            index='STRIKE',
            columns='DTE',
            aggfunc='mean'
        )

        # Sort for proper visualization
        pivot_df = pivot_df.sort_index(ascending=True)
        pivot_df = pivot_df[sorted(pivot_df.columns)]

        # Extract arrays
        x = pivot_df.columns.values  # DTE
        y = pivot_df.index.values    # STRIKE
        z = pivot_df.values          # P_IV

        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=theme['heatmap'],
            hovertemplate='DTE: %{x}<br>Strike: %{y}<br>Put IV: %{z:.2%}<extra></extra>',
            lighting=dict(
                ambient=0.4,
                diffuse=0.5,
                fresnel=0.2,
                specular=0.05,
                roughness=0.5
            ),
            colorbar=dict(
                title=dict(text="Put IV", side='right'),
                tickformat='.0%',
                len=0.7
            )
        )])

        # Update layout for 3D
        fig.update_layout(
            title="3D IV Surface: The Landscape",
            scene=dict(
                xaxis=dict(
                    title='Days to Expiration (DTE)',
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(128,128,128,0.3)',
                    showbackground=True
                ),
                yaxis=dict(
                    title='Strike Price',
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(128,128,128,0.3)',
                    showbackground=True
                ),
                zaxis=dict(
                    title='Put Implied Volatility',
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(128,128,128,0.3)',
                    showbackground=True,
                    tickformat='.0%'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme['text']),
            height=700,
            margin=dict(l=0, r=0, t=60, b=0)
        )

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating 3D surface: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

# ============================================================================
# LOAD DATA
# ============================================================================
df = load_data()

# Extract date range from data
if 'QUOTE_DATE' in df.columns:
    min_date = df['QUOTE_DATE'].min()
    max_date = df['QUOTE_DATE'].max()
    date_list = sorted(df['QUOTE_DATE'].unique())
else:
    st.error("❌ Data must contain a 'QUOTE_DATE' column")
    st.stop()

# ============================================================================
# TOP CONTROL PANEL - THE SETTINGS
# ============================================================================
# Add main project title (centered)
st.markdown("<h1 style='text-align: center;'>Market Sentiment Radar</h1>", unsafe_allow_html=True)

with st.expander("Dashboard Settings & Filters", expanded=True):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Time Travel")
        # Date selection using select_slider for smooth navigation
        target_date = st.select_slider(
            "Select Target Date",
            options=date_list,
            value=max_date,
            format_func=lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
        )

        # Alternative: Date input for direct date selection
        # target_date = st.date_input(
        #     "Select Target Date",
        #     value=max_date,
        #     min_value=min_date,
        #     max_value=max_date
        # )

        st.caption(f"Available range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    with col2:
        st.subheader("Accessibility")
        color_theme = st.radio(
            "Color Theme",
            options=['Dark Mode', 'Light Mode', 'Colorblind Safe'],
            index=0,
            horizontal=True,
            help="Choose a theme optimized for your viewing preferences"
        )

        # Display theme preview
        theme = get_theme(color_theme)
        st.caption(
            f"Active: Call: `{theme['call']}` | "
            f"Put: `{theme['put']}` | "
            f"Heatmap: `{theme['heatmap']}`"
        )

# ============================================================================
# DATA FILTERING
# ============================================================================
# Filter dataframe to selected date
if hasattr(target_date, 'strftime'):
    day_df = df[df['QUOTE_DATE'] == target_date].copy()
else:
    day_df = df[df['QUOTE_DATE'] == pd.to_datetime(target_date)].copy()

# Apply selected theme
theme = get_theme(color_theme)

# ============================================================================
# DYNAMIC DATE HEADER
# ============================================================================
st.markdown(f"### Selected Date: {target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)}")

# ============================================================================
# CALCULATE KEY METRICS & MARKET ANALYTICS
# ============================================================================
# Initialize MarketMath for advanced calculations
market = MarketMath(day_df)
summary_stats = market.get_summary_stats()
walls = market.get_walls()
flip = market.get_flip_level()

# Stock Price
stock_price = summary_stats.get('current_price', 0)

# Fear Score (ATM Put IV)
fear_score = 0
if 'STRIKE' in day_df.columns and 'UNDERLYING_LAST' in day_df.columns and 'P_IV' in day_df.columns and len(day_df) > 0:
    lower_bound = stock_price * 0.95
    upper_bound = stock_price * 1.05
    atm_options = day_df[
        (day_df['STRIKE'] >= lower_bound) &
        (day_df['STRIKE'] <= upper_bound)
    ]
    if len(atm_options) > 0:
        fear_score = atm_options['P_IV'].mean() * 100

# Generate forecast cone for strategy tab
cone_data = market.get_projection_cone(stock_price, fear_score / 100, days=60) if stock_price > 0 and fear_score > 0 else None

# ============================================================================
# HEADS-UP DISPLAY: COMMAND CENTER METRICS
# ============================================================================
st.markdown("### Command Center: Market Structure")

# Row 1: Metrics + Fear Gauge
metrics_col, gauge_col = st.columns([2, 1])

with metrics_col:
    hud_col1, hud_col2, hud_col3, hud_col4 = st.columns(4)

    with hud_col1:
        st.metric(
            label="Current Price",
            value=f"${stock_price:,.2f}",
            help="Real-time underlying stock price"
        )

    with hud_col2:
        ceiling_value = f"${walls['call_wall']:.0f}" if walls['call_wall'] else "N/A"
        st.metric(
            label="The Ceiling",
            value=ceiling_value,
            help="Resistance Level: Strike with highest call volume acts as price ceiling"
        )

    with hud_col3:
        floor_value = f"${walls['put_wall']:.0f}" if walls['put_wall'] else "N/A"
        st.metric(
            label="The Floor",
            value=floor_value,
            help="Support Level: Strike with highest put volume acts as price floor"
        )

    with hud_col4:
        flip_value = f"${flip['flip_strike']:.0f}" if flip['flip_strike'] else "N/A"
        st.metric(
            label="Turbulence Zone",
            value=flip_value,
            help="Gamma Flip Point: Where market dynamics shift from bearish to bullish"
        )

with gauge_col:
    st.markdown("#### Fear Index")
    try:
        fig_gauge = plot_gauge(day_df, theme)
        st.plotly_chart(fig_gauge, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering gauge: {str(e)}")

    st.caption(f"""
    **Current Level:** {fear_score:.1f}%
    - **0-30%**: Market calm
    - **30-60%**: Normal volatility
    - **60-100%**: High anxiety
    """)

# ============================================================================
# SECTION 1: MARKET STRUCTURE ANALYSIS
# ============================================================================
st.divider()
st.markdown("### Market Structure Analysis")
st.caption("Understanding support and resistance levels based on options positioning")

struct_col1, struct_col2 = st.columns([2, 1])

with struct_col1:
    st.markdown("#### Volume Profile: Where Traders Are Positioned")
    try:
        fig_crowd = plot_crowd_bets(day_df, theme)
        st.plotly_chart(fig_crowd, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering volume profile: {str(e)}")

    st.caption("""
    **Reading the chart:** Bars extending right show call volume (bullish positioning).
    Bars extending left show put volume (bearish positioning). The largest concentrations
    indicate where institutional money is positioned.
    """)

with struct_col2:
    st.markdown("#### GEX Profile: The Battlefield")
    try:
        fig_gex = plot_gex_profile(day_df, theme)
        st.plotly_chart(fig_gex, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering GEX profile: {str(e)}")

    if color_theme == 'Colorblind Safe':
        st.caption("""
        **🔵 Blue Bars** = Dealer buying pressure (support)
        **🟠 Orange Bars** = Dealer selling pressure (resistance)
        """)
    else:
        st.caption("""
        **🟢 Green Bars** = Dealer buying pressure (support)
        **🔴 Red Bars** = Dealer selling pressure (resistance)
        """)

# Add interpretation section
st.divider()
st.markdown("#### Market Structure Interpretation")

interp_col1, interp_col2, interp_col3 = st.columns(3)

with interp_col1:
    st.markdown("**Resistance (Ceiling)**")
    if walls['call_wall']:
        st.write(f"${walls['call_wall']:.0f}")
        st.caption("Heavy call volume acts as a price ceiling. Dealers hedge by selling stock as price approaches.")
    else:
        st.write("N/A")

with interp_col2:
    st.markdown("**Support (Floor)**")
    if walls['put_wall']:
        st.write(f"${walls['put_wall']:.0f}")
        st.caption("Heavy put volume acts as a price floor. Dealers hedge by buying stock as price approaches.")
    else:
        st.write("N/A")

with interp_col3:
    st.markdown("**Trading Range**")
    if walls['call_wall'] and walls['put_wall']:
        range_width = abs(walls['call_wall'] - walls['put_wall'])
        st.write(f"${range_width:.0f}")
        st.caption(f"Expected price range between support and resistance levels.")
    else:
        st.write("N/A")

# ============================================================================
# SECTION 2: VOLATILITY ANALYSIS
# ============================================================================
st.divider()
st.markdown("### Volatility Analysis")
st.caption("Monitor implied volatility patterns and market anxiety levels")

vol_col1, vol_col2 = st.columns(2)

with vol_col1:
    st.markdown("#### 3D Volatility Landscape")
    try:
        fig_3d = plot_3d_surface(day_df, theme)
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering 3D surface: {str(e)}")

    st.caption("""
    **Reading the landscape:** Peaks (mountains) show high implied volatility - markets expect big moves.
    Valleys show low IV - markets expect stability. The surface shape reveals how volatility changes
    across strike prices and time to expiration.
    """)

with vol_col2:
    st.markdown("#### IV Heatmap: Strike × Time")
    try:
        fig_heatmap = plot_heatmap(day_df, theme)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering heatmap: {str(e)}")

    st.caption("""
    **Darker colors** = Higher implied volatility. The dashed line shows current stock price.
    Watch for volatility clustering around key strikes - these often become magnets for price action.
    """)

# ============================================================================
# SECTION 3: PROBABILITY CONE
# ============================================================================
st.divider()
st.markdown("### Market Forecast: Probability Cone")
st.caption("Statistical price projections based on current implied volatility")

if cone_data and len(cone_data['days']) > 0:
    try:
        # Calculate 2SD bands (95% probability)
        upper_2sd = []
        lower_2sd = []
        for i, day in enumerate(cone_data['days']):
            t = day / 365.0
            sigma = stock_price * (fear_score / 100) * (t ** 0.5)
            upper_2sd.append(stock_price + 2 * sigma)
            lower_2sd.append(max(0, stock_price - 2 * sigma))

        # Create probability cone chart
        fig_cone = go.Figure()

        # 2SD band (95% probability) - outer cone
        fig_cone.add_trace(go.Scatter(
            x=cone_data['days'] + cone_data['days'][::-1],
            y=upper_2sd + lower_2sd[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='95% (2σ)',
            hoverinfo='skip'
        ))

        # 1SD band (68% probability) - inner cone
        fig_cone.add_trace(go.Scatter(
            x=cone_data['days'] + cone_data['days'][::-1],
            y=cone_data['upper_band'] + cone_data['lower_band'][::-1],
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(color='rgba(0,255,0,0)'),
            name='68% (1σ)',
            hoverinfo='skip'
        ))

        # Center line (current price)
        fig_cone.add_trace(go.Scatter(
            x=cone_data['days'],
            y=[stock_price] * len(cone_data['days']),
            mode='lines',
            name='Current Price',
            line=dict(color=theme['text'], width=2, dash='dash'),
            hovertemplate='Day %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        # Spot marker
        fig_cone.add_trace(go.Scatter(
            x=[0],
            y=[stock_price],
            mode='markers',
            marker=dict(color=theme['text'], size=12, symbol='circle'),
            name='Spot',
            hovertemplate='Current: $%{y:.2f}<extra></extra>'
        ))

        # Update layout
        fig_cone.update_layout(
            title="60-Day Price Forecast",
            xaxis_title="Days Ahead",
            yaxis_title="Projected Price ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme['text']),
            height=500,
            hovermode="x unified",
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=80, r=80, t=80, b=60)
        )

        st.plotly_chart(fig_cone, use_container_width=True)

        # Display key levels
        day_30_idx = min(30, len(cone_data['days']) - 1)
        day_60_idx = len(cone_data['days']) - 1

        cone_metrics_col1, cone_metrics_col2, cone_metrics_col3 = st.columns(3)

        with cone_metrics_col1:
            st.metric(
                "Current Price",
                f"${stock_price:,.2f}",
                help="Starting point for projection"
            )

        with cone_metrics_col2:
            upper_30 = cone_data['upper_band'][day_30_idx]
            lower_30 = cone_data['lower_band'][day_30_idx]
            upper_30_2sd = upper_2sd[day_30_idx]
            lower_30_2sd = lower_2sd[day_30_idx]
            st.metric(
                "30-Day Range",
                f"${lower_30:.0f} - ${upper_30:.0f}",
                help="68% probability (1σ)"
            )
            st.caption(f"95% range: ${lower_30_2sd:.0f} - ${upper_30_2sd:.0f}")

        with cone_metrics_col3:
            upper_60 = cone_data['upper_band'][day_60_idx]
            lower_60 = cone_data['lower_band'][day_60_idx]
            upper_60_2sd = upper_2sd[day_60_idx]
            lower_60_2sd = lower_2sd[day_60_idx]
            st.metric(
                "60-Day Range",
                f"${lower_60:.0f} - ${upper_60:.0f}",
                help="68% probability (1σ)"
            )
            st.caption(f"95% range: ${lower_60_2sd:.0f} - ${upper_60_2sd:.0f}")

        st.caption("""
        **How to read:** The green zone shows where price has a 68% probability of being (±1 standard deviation).
        The red zone shows a 95% probability range (±2 standard deviations). These projections assume no drift
        and are based on current implied volatility.
        """)

    except Exception as e:
        st.error(f"Error rendering probability cone: {str(e)}")
else:
    st.warning("Insufficient data to generate probability cone. Need valid price and IV data.")

# ============================================================================
# DATA TABLE (Optional)
# ============================================================================
st.divider()
with st.expander("View Raw Data", expanded=False):
    st.dataframe(
        day_df,
        use_container_width=True,
        height=400
    )
    st.caption(f"Showing {len(day_df)} records for {target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)}")

# ============================================================================
# FOOTER WITH DYNAMIC THEME EXPLANATION
# ============================================================================
st.divider()

# Dynamic footer based on current theme
footer_col1, footer_col2 = st.columns([2, 1])

with footer_col1:
    if color_theme == 'Colorblind Safe':
        st.caption(f"""
        **Currently showing:** 🔵 Blue = Calls (Bullish) | 🟠 Orange = Puts (Fear/Bearish) |
        **Theme:** {color_theme} - Optimized for colorblind accessibility using Okabe-Ito palette
        """)
    elif color_theme == 'Dark Mode':
        st.caption(f"""
        **Currently showing:** 🟢 Teal = Calls (Bullish) | 🔴 Red = Puts (Fear/Bearish) |
        **Theme:** {color_theme} - Dark theme for reduced eye strain
        """)
    else:
        st.caption(f"""
        **Currently showing:** 🟢 Green = Calls (Bullish) | 🔴 Red = Puts (Fear/Bearish) |
        **Theme:** {color_theme} - Classic light theme
        """)

with footer_col2:
    st.caption("Powered by Streamlit | Market Sentiment Radar v1.0")

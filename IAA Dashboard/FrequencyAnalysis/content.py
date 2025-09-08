import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import krippendorff
from sklearn.preprocessing import LabelEncoder
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import dash_bootstrap_components as dbc
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from .calculator import FrequencyAnalysisCalculator

current_dir = Path(__file__).parent
data_file_path = current_dir.parent / 'data.csv'

agreement_df = pd.read_csv(data_file_path)
freq_calculator = FrequencyAnalysisCalculator(agreement_df)


def create_frequency_tab_content():
    """Create Frequency-Based Analysis tab content."""
    return html.Div([
        html.H4("Frequency-Based Analysis", 
                style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'backgroundClip': 'text',
                    'fontWeight': '700',
                    'fontSize': '2rem',
                    'marginBottom': '2rem',
                    'textAlign': 'center'
                }),

        # Controls Section - Redesigned
        dbc.Card([
            dbc.CardHeader([
                html.H5("Frequency Stratification Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Label Type and Thresholds
                dbc.Row([
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='freq-label-type-selector',
                            options=[
                                {'label': 'Full Hierarchical Labels', 'value': 'full_label'},
                                {'label': 'L1 (Parent) Labels', 'value': 'L1_label'},
                                {'label': 'L2 (Child) Labels', 'value': 'L2_label'}
                            ],
                            value='full_label',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Rare Threshold (≤):", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='rare-threshold-slider',
                                min=500,
                                max=1200,
                                step=10,
                                value=1000,
                                marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(500, 1201, 100)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=4),
                    dbc.Col([
                        html.Label("Common Threshold (≥):", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='common-threshold-slider',
                                min=1400,
                                max=2000,
                                step=10,
                                value=1500,
                                marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(1400, 2001, 100)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=4)
                ], className="mb-4"),
                
                # Row 2: Annotators
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem', 'textAlign': 'center', 'display': 'block'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='freq-annotator-selector',
                                    options=[{'label': f'Annotator {ann.split("_")[1]}', 'value': ann} for ann in freq_calculator.annotators],
                                    value=freq_calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem','textAlign': 'center'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=9),
                    dbc.Col([
                        html.Div(style={'height': '3rem'}),  # Spacer
                        dbc.Button("Calculate Frequency Analysis", 
                                 id="freq-calculate-btn",
                                 size="lg", 
                                 style={
                                     'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                     'border': 'none',
                                     'borderRadius': '12px',
                                     'fontWeight': '600',
                                     'fontSize': '1.1rem',
                                     'padding': '12px 30px',
                                     'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)',
                                     'transition': 'all 0.3s ease',
                                     'width': '100%',
                                     'minHeight': '50px'
                                 })
                    ], md=3)
                ])
            ], style={'padding': '2rem'})
        ], style={
            'border': '1px solid rgba(102, 126, 234, 0.3)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)',
            'marginBottom': '3rem'
        }),


        # Progress indicator
        dbc.Row([
            dbc.Col([
                dbc.Progress(id="freq-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),
        
        # Results container with better spacing
        html.Div(
            id="freq-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})


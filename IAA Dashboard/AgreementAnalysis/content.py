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


from .calculator import IAAAgreementCalculator

current_dir = Path(__file__).parent
data_file_path = current_dir.parent / 'data.csv'

agreement_df = pd.read_csv(data_file_path)

calculator = IAAAgreementCalculator(agreement_df)

def create_overall_tab_content():
    """Create Overall Agreement Analysis tab content."""
    return html.Div([
        html.H4("Overall Agreement Analysis", 
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
                html.H5("Analysis Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea',
                           'textAlign': 'center'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Annotators
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem', 'textAlign': 'center', 'display': 'block'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='overall-annotator-selector',
                                    options=[{'label': f'Annotator {ann.split("_")[1]}', 'value': ann} for ann in calculator.annotators],
                                    value=calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem', 'textAlign': 'center'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=12)
                ], className="mb-4"),
                
                # Row 2: Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='overall-label-type-selector',
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
                        html.Label("Confidence Level:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='overall-confidence-slider',
                                min=0.90,
                                max=0.99,
                                step=0.01,
                                value=0.95,
                                marks={0.90: {'label': '90%', 'style': {'fontSize': '14px'}}, 
                                       0.95: {'label': '95%', 'style': {'fontSize': '14px'}}, 
                                       0.99: {'label': '99%', 'style': {'fontSize': '14px'}}},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=4),
                    dbc.Col([
                        html.Div(style={'height': '2rem'}),  # Spacer
                        dbc.Button("Calculate Agreement Analysis", 
                                 id="overall-calculate-btn",
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
                                     'width': '100%'
                                 })
                    ], md=4)
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
                dbc.Progress(id="overall-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),
            
        # Results container with better spacing
        html.Div(
            id="overall-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})


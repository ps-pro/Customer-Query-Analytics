import pandas as pd
import numpy as np
import itertools
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import dash_bootstrap_components as dbc
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from .calculator import DisagreementAnalysisCalculator

current_dir = Path(__file__).parent
data_file_path = current_dir.parent / 'data.csv'

disagreement_df = pd.read_csv(data_file_path)
disagreement_calculator = DisagreementAnalysisCalculator(disagreement_df)



def create_disagreement_tab():
    """Create the disagreement analysis tab."""
    return html.Div([
        html.H4("Top Disagreement Items - Sample Analysis", 
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
                html.H5("Disagreement Analysis Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Main Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='disagreement-label-type-selector',
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
                    ], md=3),
                    dbc.Col([
                        html.Label("Top N Samples:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='top-n-selector',
                            options=[
                                {'label': 'Top 10', 'value': 10},
                                {'label': 'Top 25', 'value': 25},
                                {'label': 'Top 50', 'value': 50},
                                {'label': 'Top 100', 'value': 100}
                            ],
                            value=25,
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("Min Disagreement Score:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='min-disagreement-slider',
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                value=0.0,
                                marks={i/10: {'label': f'{i/10:.1f}', 'style': {'fontSize': '12px'}} for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=3),
                    dbc.Col([
                        html.Label("Max Disagreement Score:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='max-disagreement-slider',
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                value=1.0,
                                marks={i/10: {'label': f'{i/10:.1f}', 'style': {'fontSize': '12px'}} for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=3)
                ], className="mb-4"),
                
                # Row 2: Annotators and Calculate Button
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem', 'textAlign': 'center', 'display': 'block'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='disagreement-annotator-selector',
                                    options=[{'label': f'Annotator {ann.split("_")[1]}', 'value': ann} for ann in disagreement_calculator.annotators],
                                    value=disagreement_calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem', 'textAlign': 'center'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=9),
                    dbc.Col([
                        html.Div(style={'height': '3rem'}),  # Spacer
                        dbc.Button("Calculate Disagreement Analysis", 
                                 id="disagreement-calculate-btn",
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
                dbc.Progress(id="disagreement-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),

        # Results Section
        html.Div(
            id="disagreement-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})


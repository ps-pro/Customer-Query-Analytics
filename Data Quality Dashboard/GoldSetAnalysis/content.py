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

from .calculator import GoldSetAnalysisCalculator

current_dir = Path(__file__).parent
data_file_path = current_dir.parent / 'data.csv'

goldset_df = pd.read_csv(data_file_path)
goldset_calculator = GoldSetAnalysisCalculator(goldset_df)


def create_goldset_tab():
    """Create the gold-set refresh analysis tab."""
    return html.Div([
        html.H4("Suggested Gold-Set Refresh Analysis", 
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
                html.H5("Gold-Set Strategy Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Main Strategy Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='goldset-label-type-selector',
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
                        html.Label("Gold-Set Strategy:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='goldset-strategy-selector',
                            options=[
                                {'label': 'High Agreement Only', 'value': 'high_only'},
                                {'label': 'Mixed (Recommended)', 'value': 'mixed'},
                                {'label': 'Include More Disagreement', 'value': 'disagreement_focus'}
                            ],
                            value='mixed',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("Samples per Label:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='samples-per-label-slider',
                                min=2,
                                max=10,
                                step=1,
                                value=5,
                                marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(2, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=3),
                    dbc.Col([
                        html.Label("High Agreement Threshold:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='high-agreement-threshold-slider',
                                min=0.7,
                                max=1.0,
                                step=0.05,
                                value=0.9,
                                marks={i/100: {'label': f'{i/100:.2f}', 'style': {'fontSize': '12px'}} for i in range(70, 101, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=3)
                ], className="mb-4"),
                
                # Row 2: Range Controls and Annotators
                dbc.Row([
                    dbc.Col([
                        html.Label("Useful Disagreement Range:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.RangeSlider(
                                id='disagreement-range-slider',
                                min=0.3,
                                max=0.8,
                                step=0.05,
                                value=[0.4, 0.7],
                                marks={i/100: {'label': f'{i/100:.2f}', 'style': {'fontSize': '12px'}} for i in range(30, 81, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=6),
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem', 'textAlign': 'center', 'display': 'block'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='goldset-annotator-selector',
                                    options=[{'label': f'Annotator {ann.split("_")[1]}', 'value': ann} for ann in goldset_calculator.annotators],
                                    value=goldset_calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem', 'textAlign': 'center'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=6)
                ], className="mb-4"),
                
                # Row 3: Calculate Button
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Generate Gold-Set Recommendations", 
                                 id="goldset-calculate-btn",
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
                                     'maxWidth': '400px',
                                     'margin': '0 auto',
                                     'display': 'block'
                                 })
                    ], md=12, style={'textAlign': 'center'})
                ])
            ], style={'padding': '2rem'})
        ], style={
            'border': '1px solid rgba(102, 126, 234, 0.3)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)',
            'marginBottom': '3rem'
        }),

        # Strategy Explanation Card
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6("Gold-Set Strategy Guide", 
                           style={'fontWeight': '700','fontSize': '1.5rem', 'color': '#667eea', 'marginBottom': '1rem','textAlign': 'center'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("High Agreement Only", style={'fontWeight': '600', 'color': '#27ae60'}),
                                html.P("Selects only documents with high annotator agreement for training stability", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Mixed (Recommended)", style={'fontWeight': '600', 'color': '#667eea'}),
                                html.P("Combines high-agreement documents with useful edge cases for comprehensive coverage", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Include More Disagreement", style={'fontWeight': '600', 'color': '#f39c12'}),
                                html.P("Focuses on edge cases and disagreement examples for guideline development", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4)
                    ])
                ])
            ], style={'padding': '1.5rem'})
        ], style={
            'backgroundColor': 'rgba(102, 126, 234, 0.05)',
            'border': '1px solid rgba(102, 126, 234, 0.2)',
            'borderRadius': '12px',
            'marginBottom': '3rem'
        }),

        # Progress indicator
        dbc.Row([
            dbc.Col([
                dbc.Progress(id="goldset-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),

        # Results Section
        html.Div(
            id="goldset-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})

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

from .calculator import HierarchicalAnalysisCalculator

current_dir = Path(__file__).parent
data_file_path = current_dir.parent / 'data.csv'

agreement_df = pd.read_csv(data_file_path)
hier_calculator = HierarchicalAnalysisCalculator(agreement_df)


def create_hierarchical_tab_content():
    """Create Hierarchical Analysis tab content."""
    return html.Div([
        html.H4("Hierarchical Analysis", 
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
                html.H5("Hierarchical Analysis Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: All controls side by side
                dbc.Row([
                    dbc.Col([
                        html.Label("Analysis Type:", 
                                style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='analysis-type-selector',
                            options=[
                                {'label': 'Complete Analysis', 'value': 'complete'},
                                {'label': 'Level Comparison Only', 'value': 'levels'},
                                {'label': 'Conditional Analysis Only', 'value': 'conditional'}
                            ],
                            value='complete',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("Select Parent Categories:", 
                                style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='parent-category-selector',
                                    options=[{'label': cat, 'value': cat} for cat in hier_calculator.parent_categories],
                                    value=hier_calculator.parent_categories,
                                    inline=True,
                                    style={'fontSize': '1rem'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=4),
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='hier-annotator-selector',
                                    options=[{'label': f'Annotator {ann.split("_")[1]}', 'value': ann} for ann in hier_calculator.annotators],
                                    value=hier_calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=5)
                ], className="mb-4"),

                # Row 2: Centered Calculate Button
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Button("Calculate Hierarchical Analysis", 
                                    id="hier-calculate-btn",
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
                                        'minHeight': '50px'
                                    })
                        ], style={'textAlign': 'center'})
                    ], md=12)
                ])

            ], style={'padding': '2rem'})
        ], style={
            'border': '1px solid rgba(102, 126, 234, 0.3)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)',
            'marginBottom': '3rem'
        }),

        # Analysis Type Explanation Card
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6("Analysis Type Guide", 
                           style={'fontWeight': '600', 'color': '#667eea', 'marginBottom': '1rem'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("Complete Analysis", style={'fontWeight': '600', 'color': '#667eea'}),
                                html.P("Performs both level comparison and conditional analysis for comprehensive insights", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Level Comparison", style={'fontWeight': '600', 'color': '#f39c12'}),
                                html.P("Compares agreement across L1, L2, and full hierarchical labels", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Conditional Analysis", style={'fontWeight': '600', 'color': '#27ae60'}),
                                html.P("Analyzes agreement within each parent category separately", 
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
                dbc.Progress(id="hier-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),
        
        # Results container with better spacing
        html.Div(
            id="hier-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})


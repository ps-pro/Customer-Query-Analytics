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

from .calculator import AnnotatorConfusionCalculator


current_dir = Path(__file__).parent
data_file_path = current_dir.parent / 'data.csv'

confusion_df = pd.read_csv(data_file_path)
confusion_calculator = AnnotatorConfusionCalculator(confusion_df)


def create_confusion_tab():
    """Create the annotator confusion analysis tab with improved matrix display."""
    return html.Div([
        html.H2("Per-Annotator Confusion Analysis", 
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
                html.H5("Annotator Confusion Analysis Configuration", 
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
                        html.Label("Analysis Mode:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='confusion-analysis-mode',
                            options=[
                                {'label': 'Complete Analysis', 'value': 'complete'},
                                {'label': 'Individual Performance Only', 'value': 'individual'},
                                {'label': 'Pairwise Comparison Only', 'value': 'pairwise'}
                            ],
                            value='complete',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='confusion-label-type-selector',
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
                        html.Label("Individual Matrix View:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='matrix-annotator-selector',
                            options=[{'label': 'Select after analysis', 'value': 'none'}],
                            value='none',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            },
                            placeholder="Run analysis first"
                        )
                    ], md=4)
                ], className="mb-4"),
                
                # Row 2: Annotators and Calculate Button
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem', 'textAlign': 'center', 'display': 'block'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='confusion-annotator-selector',
                                    options=[{'label': f'Annotator {ann.split("_")[1]}', 'value': ann} for ann in confusion_calculator.annotators],
                                    value=confusion_calculator.annotators,
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
                        dbc.Button("Calculate Confusion Analysis", 
                                 id="confusion-calculate-btn",
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

        # Analysis Mode Explanation Card
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6("Analysis Mode Guide", 
                           style={'fontWeight': '700','fontSize': '1.5rem', 'textAlign': 'center', 'color': '#667eea', 'marginBottom': '1rem'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("Complete Analysis", style={'fontWeight': '600', 'color': '#667eea'}),
                                html.P("Performs both individual performance and pairwise comparison for comprehensive insights", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Individual Performance", style={'fontWeight': '600', 'color': '#f39c12'}),
                                html.P("Analyzes each annotator's performance against majority vote", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Pairwise Comparison", style={'fontWeight': '600', 'color': '#27ae60'}),
                                html.P("Compares agreement rates between each pair of annotators", 
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
                dbc.Progress(id="confusion-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),

        # Results Section
        html.Div(
            id="confusion-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State, ALL, MATCH
import dash_bootstrap_components as dbc
from collections import Counter, defaultdict
import re
import json
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from HITL.analyzer import HITLAnalyzer
from RuleBased.classifier import BaselineRuleClassifier
from FuzzyMatching.classifier import FuzzyMatchingClassifier



current_dir = Path(__file__).parent
data_file_path = current_dir.parent / 'data.csv'
agreement_df = pd.read_csv(data_file_path)


hitl_analyzer = HITLAnalyzer(agreement_df)
rule_classifier = BaselineRuleClassifier()
fuzzy_classifier = FuzzyMatchingClassifier()

def create_performance_overview_tab(analysis_data):
    """Create the performance overview tab."""

    config_section = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Fuzzy Matching Configuration", style={'color': 'white'})),
                dbc.CardBody([
                    dbc.RadioItems(
                        id="similarity-method-radio",
                        options=[
                            {"label": " Character-Level Similarity", "value": "character"},
                            {"label": " Semantic Similarity (TF-IDF)", "value": "semantic"}
                        ],
                        value="character",
                        inline=True,
                        style={'fontSize': '1.1rem', 'marginTop': '1rem'}
                    )
                ])
            ])
        ], md=6),
        dbc.Col([
            dbc.Button("Analyze Performance", id="analyze-performance-btn",
                     color="primary", size="lg", className="mt-3",
                     style={'width': '100%', 'height': '60px', 'fontSize': '1.2rem'})
        ], md=6)
    ], className="mb-4")

    results_section = html.Div(id="performance-results-display")

    if analysis_data:
        results_section = create_performance_results_display(analysis_data)

    return html.Div([config_section, results_section])

def create_error_analysis_tab(error_data):
    """Create the error analysis tab."""

    if not error_data:
        return html.Div([
            html.H4("Error Analysis & Improvement Opportunities", 
                style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'backgroundClip': 'text',
                    'fontWeight': '700',
                    'fontSize': '2.5rem',
                    'marginBottom': '2rem',
                    'textAlign': 'center'
                }),
            dbc.Alert([
                html.H5("No Analysis Data Available", className="alert-heading"),
                html.P("Please go to the Performance Overview tab and click 'Analyze Performance' first.", 
                      style={'marginBottom': '0'})
            ], color="info", className="mt-4")
        ])

    return html.Div([
        html.H4("Error Analysis & Improvement Opportunities", 
            style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'WebkitBackgroundClip': 'text',
                'WebkitTextFillColor': 'transparent',
                'backgroundClip': 'text',
                'fontWeight': '700',
                'fontSize': '2.5rem',
                'marginBottom': '2rem',
                'textAlign': 'center'
            }),
        create_error_analysis_display(error_data)
    ])

def create_crud_management_tab():
    """Create the CRUD management tab."""
    return html.Div([
        html.H4("HITL Rule Management Interface", 
            style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'WebkitBackgroundClip': 'text',
                'WebkitTextFillColor': 'transparent',
                'backgroundClip': 'text',
                'fontWeight': '700',
                'fontSize': '2.5rem',
                'marginBottom': '2rem',
                'textAlign': 'center'
            }),

        dbc.Tabs([
            dbc.Tab(label="Rule-Based Classifier Rules", tab_id="rule-crud-tab"),
            dbc.Tab(label="Fuzzy Matching Examples", tab_id="fuzzy-crud-tab"),
            dbc.Tab(label="Test Environment", tab_id="test-crud-tab")
        ], id="crud-sub-tabs", active_tab="rule-crud-tab"),

        html.Div(id="crud-sub-content", className="mt-4")
    ])

def create_performance_results_display(analysis_data):
    """Create performance results display with BIGGER plots."""
    try:
        comparison_results = analysis_data['comparison_results']

        # BIGGER Performance comparison chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        rule_values = [comparison_results['rule_based'][metric] for metric in metrics]
        fuzzy_values = [comparison_results['fuzzy_matching'][metric] for metric in metrics]

        performance_fig = go.Figure()
        performance_fig.add_trace(go.Bar(
            name='Rule-based Classifier',
            x=metrics,
            y=rule_values,
            text=[f'{val:.3f}' for val in rule_values],
            textposition='outside',
            marker_color='#667eea'
        ))
        performance_fig.add_trace(go.Bar(
            name='Fuzzy Matching Classifier',
            x=metrics,
            y=fuzzy_values,
            text=[f'{val:.3f}' for val in fuzzy_values],
            textposition='outside',
            marker_color='#764ba2'
        ))
        performance_fig.update_layout(
            title={
                'text': "Classifier Performance Comparison vs Human Consensus",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            title_font_size=20,
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            barmode='group',
            height=600,  # BIGGER
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        # BIGGER Confidence distribution chart
        rule_conf = comparison_results['rule_based']['confidences']
        fuzzy_conf = comparison_results['fuzzy_matching']['confidences']

        confidence_fig = go.Figure()
        confidence_fig.add_trace(go.Histogram(
            x=rule_conf,
            name='Rule-based Confidence',
            opacity=0.7,
            nbinsx=20,
            marker_color='#667eea'
        ))
        confidence_fig.add_trace(go.Histogram(
            x=fuzzy_conf,
            name='Fuzzy Matching Confidence',
            opacity=0.7,
            nbinsx=20,
            marker_color='#764ba2'
        ))
        confidence_fig.update_layout(
            title={
                'text': "Confidence Score Distributions",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            title_font_size=20,
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            barmode='overlay',
            height=600,  # BIGGER
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=performance_fig, style={'height': '650px'})  # BIGGER
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=confidence_fig, style={'height': '650px'})  # BIGGER
                ], md=6)
            ]),

            # Summary metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Rule-based Classifier", style={'color': 'white', 'fontWeight': '700', 'textAlign': 'center', 'fontSize': '1.4rem', 'margin': '0'})),
                        dbc.CardBody([
                            html.P(f"Accuracy: {comparison_results['rule_based']['accuracy']:.1%}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"F1-Score: {comparison_results['rule_based']['f1_score']:.3f}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"Avg Confidence: {np.mean(rule_conf):.3f}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'})
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Fuzzy Matching Classifier", style={'color': 'white', 'fontWeight': '700', 'textAlign': 'center', 'fontSize': '1.4rem', 'margin': '0'})),
                        dbc.CardBody([
                            html.P(f"Accuracy: {comparison_results['fuzzy_matching']['accuracy']:.1%}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"F1-Score: {comparison_results['fuzzy_matching']['f1_score']:.3f}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"Avg Confidence: {np.mean(fuzzy_conf):.3f}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'})
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Human Consensus", style={'color': 'white', 'fontWeight': '700', 'textAlign': 'center', 'fontSize': '1.4rem', 'margin': '0'})),                        
                        dbc.CardBody([
                            html.P(f"Documents: {len(hitl_analyzer.human_consensus)}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"Avg Confidence: {np.mean([d['confidence'] for d in hitl_analyzer.human_consensus.values()]):.1%}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P("Gold Standard Baseline", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'})
                        ])
                    ])
                ], md=4)
            ], className="mt-4")
        ])

    except Exception as e:
        return dbc.Alert(f"Error displaying results: {str(e)}", color="danger")

def create_error_analysis_display(error_data):
    """Create error analysis display with BIGGER plots."""
    try:
        rule_disagreements = error_data['rule_disagreements']
        fuzzy_disagreements = error_data['fuzzy_disagreements']
        rule_suggestions = error_data['rule_suggestions']
        fuzzy_suggestions = error_data['fuzzy_suggestions']

        # BIGGER Error analysis chart
        rule_errors = Counter([d['true_label'] for d in rule_disagreements])
        fuzzy_errors = Counter([d['true_label'] for d in fuzzy_disagreements])

        all_labels = set(rule_errors.keys()) | set(fuzzy_errors.keys())
        rule_counts = [rule_errors.get(label, 0) for label in all_labels]
        fuzzy_counts = [fuzzy_errors.get(label, 0) for label in all_labels]

        error_fig = go.Figure()
        error_fig.add_trace(go.Bar(
            name='Rule-based Errors',
            x=list(all_labels),
            y=rule_counts,
            text=rule_counts,
            textposition='outside',
            marker_color='#667eea'
        ))
        error_fig.add_trace(go.Bar(
            name='Fuzzy Matching Errors',
            x=list(all_labels),
            y=fuzzy_counts,
            text=fuzzy_counts,
            textposition='outside',
            marker_color='#764ba2'
        ))
        error_fig.update_layout(
            title={
                'text': "Error Count by True Label Category",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            title_font_size=20,
            xaxis_title="True Label",
            yaxis_title="Number of Errors",
            barmode='group',
            height=700,  # BIGGER
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return html.Div([
            dcc.Graph(figure=error_fig, style={'height': '750px'}),  # BIGGER

            dbc.Row([
                dbc.Col([
                    html.H5("Rule-based Improvement Suggestions", 
                           style={'fontWeight': '600', 'fontSize': '2rem', 'marginBottom': '1rem', 'textAlign': 'center'}),
                    create_suggestions_display(rule_suggestions)
                ], md=6),
                dbc.Col([
                    html.H5("Fuzzy Matching Improvement Suggestions", 
                           style={'fontWeight': '600', 'fontSize': '2rem', 'marginBottom': '1rem', 'textAlign': 'center'}),
                    create_suggestions_display(fuzzy_suggestions)
                ], md=6)
            ], className="mt-4")
        ])

    except Exception as e:
        return dbc.Alert(f"Error displaying error analysis: {str(e)}", color="danger")

def create_suggestions_display(suggestions):
    """Create display for improvement suggestions."""
    if not suggestions:
        return html.P("No specific suggestions available.", 
                     style={'fontSize': '1.1rem', 'fontStyle': 'italic'})

    suggestion_cards = []
    for suggestion in suggestions:
        color_map = {"high": "danger", "medium": "warning", "low": "info"}
        color = color_map[suggestion['priority']]

        card = dbc.Card([
            dbc.CardBody([
                html.H6(f"Error Pattern: {suggestion['error_pattern']}", 
                        className="card-title", 
                        style={'fontWeight': '700', 'fontSize': '1.3rem', 'marginBottom': '1rem', 'textAlign': 'center'}),
                html.P(suggestion['suggestion'], 
                    style={'fontSize': '1.1rem', 'marginBottom': '0.8rem', 'lineHeight': '1.4'}),
                html.Small(f"Frequency: {suggestion['frequency']} errors", 
                        className="text-muted",
                        style={'fontSize': '1rem', 'fontWeight': '500'})
            ], style={'padding': '1.5rem'})
        ], color=color, outline=True, className="mb-3", style={'borderRadius': '12px'})

        suggestion_cards.append(card)

    return html.Div(suggestion_cards)

def create_rule_crud_interface():
    """Create rule CRUD interface."""
    current_rules = []
    for label, rule_data in rule_classifier.rules.items():
        current_rules.append({
            'Label': label,
            'Rule Expression': rule_data['rule'],
            'Weight': rule_data['weight'],
            'Description': rule_data['description']
        })

    rules_table = dash_table.DataTable(
        id='rules-table',
        data=current_rules,
        columns=[
            {"name": "Label", "id": "Label", "editable": True},
            {"name": "Rule Expression", "id": "Rule Expression", "editable": True},
            {"name": "Weight", "id": "Weight", "editable": True, "type": "numeric"},
            {"name": "Description", "id": "Description", "editable": True},
        ],
        style_cell={'textAlign': 'left', 'fontSize': 14, 'padding': '12px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'fontSize': 16},
        style_data={'height': 'auto', 'lineHeight': '20px'},
        page_size=10,
        editable=True,
        row_deletable=True
    )

    # Add delete buttons to each row
    for i, rule in enumerate(current_rules):
        current_rules[i]['Actions'] = f'[Delete](#{i})'

    return html.Div([
        html.H5("Rule-Based Classifier Management", 
               style={'fontWeight': '600', 'fontSize': '1.8rem', 'marginBottom': '2rem' , 'textAlign': 'center'}),

        html.H6("Current Rules:", 
               style={'fontWeight': '600', 'fontSize': '1.4rem', 'marginBottom': '1rem'}),
        rules_table,

        dbc.Button("Update All Rules", id="update-rules-btn", color="primary", 
              className="mt-3", style={'width': '200px'}),

        html.Hr(style={'margin': '2rem 0'}),

        html.H6("Add New Rule:", 
               style={'fontWeight': '600', 'fontSize': '1.4rem', 'marginBottom': '1.5rem'}),
        dbc.Row([
            dbc.Col([
                html.Label("Label:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-rule-label", type="text", 
                         placeholder="e.g., Technical Issue_Bug Report",
                         style={'width': '100%'})
            ], md=6),
            dbc.Col([
                html.Label("Weight:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-rule-weight", type="number", value=1.0, step=0.1,
                         style={'width': '100%'})
            ], md=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Boolean Rule Expression:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-rule-expression", type="text",
                         placeholder="e.g., (bug OR error) AND NOT login",
                         style={'width': '100%'})
            ], md=12)
        ], className="mt-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Description:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-rule-description", type="text",
                         placeholder="Brief description of this rule",
                         style={'width': '100%'})
            ], md=8),
            dbc.Col([
                dbc.Button("Add Rule", id="add-rule-btn", color="success", 
                          className="mt-4", style={'width': '100%'})
            ], md=4)
        ], className="mt-3"),

        html.Div(id="rule-crud-feedback", className="mt-3")
    ])

def create_fuzzy_crud_interface():
    """Create fuzzy matching CRUD interface."""
    examples_data = []
    for label, examples_list in fuzzy_classifier.examples.items():
        for example in examples_list:
            examples_data.append({
                'Label': label,
                'Example Text': example[:100] + "..." if len(example) > 100 else example
            })

    examples_table = dash_table.DataTable(
        id='examples-table',
        data=examples_data,
        columns=[
            {"name": "Label", "id": "Label", "editable": True},
            {"name": "Example Text", "id": "Example Text", "editable": True}
        ],
        style_cell={'textAlign': 'left', 'fontSize': 14, 'padding': '12px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'fontSize': 16},
        style_data={'height': 'auto', 'lineHeight': '20px'},
        page_size=10,
        editable=True,
        row_deletable=True
    )

    return html.Div([
        html.H5("Fuzzy Matching Examples Management", 
               style={'fontWeight': '600', 'fontSize': '1.8rem', 'marginBottom': '2rem', 'textAlign': 'center'}),

        html.H6("Current Examples:", 
               style={'fontWeight': '600', 'fontSize': '1.4rem', 'marginBottom': '1rem'}),
        examples_table,

        html.Hr(style={'margin': '2rem 0'}),

        html.H6("Add New Example:", 
               style={'fontWeight': '600', 'fontSize': '1.4rem', 'marginBottom': '1.5rem'}),
        dbc.Row([
            dbc.Col([
                html.Label("Label:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-example-label", type="text",
                         placeholder="e.g., Technical Issue_Bug Report",
                         style={'width': '100%'})
            ], md=6),
            dbc.Col([
                dbc.Button("Add Example", id="add-example-btn", color="success", 
                          className="mt-4", style={'width': '100%'})
            ], md=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Example Text:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Textarea(id="new-example-text",
                           placeholder="Enter example text that represents this label...",
                           style={'width': '100%', 'height': 120})
            ], md=12)
        ], className="mt-3"),

        html.Div(id="example-crud-feedback", className="mt-3")
    ])

def create_test_interface():
    """Create testing interface."""
    return html.Div([
        html.H5("Test Environment - Sandbox Mode", 
               style={'fontWeight': '600', 'fontSize': '1.8rem', 'marginBottom': '2rem', 'textAlign': 'center'}),

        dbc.Row([
            dbc.Col([
                html.Label("Test Text:", style={'fontWeight': '600', 'fontSize': '1.2rem'}),
                dcc.Textarea(id="test-text-input",
                           placeholder="Enter text to test classification...",
                           style={'width': '100%', 'height': 150})
            ], md=8),
            dbc.Col([
                dbc.Button("Test Classifications", id="test-classify-btn",
                         color="primary", size="lg", className="mt-4",
                         style={'width': '100%', 'height': '60px'})
            ], md=4)
        ]),

        html.Hr(style={'margin': '2rem 0'}),

        html.Div(id="test-results-display")
    ])



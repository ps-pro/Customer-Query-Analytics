# ============================================================================
# IMPORTS
# ============================================================================
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

from RuleBased.classifier import BaselineRuleClassifier
from FuzzyMatching.classifier import FuzzyMatchingClassifier
from HITL.analyzer import HITLAnalyzer
from utils.helper import (
    create_performance_overview_tab,
    create_error_analysis_tab,
    create_crud_management_tab,
    create_rule_crud_interface,
    create_fuzzy_crud_interface,
    create_test_interface
)


# ============================================================================
# DATA LOADING
# ============================================================================
print("[INFO] Loading agreement data from CSV...")
agreement_df = pd.read_csv('data.csv')
print(f"[DEBUG] Loaded {len(agreement_df)} annotations")

# ============================================================================
# COMPONENT INITIALIZATION
# ============================================================================
print("[INFO] Initializing HITL Demonstration Components...")

rule_classifier = BaselineRuleClassifier()
fuzzy_classifier = FuzzyMatchingClassifier(similarity_method='character')
hitl_analyzer = HITLAnalyzer(agreement_df)

# ============================================================================
# DASH APP SETUP
# ============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Enhanced CSS with purple gradients and proper theme handling
app.index_string =  open('page.html').read()


# ============================================================================
# APP LAYOUT - FIXED
# ============================================================================
app.layout = dbc.Container([
    # Data stores for persistence across tabs - FIXED: Added missing stores
    dcc.Store(id='theme-store', data='light'),
    dcc.Store(id='analysis-results-store'),
    dcc.Store(id='error-analysis-store'),
    
    # Header Section with Theme Toggle
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H1("Human-in-the-Loop Baseline Model Demonstration", 
                        id="main-title",
                        style={
                            'margin': 0, 
                            'fontWeight': '800', 
                            'letterSpacing': '-1px',
                            'fontSize': '3rem',
                            'textAlign': 'center',
                            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent',
                            'backgroundClip': 'text'
                        }),
                    html.P("Advanced Rule-Based and Fuzzy Matching Classifiers with Interactive Training",
                        id="main-subtitle",
                        style={
                            'margin': 0, 
                            'opacity': 0.7, 
                            'fontSize': '1.2rem',
                            'textAlign': 'center',
                            'fontWeight': '400'
                        })
                ], style={'textAlign': 'center', 'flex': 1}),
                
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button("Light", id="light-theme-btn", size="sm", outline=True, color="secondary"),
                        dbc.Button("Dark", id="dark-theme-btn", size="sm", outline=True, color="secondary")
                    ], size="sm")
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={
                'display': 'flex', 
                'justifyContent': 'space-between', 
                'alignItems': 'center',
                'padding': '2rem 0 1.5rem 0',
                'borderBottom': '1px solid #e9ecef'
            }) 
        ], width=12)
    ]),
    
    # Navigation Tabs - FIXED: Correct tab IDs
    dbc.Row([
        dbc.Col([
            html.Div([
               dbc.Tabs([
                    dbc.Tab(
                        label="Performance Overview", 
                        tab_id="performance-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="Error Analysis & Opportunities",
                        tab_id="error-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="HITL Rule Management (CRUD)", 
                        tab_id="crud-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    )
                ], 
                id="main-hitl-tabs",  # FIXED: Correct ID
                active_tab="performance-tab",  # FIXED: Correct active tab
                style={'borderBottom': '1px solid #dee2e6', 'marginBottom': '0'})
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ])
    ], style={'marginTop': '1rem'}),
    
    # Main Content Area - Full Width
    dbc.Row([
        dbc.Col([
            html.Div(
                id="hitl-tab-content",
                style={
                    'minHeight': '80vh',
                    'padding': '2rem 0'
                }
            )
        ], width=12)
    ], style={'margin': '0'}),

], 
fluid=True, 
id="main-container",
style={
    'padding': '0 3rem',
    'maxWidth': '100%',
    'minHeight': '100vh'
})

# ============================================================================
# CALLBACKS - FIXED
# ============================================================================

# Theme switching callback - NEW
@app.callback(
    [Output('theme-store', 'data'),
     Output('light-theme-btn', 'className'),
     Output('dark-theme-btn', 'className')],
    [Input('light-theme-btn', 'n_clicks'),
     Input('dark-theme-btn', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(light_clicks, dark_clicks, current_theme):
    """Toggle between light and dark themes."""
    ctx = dash.callback_context
    if not ctx.triggered:
        if current_theme == 'light':
            return 'light', 'btn btn-outline-secondary active', 'btn btn-outline-secondary'
        else:
            return 'dark', 'btn btn-outline-secondary', 'btn btn-outline-secondary active'
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'light-theme-btn':
        return 'light', 'btn btn-outline-secondary active', 'btn btn-outline-secondary'
    elif button_id == 'dark-theme-btn':
        return 'dark', 'btn btn-outline-secondary', 'btn btn-outline-secondary active'
    
    return current_theme, 'btn btn-outline-secondary', 'btn btn-outline-secondary'

# Apply theme to body - NEW
app.clientside_callback(
    """
    function(theme) {
        document.body.className = theme === 'dark' ? 'theme-dark' : 'theme-light';
        return '';
    }
    """,
    Output('main-container', 'className'),
    [Input('theme-store', 'data')]
)

# Main tab content callback - FIXED
@app.callback(
    Output("hitl-tab-content", "children"),
    [Input("main-hitl-tabs", "active_tab"),  # FIXED: Correct tab ID
     Input("analysis-results-store", "data"),
     Input("error-analysis-store", "data")]
)
def render_hitl_tab_content(active_tab, analysis_data, error_data):
    """Render content based on selected tab with persistent data."""

    if active_tab == "performance-tab":
        return create_performance_overview_tab(analysis_data)
    elif active_tab == "error-tab":
        return create_error_analysis_tab(error_data)
    elif active_tab == "crud-tab":
        return create_crud_management_tab()
    else:
        return html.Div("Tab content not found")

# Performance analysis callback
@app.callback(
    [Output("analysis-results-store", "data"),
     Output("error-analysis-store", "data")],
    [Input("analyze-performance-btn", "n_clicks")],
    [State("similarity-method-radio", "value")]
)
def update_performance_analysis(n_clicks, similarity_method):
    """Update performance analysis and store results."""

    if n_clicks is None:
        return None, None

    print(f"[INFO] Starting HITL performance analysis with {similarity_method} similarity")

    try:
        fuzzy_classifier.set_similarity_method(similarity_method)
        comparison_results = hitl_analyzer.compare_classifiers(rule_classifier, fuzzy_classifier)

        rule_disagreements, rule_error_patterns = hitl_analyzer.identify_error_patterns(
            comparison_results['rule_based'], "Rule-based"
        )
        fuzzy_disagreements, fuzzy_error_patterns = hitl_analyzer.identify_error_patterns(
            comparison_results['fuzzy_matching'], "Fuzzy Matching"
        )

        rule_suggestions = hitl_analyzer.suggest_improvements(rule_error_patterns, "rule_based")
        fuzzy_suggestions = hitl_analyzer.suggest_improvements(fuzzy_error_patterns, "fuzzy_matching")

        analysis_data = {
            'comparison_results': comparison_results,
            'similarity_method': similarity_method
        }

        error_data = {
            'rule_disagreements': rule_disagreements,
            'fuzzy_disagreements': fuzzy_disagreements,
            'rule_suggestions': rule_suggestions,
            'fuzzy_suggestions': fuzzy_suggestions
        }

        return analysis_data, error_data

    except Exception as e:
        print(f"[ERROR] Performance analysis failed: {str(e)}")
        return None, None
    

@app.callback(
    Output("rule-crud-feedback", "children"),
    [Input("update-rules-btn", "n_clicks"),
     Input("rules-table", "data")],
    [State("rules-table", "data")]
)
def handle_rules_crud(update_clicks, table_data, current_data):
    """Handle rules table updates."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "update-rules-btn" and update_clicks:
        try:
            # Update the rule classifier with current table data
            rule_classifier.rules = {}
            for row in current_data:
                rule_classifier.rules[row['Label']] = {
                    'rule': row['Rule Expression'],
                    'weight': float(row['Weight']),
                    'description': row['Description'],
                    'keywords': []
                }
            
            return dbc.Alert("All rules updated successfully!", color="success", dismissable=True)
        except Exception as e:
            return dbc.Alert(f"Error updating rules: {str(e)}", color="danger", dismissable=True)
    
    return ""

# Examples table CRUD operations  
@app.callback(
    Output("example-crud-feedback", "children"),
    [Input("examples-table", "data"),
     Input("examples-table", "data_previous")]
)
def update_examples_table(current_data, previous_data):
    """Handle examples table updates and deletions."""
    if current_data != previous_data and previous_data is not None:
        # Update the fuzzy classifier with new data
        fuzzy_classifier.examples = {}
        for row in current_data:
            label = row['Label']
            if label not in fuzzy_classifier.examples:
                fuzzy_classifier.examples[label] = []
            fuzzy_classifier.examples[label].append(row['Example Text'])
        
        # Reset vectors for semantic similarity
        if fuzzy_classifier.similarity_method == 'semantic':
            fuzzy_classifier.example_vectors = None
            fuzzy_classifier._fit_semantic_vectors()
        
        return dbc.Alert("Examples updated successfully!", color="success", dismissable=True)
    return ""

# CRUD sub-tabs callback
@app.callback(
    Output("crud-sub-content", "children"),
    Input("crud-sub-tabs", "active_tab")
)
def render_crud_sub_content(active_tab):
    """Render CRUD sub-tab content."""

    if active_tab == "rule-crud-tab":
        return create_rule_crud_interface()
    elif active_tab == "fuzzy-crud-tab":
        return create_fuzzy_crud_interface()
    elif active_tab == "test-crud-tab":
        return create_test_interface()
    else:
        return html.Div("CRUD content not found")

# Test classification callback - ENHANCED
@app.callback(
    Output("test-results-display", "children"),
    [Input("test-classify-btn", "n_clicks")],
    [State("test-text-input", "value")]
)
def test_classification(n_clicks, test_text):
    """Test classification on input text."""

    if n_clicks is None or not test_text:
        return html.P("Enter text and click 'Test Classifications' to see results.", 
                     style={'fontSize': '1.1rem', 'textAlign': 'center', 'marginTop': '2rem'})

    try:
        rule_pred, rule_conf, rule_detail = rule_classifier.predict_single(test_text)
        fuzzy_pred, fuzzy_conf, fuzzy_detail = fuzzy_classifier.predict_single(test_text)

        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Rule-based Classification", style={'color': 'white', 'fontWeight': '600'})),
                    dbc.CardBody([
                        html.P(f"Predicted Label: {rule_pred}", style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                        html.P(f"Confidence: {rule_conf:.3f}", style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                        html.P(f"Matched Keywords: {', '.join(rule_detail.get('matched_keywords', []))}" if rule_detail else "No keywords matched", 
                              style={'fontSize': '1.1rem'}),
                        html.P(f"Rule Fired: {rule_detail.get('rule_fired', 'None')}" if rule_detail else "No rule fired", 
                              style={'fontSize': '1rem', 'fontStyle': 'italic'})
                    ])
                ])
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Fuzzy Matching Classification", style={'color': 'white', 'fontWeight': '600'})),
                    dbc.CardBody([
                        html.P(f"Predicted Label: {fuzzy_pred}", style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                        html.P(f"Confidence: {fuzzy_conf:.3f}", style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                        html.P(f"Best Match: {fuzzy_detail.get('best_example', 'None')[:100]}..." if fuzzy_detail and fuzzy_detail.get('best_example') else "No match found", 
                              style={'fontSize': '1.1rem'}),
                        html.P(f"Method: {fuzzy_detail.get('method', 'Unknown')}" if fuzzy_detail else "Unknown method", 
                              style={'fontSize': '1rem', 'fontStyle': 'italic'})
                    ])
                ])
            ], md=6)
        ])

    except Exception as e:
        return dbc.Alert(f"Error testing classification: {str(e)}", color="danger")

# ============================================================================
# RUN APP
# ============================================================================

# if __name__ == '__main__':
#     app.run(debug=True, host='127.0.0.1', port=8053)
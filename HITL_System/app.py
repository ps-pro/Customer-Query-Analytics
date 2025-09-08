# app.py - Complete HITL System with Real Data
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import time
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from database.init_db import DatabaseInitializer
from utils.database_manager import DatabaseManager
from models.classifiers import EnhancedRuleClassifier, EnhancedFuzzyClassifier, UncertaintyDetector
from models.active_learning import ActiveLearningManager
from utils.evaluation import HITLEvaluator

# ============================================================================
# SYSTEM INITIALIZATION WITH REAL DATA
# ============================================================================
print("[INFO] Initializing HITL System with Real Data...")

# Initialize database
db_initializer = DatabaseInitializer()
try:
    db_initializer.full_initialization()
except Exception as e:
    print(f"[WARNING] Database initialization issue: {e}")

# Initialize components
db_manager = DatabaseManager()
uncertainty_detector = UncertaintyDetector()
active_learning = ActiveLearningManager(db_manager, uncertainty_detector)
evaluator = HITLEvaluator(db_manager)

# Load models from database
current_rules = db_manager.get_current_rules()
current_examples = db_manager.get_current_examples()

rule_classifier = EnhancedRuleClassifier(current_rules)
fuzzy_classifier = EnhancedFuzzyClassifier(current_examples, 'character')

print(f"[INFO] Loaded {len(current_rules)} rules and {len(current_examples)} example categories")

# ============================================================================
# PROCESS REAL DATA TO CREATE UNCERTAIN CASES
# ============================================================================
def populate_initial_uncertain_cases():
    """Process real data to create initial uncertain cases for annotation."""
    print("[INFO] Processing real data to find uncertain cases...")
    
    # Get training annotations from database (these are your real consensus labels)
    training_annotations = db_manager.get_annotations_for_training()
    
    if not training_annotations:
        print("[WARNING] No training annotations found in database")
        return
    
    # Take a sample of texts to process (not all 3870 at once)
    sample_size = min(200, len(training_annotations))
    sample_annotations = training_annotations[:sample_size]
    
    texts = [ann['text'] for ann in sample_annotations]
    true_labels = [ann['human_label'] for ann in sample_annotations]
    
    print(f"[INFO] Processing {len(texts)} texts through classifiers...")
    
    # Get predictions from both classifiers
    rule_results = rule_classifier.predict_batch_with_uncertainty(texts)
    fuzzy_results = fuzzy_classifier.predict_batch_with_uncertainty(texts)
    
    # Find uncertain cases
    uncertain_count = 0
    for i, (text, true_label) in enumerate(zip(texts, true_labels)):
        rule_result = rule_results[i]
        fuzzy_result = fuzzy_results[i]
        
        # Check if case is uncertain
        is_uncertain, uncertainty_score, reason = uncertainty_detector.is_uncertain(rule_result, fuzzy_result)
        
        if is_uncertain:
            # Add to uncertain cases queue
            text_hash = db_manager.add_uncertain_case(
                text=text,
                rule_pred=rule_result[0],
                rule_conf=rule_result[1],
                fuzzy_pred=fuzzy_result[0],
                fuzzy_conf=fuzzy_result[1],
                uncertainty_score=uncertainty_score,
                disagreement=(rule_result[0] != fuzzy_result[0])
            )
            uncertain_count += 1
    
    print(f"[INFO] Created {uncertain_count} uncertain cases from {len(texts)} processed texts")

# Populate initial uncertain cases on startup
populate_initial_uncertain_cases()

# ============================================================================
# DASH APP SETUP WITH ORIGINAL STYLING
# ============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Same CSS styling as original app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Light Theme */
            .theme-light {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --text-primary: #212529;
                --text-secondary: #6c757d;
                --border-color: #dee2e6;
                --card-bg: #ffffff;
                --shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
                --purple-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --purple-gradient-hover: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
                --purple-shadow: rgba(102, 126, 234, 0.4);
            }
            
            /* Dark Theme */
            .theme-dark {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --text-primary: #ffffff;
                --text-secondary: #cbd5e0;
                --border-color: #4a5568;
                --card-bg: #2d2d2d;
                --shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.3);
                --purple-gradient: linear-gradient(135deg, #805ad5 0%, #9f7aea 100%);
                --purple-gradient-hover: linear-gradient(135deg, #6b46c1 0%, #8b5cf6 100%);
                --purple-shadow: rgba(139, 92, 246, 0.4);
            }
            
            /* Apply theme variables */
            body {
                background-color: var(--bg-primary) !important;
                color: var(--text-primary) !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
                transition: all 0.3s ease;
            }
            
            #main-container {
                background-color: var(--bg-primary);
                color: var(--text-primary);
            }
            
            /* Purple Gradient Buttons */
            .btn-primary, .btn.btn-primary {
                background: var(--purple-gradient) !important;
                border: none !important;
                color: white !important;
                font-weight: 600 !important;
                border-radius: 12px !important;
                padding: 12px 24px !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2) !important;
                transition: all 0.3s ease !important;
                transform: translateY(0) !important;
            }

            .btn-primary:hover, .btn.btn-primary:hover {
                background: var(--purple-gradient-hover) !important;
                transform: translateY(-3px) !important;
                box-shadow: 0 8px 25px var(--purple-shadow) !important;
                color: white !important;
            }

            .btn-success {
                background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
                border: none !important;
                color: white !important;
                border-radius: 12px !important;
                transition: all 0.3s ease !important;
            }
            
            /* Card styling */
            .card {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                box-shadow: var(--shadow) !important;
                border-radius: 15px !important;
                transition: all 0.3s ease !important;
            }

            .card:hover {
                box-shadow: 0 12px 30px rgba(102, 126, 234, 0.15) !important;
                transform: translateY(-2px) !important;
            }
            
            .card-header {
                background: var(--purple-gradient) !important;
                border-bottom: none !important;
                color: white !important;
                font-weight: 600 !important;
                border-radius: 15px 15px 0 0 !important;
            }
            
            .card-body {
                color: var(--text-primary) !important;
                padding: 2rem !important;
            }

            /* Form controls */
            .form-control, .form-select, input, textarea {
                background-color: var(--card-bg) !important;
                border: 2px solid var(--border-color) !important;
                color: var(--text-primary) !important;
                border-radius: 12px !important;
                padding: 12px 16px !important;
                transition: all 0.3s ease !important;
            }
            
            .form-control:focus, .form-select:focus, input:focus, textarea:focus {
                background-color: var(--card-bg) !important;
                border-color: #667eea !important;
                color: var(--text-primary) !important;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
            }

            /* Text and label colors */
            label, .form-label, p, h1, h2, h3, h4, h5, h6, span, div {
                color: var(--text-primary) !important;
            }

            .text-muted, .text-secondary {
                color: var(--text-secondary) !important;
            }

            /* Tab styling */
            .nav-tabs {
                border-bottom: 2px solid var(--border-color) !important;
                background: transparent !important;
            }

            .nav-tabs .nav-link {
                color: var(--text-secondary) !important;
                border: none !important;
                background: transparent !important;
                font-weight: 600 !important;
                font-size: 1.1rem !important;
                padding: 1rem 2rem !important;
                border-radius: 12px 12px 0 0 !important;
                transition: all 0.3s ease !important;
            }

            .nav-tabs .nav-link:hover {
                background: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
                transform: translateY(-2px) !important;
            }

            .nav-tabs .nav-link.active {
                background: var(--purple-gradient) !important;
                color: white !important;
                border: none !important;
                font-weight: 700 !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
            }

            /* Slider styling */
            .rc-slider-track {
                background: var(--purple-gradient) !important;
                height: 8px !important;
            }

            .rc-slider-handle {
                border: 3px solid #667eea !important;
                background: white !important;
                width: 24px !important;
                height: 24px !important;
                margin-top: -8px !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
            }

            .rc-slider-rail {
                background: var(--border-color) !important;
                height: 8px !important;
            }

            /* Radio button styling */
            input[type="radio"] {
                appearance: none !important;
                -webkit-appearance: none !important;
                width: 20px !important;
                height: 20px !important;
                border: 2px solid var(--border-color) !important;
                border-radius: 50% !important;
                background-color: var(--card-bg) !important;
                position: relative !important;
                cursor: pointer !important;
                outline: none !important;
            }

            input[type="radio"]:checked {
                border-color: #667eea !important;
            }

            input[type="radio"]:checked::before {
                content: '' !important;
                position: absolute !important;
                top: 50% !important;
                left: 50% !important;
                width: 12px !important;
                height: 12px !important;
                border-radius: 50% !important;
                background-color: #667eea !important;
                transform: translate(-50%, -50%) !important;
            }

            /* Plot styling - KEEP WHITE BACKGROUNDS */
            .js-plotly-plot {
                border-radius: 15px !important;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15) !important;
                background: #ffffff !important;
                min-height: 600px !important;
            }

            /* Hide page number input field */
            .dash-table-container .previous-next-container input.current-page {
                display: none !important;
            }
            
            /* Table styling */
            .dash-table-container {
                background-color: var(--card-bg) !important;
                border-radius: 15px !important;
                overflow: hidden !important;
                border: 1px solid var(--border-color) !important;
            }

            .dash-table-container th {
                background: var(--purple-gradient) !important;
                color: white !important;
                font-weight: 600 !important;
            }

            .dash-table-container td {
                background-color: var(--card-bg) !important;
                color: var(--text-primary) !important;
                border-color: var(--border-color) !important;
            }

            /* Alert styling */
            .alert {
                border-radius: 12px !important;
                border: none !important;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
            }

            .alert-info {
                background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%) !important;
                color: white !important;
            }

            .alert-success {
                background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
                color: white !important;
            }

            .alert-warning {
                background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%) !important;
                color: white !important;
            }

            /* Confidence rating buttons */
            .confidence-btn {
                border-radius: 8px !important;
                margin: 2px !important;
                border: 2px solid var(--border-color) !important;
                transition: all 0.3s ease !important;
            }

            .confidence-btn.selected {
                background: var(--purple-gradient) !important;
                color: white !important;
                border-color: #667eea !important;
            }
        </style>
    </head>
    <body class="theme-light">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============================================================================
# APP LAYOUT
# ============================================================================
app.layout = dbc.Container([
    # Data stores
    dcc.Store(id='theme-store', data='light'),
    dcc.Store(id='selected-confidence-store', data={}),
    dcc.Store(id='annotation-batch-store'),
    
    # Update interval
    dcc.Interval(id='update-interval', interval=5000, n_intervals=0),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H1("HITL Active Learning System", 
                        id="main-title",
                        style={
                            'margin': 0, 
                            'fontWeight': '800', 
                            'letterSpacing': '-1px',
                            'fontSize': '2.8rem',
                            'textAlign': 'center',
                            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent',
                            'backgroundClip': 'text'
                        }),
                    html.P("Real-time Uncertainty Detection and Human-in-the-Loop Learning",
                        id="main-subtitle",
                        style={
                            'margin': 0, 
                            'opacity': 0.7, 
                            'fontSize': '1.1rem',
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
    
    # Navigation Tabs
    dbc.Row([
        dbc.Col([
            html.Div([
               dbc.Tabs([
                    dbc.Tab(label="HITL Dashboard", tab_id="dashboard-tab"),
                    dbc.Tab(label="Human Annotation", tab_id="annotation-tab"),
                    dbc.Tab(label="Performance Analysis", tab_id="performance-tab"),
                    dbc.Tab(label="Model Management", tab_id="management-tab"),
                    dbc.Tab(label="System Controls", tab_id="controls-tab")
                ], 
                id="main-tabs",
                active_tab="dashboard-tab",
                style={'borderBottom': '1px solid #dee2e6', 'marginBottom': '0'})
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ])
    ], style={'marginTop': '1rem'}),
    
    # Main Content
    dbc.Row([
        dbc.Col([
            html.Div(id="tab-content", style={'minHeight': '80vh', 'padding': '2rem 0'})
        ], width=12)
    ], style={'margin': '0'}),

], 
fluid=True, 
id="main-container",
style={'padding': '0 3rem', 'maxWidth': '100%', 'minHeight': '100vh'})

# ============================================================================
# TAB CONTENT FUNCTIONS
# ============================================================================

def create_dashboard_tab():
    """Create HITL dashboard with real metrics."""
    
    # Get real system metrics
    queue_count = db_manager.get_uncertain_cases_count()
    annotation_count = db_manager.get_annotation_count()
    config = db_manager.get_all_config()
    
    return html.Div([
        html.H4("HITL Active Learning Dashboard", 
                style={'textAlign': 'center', 'marginBottom': '2rem', 'fontWeight': '700'}),
        
        # System Status Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Uncertain Cases Queue", style={'color': 'white', 'textAlign': 'center', 'margin': '0'})),
                    dbc.CardBody([
                        html.H3(f"{queue_count}", style={'textAlign': 'center', 'fontSize': '2.5rem', 'fontWeight': '700'}),
                        html.P("Cases needing human annotation", style={'textAlign': 'center', 'marginBottom': '0'})
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Total Annotations", style={'color': 'white', 'textAlign': 'center', 'margin': '0'})),
                    dbc.CardBody([
                        html.H3(f"{annotation_count}", style={'textAlign': 'center', 'fontSize': '2.5rem', 'fontWeight': '700'}),
                        html.P("Human labels in database", style={'textAlign': 'center', 'marginBottom': '0'})
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Learning Status", style={'color': 'white', 'textAlign': 'center', 'margin': '0'})),
                    dbc.CardBody([
                        html.H3("ACTIVE", style={'textAlign': 'center', 'fontSize': '2rem', 'fontWeight': '700', 'color': '#38a169'}),
                        html.P("System actively learning", style={'textAlign': 'center', 'marginBottom': '0'})
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Uncertainty Threshold", style={'color': 'white', 'textAlign': 'center', 'margin': '0'})),
                    dbc.CardBody([
                        html.H3(f"{float(config.get('uncertainty_threshold', 0.6)):.1f}", 
                               style={'textAlign': 'center', 'fontSize': '2.5rem', 'fontWeight': '700'}),
                        html.P("Current detection threshold", style={'textAlign': 'center', 'marginBottom': '0'})
                    ])
                ])
            ], md=3)
        ], className="mb-4"),
        
        # Process More Data Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Process New Data for Uncertainty Detection", style={'color': 'white'})),
                    dbc.CardBody([
                        html.Label("Number of texts to process:", style={'fontWeight': '600'}),
                        dcc.Slider(
                            id="process-batch-size-slider",
                            min=50, max=500, step=50, value=100,
                            marks={i: str(i) for i in range(50, 551, 100)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        dbc.Button("Process More Data", id="process-data-btn", color="primary", 
                                  className="mt-3", style={'width': '100%'})
                    ])
                ])
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("System Performance", style={'color': 'white'})),
                    dbc.CardBody([
                        html.Div(id="dashboard-performance-chart")
                    ])
                ])
            ], md=6)
        ], className="mb-4"),
        
        # Recent Activity
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Recent Annotation Activity", style={'color': 'white'})),
                    dbc.CardBody([
                        html.Div(id="recent-activity-display")
                    ])
                ])
            ], md=12)
        ])
    ])

def create_annotation_tab():
    """Create human annotation interface."""
    
    return html.Div([
        html.H4("Human Annotation Interface", 
                style={'textAlign': 'center', 'marginBottom': '2rem', 'fontWeight': '700'}),
        
        # Control Panel
        dbc.Row([
            dbc.Col([
                html.Label("Batch Size for Annotation:", style={'fontWeight': '600'}),
                dcc.Slider(
                    id="annotation-batch-size-slider",
                    min=5, max=25, step=5, value=10,
                    marks={i: str(i) for i in range(5, 30, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], md=4),
            dbc.Col([
                html.Label("Auto-update models after:", style={'fontWeight': '600'}),
                dcc.Slider(
                    id="auto-update-frequency-slider",
                    min=1, max=10, step=1, value=3,
                    marks={i: str(i) for i in range(1, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], md=4),
            dbc.Col([
                dbc.Button("Load Annotation Batch", id="load-annotation-batch-btn", 
                          color="primary", size="lg", style={'width': '100%', 'marginTop': '1.5rem'})
            ], md=4)
        ], className="mb-4"),
        
        # Annotation Area
        html.Div(id="annotation-area"),
        
        # Feedback
        html.Div(id="annotation-feedback", className="mt-3")
    ])

def create_performance_tab():
    """Create performance analysis tab."""
    
    return html.Div([
        html.H4("Performance Analysis & HITL Effectiveness", 
                style={'textAlign': 'center', 'marginBottom': '2rem', 'fontWeight': '700'}),
        
        # Analysis Controls
        dbc.Row([
            dbc.Col([
                dbc.Button("Run Performance Analysis", id="run-analysis-btn", 
                          color="primary", size="lg", style={'width': '100%'})
            ], md=6),
            dbc.Col([
                html.Label("Enable Auto-Updates:", style={'fontWeight': '600'}),
                dbc.Switch(
                    id="enable-auto-updates-switch",
                    label="Automatically update models after annotations",
                    value=True,
                    style={'marginTop': '1rem'}
                )
            ], md=6)
        ], className="mb-4"),
        
        # Performance Results
        html.Div(id="performance-results")
    ])

def create_management_tab():
    """Create model management tab."""
    return html.Div([
        html.H4("Model Management", 
                style={'textAlign': 'center', 'marginBottom': '2rem', 'fontWeight': '700'}),
        
        dbc.Tabs([
            dbc.Tab(label="Rules Management", tab_id="rules-mgmt"),
            dbc.Tab(label="Examples Management", tab_id="examples-mgmt")
        ], id="mgmt-sub-tabs", active_tab="rules-mgmt"),
        
        html.Div(id="mgmt-sub-content", className="mt-4")
    ])

def create_controls_tab():
    """Create system controls tab."""
    config = db_manager.get_all_config()
    
    return html.Div([
        html.H4("System Controls & Configuration", 
                style={'textAlign': 'center', 'marginBottom': '2rem', 'fontWeight': '700'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Uncertainty Detection Settings", style={'color': 'white'})),
                    dbc.CardBody([
                        html.Label("Uncertainty Threshold:", style={'fontWeight': '600'}),
                        dcc.Slider(
                            id="uncertainty-threshold-slider",
                            min=0.3, max=0.9, step=0.05, 
                            value=float(config.get('uncertainty_threshold', 0.6)),
                            marks={i/10: f"{i/10:.1f}" for i in range(3, 10, 1)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.P("Higher = more selective (fewer uncertain cases)", 
                               style={'fontSize': '0.9rem', 'fontStyle': 'italic'}),
                        
                        html.Hr(),
                        
                        html.Label("Fuzzy Matching Method:", style={'fontWeight': '600'}),
                        dbc.RadioItems(
                            id="similarity-method-radio",
                            options=[
                                {"label": " Character-Level Similarity", "value": "character"},
                                {"label": " Semantic Similarity (TF-IDF)", "value": "semantic"}
                            ],
                            value=config.get('similarity_method', 'character'),
                            style={'fontSize': '1.1rem', 'marginTop': '1rem'}
                        )
                    ])
                ])
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Learning Controls", style={'color': 'white'})),
                    dbc.CardBody([
                        html.Label("Enable Real-time Learning:", style={'fontWeight': '600'}),
                        dbc.Switch(
                            id="realtime-learning-switch",
                            label="Update models immediately after each annotation",
                            value=True,
                            style={'marginBottom': '2rem'}
                        ),
                        
                        html.Label("Confidence Threshold for Auto-Rules:", style={'fontWeight': '600'}),
                        dcc.Slider(
                            id="auto-rule-confidence-slider",
                            min=0.7, max=1.0, step=0.05, 
                            value=float(config.get('min_confidence_for_auto_rule', 0.85)),
                            marks={i/10: f"{i/10:.1f}" for i in range(7, 11, 1)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Hr(),
                        
                        html.Label("Queue Management:", style={'fontWeight': '600'}),
                        dbc.Button("Clear Annotation Queue", id="clear-queue-btn", 
                                  color="warning", outline=True, className="mt-2 me-2"),
                        dbc.Button("Regenerate Uncertain Cases", id="regenerate-queue-btn", 
                                  color="info", outline=True, className="mt-2")
                    ])
                ])
            ], md=6)
        ], className="mb-4"),
        
        # Save Configuration
        dbc.Row([
            dbc.Col([
                dbc.Button("Save Configuration", id="save-config-btn", color="success", 
                          size="lg", style={'width': '100%'})
            ], md=12)
        ]),
        
        html.Div(id="config-feedback", className="mt-3")
    ])

# ============================================================================
# MAIN CALLBACKS
# ============================================================================

# Theme switching
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

# Apply theme to body
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

# Main tab content
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Render content based on selected tab."""
    if active_tab == "dashboard-tab":
        return create_dashboard_tab()
    elif active_tab == "annotation-tab":
        return create_annotation_tab()
    elif active_tab == "performance-tab":
        return create_performance_tab()
    elif active_tab == "management-tab":
        return create_management_tab()
    elif active_tab == "controls-tab":
        return create_controls_tab()
    else:
        return html.Div("Tab content not found")

# Process more data
@app.callback(
    Output("dashboard-performance-chart", "children"),
    [Input("process-data-btn", "n_clicks"),
     Input("update-interval", "n_intervals")],
    State("process-batch-size-slider", "value")
)
def process_more_data(n_clicks, n_intervals, batch_size):
    """Process more data through uncertainty detection."""
    ctx = dash.callback_context
    
    if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == 'process-data-btn' and n_clicks:
        # Process more data
        training_annotations = db_manager.get_annotations_for_training()
        
        if training_annotations:
            # Get next batch that hasn't been processed yet
            start_idx = min(200, len(training_annotations))  # Start after initial batch
            end_idx = min(start_idx + batch_size, len(training_annotations))
            
            if start_idx < len(training_annotations):
                batch_annotations = training_annotations[start_idx:end_idx]
                texts = [ann['text'] for ann in batch_annotations]
                
                # Process through active learning
                result = active_learning.process_new_texts(texts, rule_classifier, fuzzy_classifier)
                
                # Update uncertainty threshold adaptively
                active_learning.update_uncertainty_threshold_adaptive()
    
    # Always return current performance chart
    try:
        # Get current performance metrics
        rule_performance = evaluator.get_latest_performance('rule_based')
        fuzzy_performance = evaluator.get_latest_performance('fuzzy_matching')
        
        if rule_performance and fuzzy_performance:
            fig = go.Figure()
            
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            rule_values = [rule_performance.get(m, 0) for m in metrics]
            fuzzy_values = [fuzzy_performance.get(m, 0) for m in metrics]
            
            fig.add_trace(go.Bar(name='Rule-based', x=metrics, y=rule_values, marker_color='#667eea'))
            fig.add_trace(go.Bar(name='Fuzzy Matching', x=metrics, y=fuzzy_values, marker_color='#764ba2'))
            
            fig.update_layout(
                title="Current Classifier Performance",
                yaxis=dict(range=[0, 1]),
                barmode='group',
                height=300
            )
            
            return dcc.Graph(figure=fig)
        else:
            return html.P("No performance data available yet")
            
    except Exception as e:
        return html.P(f"Error loading performance: {str(e)}")

# Load annotation batch
@app.callback(
    [Output("annotation-area", "children"),
     Output("annotation-batch-store", "data")],
    Input("load-annotation-batch-btn", "n_clicks"),
    State("annotation-batch-size-slider", "value")
)
def load_annotation_batch(n_clicks, batch_size):
    """Load batch of uncertain cases for annotation."""
    if not n_clicks:
        return html.P("Click 'Load Annotation Batch' to start annotating uncertain cases"), None
    
    try:
        # Get uncertain cases from database
        cases = db_manager.get_uncertain_cases_batch(batch_size)
        
        if not cases:
            return dbc.Alert("No uncertain cases available! Process more data first.", color="info"), None
        
        # Create annotation cards
        annotation_cards = []
        
        for i, case in enumerate(cases):
            card = dbc.Card([
                dbc.CardHeader([
                    html.H6(f"Case {i+1} of {len(cases)}", style={'color': 'white', 'margin': 0, 'display': 'inline'}),
                    dbc.Badge(f"Uncertainty: {case['uncertainty_score']:.2f}", 
                             color="warning", style={'float': 'right'})
                ]),
                dbc.CardBody([
                    # Text to annotate
                    html.P(case['text'], style={
                        'fontSize': '1.1rem', 
                        'lineHeight': '1.6', 
                        'marginBottom': '1.5rem',
                        'padding': '1rem',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '8px',
                        'border': '1px solid #dee2e6'
                    }),
                    
                    # Model predictions
                    dbc.Row([
                        dbc.Col([
                            html.Label("Rule-based Prediction:", style={'fontWeight': '600'}),
                            html.P(f"{case['rule_prediction']} (confidence: {case['rule_confidence']:.2f})",
                                  style={'color': '#667eea', 'fontWeight': '500'})
                        ], md=6),
                        dbc.Col([
                            html.Label("Fuzzy Matching Prediction:", style={'fontWeight': '600'}),
                            html.P(f"{case['fuzzy_prediction']} (confidence: {case['fuzzy_confidence']:.2f})",
                                  style={'color': '#764ba2', 'fontWeight': '500'})
                        ], md=6)
                    ], className="mb-3"),
                    
                    html.Hr(),
                    
                    # Annotation controls
                    html.Label("Select Correct Label:", style={'fontWeight': '600', 'marginBottom': '1rem'}),
                    dcc.Dropdown(
                        id={'type': 'label-dropdown', 'index': i},
                        options=[
                            {'label': label, 'value': label} 
                            for label in sorted(list(current_rules.keys()) + ['Unknown'])
                        ],
                        placeholder="Choose the correct label...",
                        style={'marginBottom': '1.5rem'}
                    ),
                    
                    html.Label("Your Confidence:", style={'fontWeight': '600', 'marginBottom': '0.5rem'}),
                    dbc.ButtonGroup([
                        dbc.Button("1", id={'type': 'conf-btn', 'index': i, 'value': 1}, 
                                  outline=True, color="secondary", size="sm", className="confidence-btn"),
                        dbc.Button("2", id={'type': 'conf-btn', 'index': i, 'value': 2}, 
                                  outline=True, color="secondary", size="sm", className="confidence-btn"),
                        dbc.Button("3", id={'type': 'conf-btn', 'index': i, 'value': 3}, 
                                  outline=True, color="secondary", size="sm", className="confidence-btn"),
                        dbc.Button("4", id={'type': 'conf-btn', 'index': i, 'value': 4}, 
                                  outline=True, color="secondary", size="sm", className="confidence-btn"),
                        dbc.Button("5", id={'type': 'conf-btn', 'index': i, 'value': 5}, 
                                  outline=True, color="secondary", size="sm", className="confidence-btn"),
                    ], style={'marginBottom': '1rem'}),
                    html.Small("1=Very Uncertain, 5=Very Confident", className="text-muted d-block mb-3"),
                    
                    dbc.Button("Submit Annotation", 
                              id={'type': 'submit-annotation', 'index': i},
                              color="success", style={'width': '100%'})
                ])
            ], className="mb-4")
            
            annotation_cards.append(card)
        
        return html.Div(annotation_cards), cases
        
    except Exception as e:
        return dbc.Alert(f"Error loading annotation batch: {str(e)}", color="danger"), None

# Handle confidence button clicks
@app.callback(
    [Output({'type': 'conf-btn', 'index': MATCH, 'value': ALL}, 'className'),
     Output('selected-confidence-store', 'data')],
    [Input({'type': 'conf-btn', 'index': MATCH, 'value': ALL}, 'n_clicks')],
    [State('selected-confidence-store', 'data'),
     State({'type': 'conf-btn', 'index': MATCH, 'value': ALL}, 'id')],
    prevent_initial_call=True
)
def handle_confidence_selection(n_clicks_list, stored_confidence, button_ids):
    """Handle confidence button selection."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return ['confidence-btn'] * len(n_clicks_list), stored_confidence or {}
    
    # Find which button was clicked
    clicked_value = None
    case_index = None
    
    for i, n_clicks in enumerate(n_clicks_list):
        if n_clicks:
            clicked_value = button_ids[i]['value']
            case_index = button_ids[i]['index']
            break
    
    # Update stored confidence
    if stored_confidence is None:
        stored_confidence = {}
    stored_confidence[case_index] = clicked_value
    
    # Update button classes
    classes = []
    for i, button_id in enumerate(button_ids):
        if button_id['value'] == clicked_value:
            classes.append('confidence-btn selected')
        else:
            classes.append('confidence-btn')
    
    return classes, stored_confidence

# Submit annotation
@app.callback(
    Output("annotation-feedback", "children"),
    [Input({'type': 'submit-annotation', 'index': ALL}, 'n_clicks')],
    [State({'type': 'label-dropdown', 'index': ALL}, 'value'),
     State('selected-confidence-store', 'data'),
     State('annotation-batch-store', 'data'),
     State('auto-update-frequency-slider', 'value'),
     State('realtime-learning-switch', 'value')],
    prevent_initial_call=True
)
def submit_annotation(n_clicks_list, selected_labels, confidence_data, batch_data, auto_freq, realtime_learning):
    """Submit human annotation and update models."""
    ctx = dash.callback_context
    if not ctx.triggered or not batch_data:
        return ""
    
    # Find which submit button was clicked
    button_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
    case_index = button_id['index']
    
    if not n_clicks_list[case_index]:
        return ""
    
    # Validate inputs
    selected_label = selected_labels[case_index] if case_index < len(selected_labels) else None
    if not selected_label:
        return dbc.Alert("Please select a label before submitting", color="warning", dismissable=True)
    
    confidence_rating = confidence_data.get(case_index) if confidence_data else None
    if confidence_rating is None:
        return dbc.Alert("Please select a confidence rating before submitting", color="warning", dismissable=True)
    
    try:
        # Get case data
        case = batch_data[case_index]
        
        # Submit annotation to database
        annotation_id = db_manager.add_human_annotation(
            text_hash=case['text_hash'],
            text=case['text'],
            original_rule_pred=case['rule_prediction'],
            original_fuzzy_pred=case['fuzzy_prediction'],
            human_label=selected_label,
            confidence_rating=confidence_rating,
            annotation_time=1.0  # Default annotation time
        )
        
        # Check if we should update models
        total_annotations = db_manager.get_annotation_count()
        should_update = realtime_learning or (total_annotations % auto_freq == 0)
        
        feedback_message = f"Annotation submitted! Total annotations: {total_annotations}"
        
        if should_update:
            # Update models with new data
            global rule_classifier, fuzzy_classifier, current_rules, current_examples
            
            # Reload models from database
            current_rules = db_manager.get_current_rules()
            current_examples = db_manager.get_current_examples()
            
            rule_classifier = EnhancedRuleClassifier(current_rules)
            fuzzy_classifier = EnhancedFuzzyClassifier(current_examples, 
                db_manager.get_config('similarity_method', 'character'))
            
            feedback_message += " Models updated with new learning!"
            
            # Save model snapshot
            db_manager.save_model_snapshot(
                snapshot_name=f"annotation_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rules=current_rules,
                examples=current_examples,
                trigger_reason=f"human_annotation_{annotation_id}"
            )
        
        return dbc.Alert(feedback_message, color="success", dismissable=True)
        
    except Exception as e:
        return dbc.Alert(f"Error submitting annotation: {str(e)}", color="danger", dismissable=True)

# Run performance analysis
@app.callback(
    Output("performance-results", "children"),
    Input("run-analysis-btn", "n_clicks")
)
def run_performance_analysis(n_clicks):
    """Run comprehensive performance analysis."""
    if not n_clicks:
        return html.P("Click 'Run Performance Analysis' to evaluate HITL system effectiveness")
    
    try:
        # Get test data from annotations
        test_annotations = db_manager.get_annotations_for_training()[-100:]  # Use last 100 as test
        test_cases = [{'text': ann['text'], 'human_label': ann['human_label']} for ann in test_annotations]
        
        if not test_cases:
            return dbc.Alert("No test data available for analysis", color="warning")
        
        # Evaluate both classifiers
        comparison_results = evaluator.compare_classifiers(rule_classifier, fuzzy_classifier, test_cases)
        
        # Create performance visualization
        rule_metrics = comparison_results['rule_based']['overall_metrics']
        fuzzy_metrics = comparison_results['fuzzy_matching']['overall_metrics']
        
        fig = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        rule_values = [rule_metrics[m] for m in metrics]
        fuzzy_values = [fuzzy_metrics[m] for m in metrics]
        
        fig.add_trace(go.Bar(name='Rule-based', x=metrics, y=rule_values, 
                            text=[f'{v:.3f}' for v in rule_values], textposition='outside',
                            marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Fuzzy Matching', x=metrics, y=fuzzy_values,
                            text=[f'{v:.3f}' for v in fuzzy_values], textposition='outside', 
                            marker_color='#764ba2'))
        
        fig.update_layout(
            title="HITL System Performance vs Human Consensus",
            yaxis=dict(range=[0, 1]),
            barmode='group',
            height=500
        )
        
        # Get learning progress
        learning_progress = evaluator.track_learning_progress(30)
        
        return html.Div([
            dcc.Graph(figure=fig),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Rule-based Performance", style={'color': 'white'})),
                        dbc.CardBody([
                            html.P(f"Accuracy: {rule_metrics['accuracy']:.1%}", style={'fontSize': '1.2rem'}),
                            html.P(f"F1-Score: {rule_metrics['f1_score']:.3f}", style={'fontSize': '1.2rem'}),
                            html.P(f"Precision: {rule_metrics['precision']:.3f}", style={'fontSize': '1.2rem'})
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Fuzzy Matching Performance", style={'color': 'white'})),
                        dbc.CardBody([
                            html.P(f"Accuracy: {fuzzy_metrics['accuracy']:.1%}", style={'fontSize': '1.2rem'}),
                            html.P(f"F1-Score: {fuzzy_metrics['f1_score']:.3f}", style={'fontSize': '1.2rem'}),
                            html.P(f"Precision: {fuzzy_metrics['precision']:.3f}", style={'fontSize': '1.2rem'})
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Learning Effectiveness", style={'color': 'white'})),
                        dbc.CardBody([
                            html.P(f"Better Classifier: {comparison_results['comparison']['better_classifier'].replace('_', ' ').title()}", 
                                  style={'fontSize': '1.1rem'}),
                            html.P(f"Accuracy Difference: {comparison_results['comparison']['accuracy_diff']:.3f}", 
                                  style={'fontSize': '1.1rem'}),
                            html.P(f"Test Cases: {len(test_cases)}", style={'fontSize': '1.1rem'})
                        ])
                    ])
                ], md=4)
            ], className="mt-4")
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error running performance analysis: {str(e)}", color="danger")

# Save configuration
@app.callback(
    Output("config-feedback", "children"),
    Input("save-config-btn", "n_clicks"),
    [State("uncertainty-threshold-slider", "value"),
     State("similarity-method-radio", "value"),
     State("auto-rule-confidence-slider", "value")]
)
def save_configuration(n_clicks, uncertainty_threshold, similarity_method, auto_rule_confidence):
    """Save system configuration."""
    if not n_clicks:
        return ""
    
    try:
        # Update configuration in database
        db_manager.set_config('uncertainty_threshold', str(uncertainty_threshold))
        db_manager.set_config('similarity_method', similarity_method)
        db_manager.set_config('min_confidence_for_auto_rule', str(auto_rule_confidence))
        
        # Update system components
        global uncertainty_detector, fuzzy_classifier
        uncertainty_detector.update_threshold(uncertainty_threshold)
        fuzzy_classifier.set_similarity_method(similarity_method)
        
        return dbc.Alert("Configuration saved and applied successfully!", color="success", dismissable=True)
        
    except Exception as e:
        return dbc.Alert(f"Error saving configuration: {str(e)}", color="danger", dismissable=True)

# Update recent activity
@app.callback(
    Output("recent-activity-display", "children"),
    Input("update-interval", "n_intervals")
)
def update_recent_activity(n_intervals):
    """Update recent activity display."""
    try:
        recent_annotations = db_manager.get_recent_annotations(10)
        
        if not recent_annotations:
            return html.P("No recent annotation activity")
        
        activity_items = []
        for ann in recent_annotations:
            activity_items.append(
                html.Div([
                    html.Strong(f"{ann['human_label']} "),
                    html.Span(f"(confidence: {ann['confidence_rating']}/5) - "),
                    html.Small(ann['created_at'][:16])
                ], className="mb-2")
            )
        
        return html.Div(activity_items)
        
    except Exception as e:
        return html.P(f"Error loading activity: {str(e)}")

print("[INFO] HITL System with Real Data initialized!")
print("[INFO] Key Features:")
print("  ✓ Real SQLite database with your 3870 annotations")
print("  ✓ Uncertainty detection on real data")
print("  ✓ Human annotation interface with real uncertain cases")
print("  ✓ Automatic model updates based on human feedback")
print("  ✓ Performance tracking with real metrics")
print("  ✓ User controls for all parameters (sliders, switches, dropdowns)")
print("[INFO] Access the application at http://127.0.0.1:8054/")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8054)
# ========================================
# IMPORTS
# ========================================

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


from AgreementAnalysis.calculator import IAAAgreementCalculator
from AgreementAnalysis.visualizer import IAAAgreementVisualizer
from AgreementAnalysis.content import create_overall_tab_content


from FrequencyAnalysis.calculator import FrequencyAnalysisCalculator
from FrequencyAnalysis.visualizer import FrequencyAnalysisVisualizer
from FrequencyAnalysis.content import create_frequency_tab_content


from HierarchicalAnalysis.calculator import HierarchicalAnalysisCalculator
from HierarchicalAnalysis.visualizer import HierarchicalAnalysisVisualizer
from HierarchicalAnalysis.content import create_hierarchical_tab_content


from utils.helpers import get_alpha_interpretation, get_agreement_interpretation, get_comparison


# ========================================
# DATA LOADING
# ========================================

# Load data from CSV file
agreement_df = pd.read_csv('data.csv')
print(f"[INFO] Loaded {len(agreement_df)} annotations from data.csv")

# ========================================
# INITIALIZE COMPONENTS
# ========================================

calculator = IAAAgreementCalculator(agreement_df)
freq_calculator = FrequencyAnalysisCalculator(agreement_df)  
hier_calculator = HierarchicalAnalysisCalculator(agreement_df)


visualizer = IAAAgreementVisualizer()
freq_visualizer = FrequencyAnalysisVisualizer() 
hier_visualizer = HierarchicalAnalysisVisualizer()

# ========================================
# CREATE DASH APP
# ========================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# CRITICAL: This line is required for Plotly Cloud deployment
server = app.server

# ========================================
# STYLING CONSTANTS
# ========================================

app.index_string = open('page.html').read()

# ========================================
# LAYOUT DEFINITION
# ========================================

app.layout = dbc.Container([
    # Store for theme state
    dcc.Store(id='theme-store', data='light'),
    
    # Header Section with Theme Toggle
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H1("PruTech Taks C : Inter-Annotator Agreement Dashboard", 
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
                    html.P("Statistical analysis of annotation agreement across multiple annotators",
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
                        dbc.Button("Light", id="light-theme-btn", size="sm", outline=True),
                        dbc.Button("Dark", id="dark-theme-btn", size="sm", outline=True)
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
    
    # Navigation Tabs - Redesigned
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Tabs([
                    dbc.Tab(
                        label="Overall Agreement Analysis", 
                        tab_id="overall-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="Frequency-Based Analysis", 
                        tab_id="frequency-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="Hierarchical Analysis", 
                        tab_id="hierarchical-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    )
                ], 
                id="main-tabs", 
                active_tab="overall-tab",
                style={'borderBottom': '1px solid #dee2e6', 'marginBottom': '0'})
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ])
    ], style={'marginTop': '1rem'}),
    
    # Main Content Area - Full Width
    dbc.Row([
        dbc.Col([
            html.Div(
                id="tab-content",
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

# ========================================
# TAB CONTENT RENDERING
# ========================================

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Render content based on selected tab."""
    
    if active_tab == "overall-tab":
        return create_overall_tab_content()
    elif active_tab == "frequency-tab":
        return create_frequency_tab_content()
    elif active_tab == "hierarchical-tab":
        return create_hierarchical_tab_content()
    else:
        return html.Div("Tab content not found")

# ========================================
# CALLBACKS
# ========================================


# Callbacks
@app.callback(
    [Output("overall-results-container", "children"),
     Output("overall-calculation-progress", "style")],
    [Input("overall-calculate-btn", "n_clicks")],
    [State("overall-annotator-selector", "value"),
     State("overall-label-type-selector", "value"),
     State("overall-confidence-slider", "value")]
)
def update_overall_analysis(n_clicks, selected_annotators, label_type, confidence_level):
    """Update analysis based on user selections."""

    if n_clicks is None:
        return html.Div("Click 'Calculate Agreement Analysis' to begin"), {"visibility": "hidden"}

    print(f"[INFO] Starting analysis with {len(selected_annotators)} annotators")
    print(f"[INFO] Label type: {label_type}, Confidence: {confidence_level}")

    # Show progress bar
    progress_style = {"visibility": "visible"}

    try:
        # Filter data based on selections
        filtered_df = calculator.agreement_df[
            calculator.agreement_df['annotator'].isin(selected_annotators)
        ].copy()

        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")

        # Calculate alpha with confidence intervals
        alpha_result = calculator.calculate_alpha_with_ci(
            filtered_df, label_type, confidence_level, n_bootstrap=500
        )

        # Calculate pairwise agreement matrix
        agreement_matrix = calculator.calculate_pairwise_agreement_matrix(filtered_df, label_type)

        # Calculate document-level agreement
        doc_agreement = calculator.calculate_document_level_agreement(filtered_df, label_type)

        # Create visualizations
        heatmap_fig = visualizer.create_agreement_heatmap(agreement_matrix)

        # Create alpha comparison (comparing with other label types for context)
        alpha_comparison = {}
        for lt in ['full_label', 'L1_label', 'L2_label']:
            if lt == label_type:
                alpha_comparison[lt] = alpha_result
            else:
                # Quick calculation for comparison
                temp_result = calculator.calculate_alpha_with_ci(filtered_df, lt, confidence_level, n_bootstrap=100)
                alpha_comparison[lt] = temp_result

        alpha_fig = visualizer.create_alpha_comparison_chart(alpha_comparison)
        doc_fig = visualizer.create_document_agreement_histogram(doc_agreement, "Sample Level Agreement Analysis")

        # Create results layout
        results = html.Div([
            # Summary Statistics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Summary Statistics", style={'textAlign': 'center', 'fontSize': '1.5rem', 'fontWeight': '600'})),
                        dbc.CardBody([
                            html.P(f"Krippendorff's Alpha: {alpha_result['alpha']:.4f}", style={'fontSize': '1.2rem', 'fontWeight': '500', 'marginBottom': '1rem'}),
                            html.P(f"{confidence_level*100:.0f}% CI: [{alpha_result['ci_lower']:.4f}, {alpha_result['ci_upper']:.4f}]", style={'fontSize': '1.1rem', 'marginBottom': '1rem'}),
                            html.P(f"Annotators: {len(selected_annotators)}", style={'fontSize': '1.1rem', 'marginBottom': '1rem'}),
                            html.P(f"Samples: {len(filtered_df['id'].unique())}", style={'fontSize': '1.1rem', 'marginBottom': '1rem'}),
                            html.P(f"Bootstrap samples: {alpha_result['n_bootstrap_valid']}", style={'fontSize': '1.1rem', 'marginBottom': '0'})
                        ], style={'padding': '2rem'})
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Interpretation", style={'textAlign': 'center', 'fontSize': '1.5rem', 'fontWeight': '600'})),
                        dbc.CardBody([
                            html.P(get_alpha_interpretation(alpha_result['alpha']), style={'fontSize': '1.2rem', 'fontWeight': '500', 'marginBottom': '1rem'}),
                            html.P(f"Perfect agreement: {doc_agreement['perfect_agreement'].mean():.1%} of documents", style={'fontSize': '1.1rem', 'marginBottom': '1rem'}),
                            html.P(f"Average pairwise agreement: {agreement_matrix.values[np.triu_indices_from(agreement_matrix.values, k=1)].mean():.1f}%", style={'fontSize': '1.1rem', 'marginBottom': '0'})
                        ], style={'padding': '2rem'})
                    ])
                ], md=6)
            ], className="mb-4"),

            # Heatmap - Full width and centered
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(figure=heatmap_fig)
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'center',
                        'alignItems': 'center',
                        'width': '100%'
                    })
                ], md=12)
            ], className="mb-4"),

            # Side by side plots
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=doc_fig)
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=alpha_fig)
                ], md=6)
            ], className="mb-4")
            ])

        return results, {"visibility": "hidden"}

    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        error_message = html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])
        return error_message, {"visibility": "hidden"}


# Main analysis callback
@app.callback(
    [Output("freq-results-container", "children"),
     Output("freq-calculation-progress", "style")],
    [Input("freq-calculate-btn", "n_clicks")],
    [State("freq-annotator-selector", "value"),
     State("freq-label-type-selector", "value"),
     State("rare-threshold-slider", "value"),
     State("common-threshold-slider", "value")]
)
def update_frequency_analysis(n_clicks, selected_annotators, label_type, rare_threshold, common_threshold):
    """Update frequency analysis based on user selections."""

    if n_clicks is None:
        return html.Div("Click 'Calculate Frequency Analysis' to begin"), {"visibility": "hidden"}

    print(f"[INFO] Starting frequency analysis")
    print(f"[INFO] Annotators: {len(selected_annotators)}, Label type: {label_type}")
    print(f"[INFO] Thresholds - Rare: ≤{rare_threshold}, Common: ≥{common_threshold}")

    # Validate thresholds
    if common_threshold <= rare_threshold:
        error_msg = dbc.Alert("Common threshold must be greater than rare threshold", color="danger")
        return error_msg, {"visibility": "hidden"}

    try:
        # Filter data
        filtered_df = freq_calculator.agreement_df[
            freq_calculator.agreement_df['annotator'].isin(selected_annotators)
        ].copy()

        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")

        # Add debug output
        freq_calculator.debug_agreement_calculation(filtered_df, label_type)


        # Calculate label frequencies
        frequency_df = freq_calculator.calculate_label_frequencies(filtered_df, label_type)

        # Create frequency strata
        frequency_strata_df = freq_calculator.create_frequency_strata(
            frequency_df, rare_threshold, common_threshold
        )

        # Calculate stratified agreement
        stratified_results = freq_calculator.calculate_stratified_agreement(
            filtered_df, label_type, frequency_strata_df, rare_threshold, common_threshold
        )

        # Calculate frequency vs agreement correlation
        correlation_df, correlation_coef = freq_calculator.calculate_frequency_vs_agreement_correlation(
            filtered_df, label_type, frequency_df
        )

        # Create visualizations
        freq_dist_fig = freq_visualizer.create_frequency_distribution_plot(
            frequency_df, rare_threshold, common_threshold
        )
        stratified_comparison_fig = freq_visualizer.create_stratified_agreement_comparison(stratified_results)
        correlation_fig = freq_visualizer.create_frequency_vs_agreement_scatter(correlation_df, correlation_coef)

        # Create results summary table
        summary_data = []
        for stratum in ['rare', 'moderate', 'common']:
            result = stratified_results[stratum]
            alpha_display = f"{result['alpha']:.4f}" if not np.isnan(result['alpha']) and result['alpha'] != 0.0 else "N/A"
            summary_data.append({
                'Frequency Stratum': stratum.title(),
                'Alpha': alpha_display,
                'Labels': result['n_labels'],  # Number of unique label types
                'Annotations': result['n_annotations'],  # Total annotation instances
                'Samples': result['n_documents'],  # Unique documents annotated
                'Avg Frequency': f"{result['avg_frequency']:.1f}" if not np.isnan(result['avg_frequency']) else "N/A"
            })

        # Add explanation
        explanation_text = html.Div([
            html.P("Table Explanation:", style={'fontWeight': '600', 'marginTop': '1rem'}),
            html.Ul([
                html.Li("Labels: Number of unique label types in this frequency range"),
                html.Li("Annotations: Total number of annotation instances across all annotators"),
                html.Li("Samples: Number of unique samples that contain these labels"),
                html.Li("Alpha: Krippendorff's agreement coefficient (higher = better agreement)")
            ], style={'fontSize': '0.9rem', 'color': 'var(--text-secondary)'})
        ])

        summary_table = dash_table.DataTable(
            data=summary_data,
            columns=[{"name": i, "id": i} for i in summary_data[0].keys()],
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )

        # Create results layout
        results = html.Div([
            # Summary Section - Side by side layout
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Frequency Analysis Summary", style={'textAlign': 'center', 'fontWeight': '600'})),
                        dbc.CardBody([
                            summary_table,
                            explanation_text,
                            html.Hr(),
                            html.P(f"Frequency-Agreement Correlation: {correlation_coef:.3f}" if not np.isnan(correlation_coef) else "Correlation: N/A (No variance in agreement rates)", style={'fontSize': '1.1rem'}),
                            html.P(f"Threshold Settings: Rare ≤ {rare_threshold}, Common ≥ {common_threshold}", style={'fontSize': '1.1rem'})
                        ])
                    ])
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Frequency Stratification Guide", style={'textAlign': 'center', 'fontWeight': '600'})),
                        dbc.CardBody([
                            html.Div([
                                html.Div([
                                    html.Span("Rare Labels", style={'fontWeight': '600', 'color': '#e74c3c', 'fontSize': '1.1rem'}),
                                    html.Br(),
                                    html.Span(f"≤ {rare_threshold} occurrences", style={'color': 'var(--text-secondary)', 'fontSize': '1rem'})
                                ], style={'marginBottom': '1rem'}),
                                html.Div([
                                    html.Span("Moderate Labels", style={'fontWeight': '600', 'color': '#f39c12', 'fontSize': '1.1rem'}),
                                    html.Br(),
                                    html.Span(f"{rare_threshold + 1} - {common_threshold - 1} occurrences", style={'color': 'var(--text-secondary)', 'fontSize': '1rem'})
                                ], style={'marginBottom': '1rem'}),
                                html.Div([
                                    html.Span("Common Labels", style={'fontWeight': '600', 'color': '#27ae60', 'fontSize': '1.1rem'}),
                                    html.Br(),
                                    html.Span(f"≥ {common_threshold} occurrences", style={'color': 'var(--text-secondary)', 'fontSize': '1rem'})
                                ])
                            ])
                        ])
                    ])
                ], md=4)
            ], className="mb-4"),


            # Visualizations
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=freq_dist_fig)
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=stratified_comparison_fig)
                ], md=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=correlation_fig)
                ], md=12)
            ])
        ])

        return results, {"visibility": "hidden"}

    except Exception as e:
        print(f"[ERROR] Frequency analysis failed: {str(e)}")
        error_message = html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])
        return error_message, {"visibility": "hidden"}


@app.callback(
    [Output("hier-results-container", "children"),
     Output("hier-calculation-progress", "style")],
    [Input("hier-calculate-btn", "n_clicks")],
    [State("hier-annotator-selector", "value"),
     State("parent-category-selector", "value"),
     State("analysis-type-selector", "value")]
)
def update_hierarchical_analysis(n_clicks, selected_annotators, selected_parents, analysis_type):
    """Update hierarchical analysis based on user selections."""

    if n_clicks is None:
        return html.Div("Click 'Calculate Hierarchical Analysis' to begin"), {"visibility": "hidden"}

    print(f"[INFO] Starting hierarchical analysis")
    print(f"[INFO] Annotators: {len(selected_annotators)}, Parents: {len(selected_parents)}")
    print(f"[INFO] Analysis type: {analysis_type}")

    if not selected_parents:
        error_msg = dbc.Alert("Please select at least one parent category", color="warning")
        return error_msg, {"visibility": "hidden"}

    try:
        # Filter data
        filtered_df = hier_calculator.agreement_df[
            (hier_calculator.agreement_df['annotator'].isin(selected_annotators)) &
            (hier_calculator.agreement_df['L1_label'].isin(selected_parents))
        ].copy()

        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")

        results_components = []

        # Level comparison analysis
        if analysis_type in ['complete', 'levels']:
            level_results = hier_calculator.calculate_hierarchical_level_comparison(filtered_df)
            consistency_metrics = hier_calculator.calculate_hierarchical_consistency_metrics(level_results)

            level_comparison_fig = hier_visualizer.create_level_comparison_chart(level_results)
            consistency_display = hier_visualizer.create_consistency_metrics_display(consistency_metrics)

            results_components.extend([
                html.H4("Hierarchical Level Comparison"),
                consistency_display,
                html.Br(),
                dcc.Graph(figure=level_comparison_fig),
                html.Hr()
            ])

        # Conditional analysis
        if analysis_type in ['complete', 'conditional']:
            conditional_results = hier_calculator.calculate_conditional_agreement_by_parent(
                filtered_df, selected_parents
            )
            combination_df = hier_calculator.calculate_specific_parent_child_combinations(
                filtered_df, selected_parents
            )

            conditional_fig = hier_visualizer.create_conditional_agreement_chart(conditional_results)
            combination_heatmap = hier_visualizer.create_parent_child_combination_heatmap(combination_df)

            # Create summary table for conditional results
            conditional_summary = []
            for parent, result in conditional_results.items():
                conditional_summary.append({
                    'Parent Category': parent,
                    'L2 Alpha': f"{result['l2_alpha']:.4f}" if not np.isnan(result['l2_alpha']) else "N/A",
                    'Full Alpha': f"{result['full_alpha']:.4f}" if not np.isnan(result['full_alpha']) else "N/A",
                    'Child Labels': result['n_child_labels'],
                    'Annotations': result['n_annotations'],
                    'Documents': result['n_documents']
                })

            conditional_table = dash_table.DataTable(
                data=conditional_summary,
                columns=[{"name": i, "id": i} for i in conditional_summary[0].keys()],
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )

            results_components.extend([
                html.H4("Conditional Agreement by Parent Category"),
                conditional_table,
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=conditional_fig)
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(figure=combination_heatmap)
                    ], md=6)
                ])
            ])

        # Wrap results
        final_results = html.Div(results_components)

        return final_results, {"visibility": "hidden"}

    except Exception as e:
        print(f"[ERROR] Hierarchical analysis failed: {str(e)}")
        error_message = html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])
        return error_message, {"visibility": "hidden"}
    

@app.callback(
    [Output('theme-store', 'data'),
     Output('light-theme-btn', 'color'),
     Output('dark-theme-btn', 'color')],
    [Input('light-theme-btn', 'n_clicks'),
     Input('dark-theme-btn', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(light_clicks, dark_clicks, current_theme):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'light', 'primary', 'outline-secondary'
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'light-theme-btn':
        return 'light', 'primary', 'outline-secondary'
    elif trigger_id == 'dark-theme-btn':
        return 'dark', 'outline-secondary', 'primary'
    
    return current_theme, 'primary' if current_theme == 'light' else 'outline-secondary', 'primary' if current_theme == 'dark' else 'outline-secondary'

# Apply theme to body
app.clientside_callback(
    """
    function(theme) {
        if (theme === 'dark') {
            document.body.className = 'theme-dark';
        } else {
            document.body.className = 'theme-light';
        }
        return theme;
    }
    """,
    Output('theme-store', 'data', allow_duplicate=True),
    Input('theme-store', 'data'),
    prevent_initial_call=True
)


# ========================================
# END OF FILE
# ========================================

##### THE BELOW CODE IS ONLY FOR LOCAL TESTING ######
# if __name__ == '__main__':
#     app.run(debug=True, host='127.0.0.1', port=8061)
# ========================================
# IMPORTS
# ========================================

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


from DisagreementAnalysis.calculator import DisagreementAnalysisCalculator
from DisagreementAnalysis.visualizer import DisagreementAnalysisVisualizer
from DisagreementAnalysis.content import create_disagreement_tab


from ConfusionAnalysis.calculator import AnnotatorConfusionCalculator
from ConfusionAnalysis.visualizer import AnnotatorConfusionVisualizer
from ConfusionAnalysis.content import create_confusion_tab


from GoldSetAnalysis.calculator import GoldSetAnalysisCalculator
from GoldSetAnalysis.visualizer import GoldSetAnalysisVisualizer
from GoldSetAnalysis.content import create_goldset_tab

# ========================================
# DATA LOADING
# ========================================

agreement_df = pd.read_csv('data.csv')
print(f"[INFO] Loaded {len(agreement_df)} annotations from data.csv")

# ========================================
# INITIALIZE COMPONENTS
# ========================================

disagreement_calculator = DisagreementAnalysisCalculator(agreement_df)
disagreement_visualizer = DisagreementAnalysisVisualizer()

confusion_calculator = AnnotatorConfusionCalculator(agreement_df)
confusion_visualizer = AnnotatorConfusionVisualizer()

goldset_calculator = GoldSetAnalysisCalculator(agreement_df)
goldset_visualizer = GoldSetAnalysisVisualizer()

# ========================================
# CREATE DASH APP
# ======================================== 

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# CRITICAL FOR PLOTLY CLOUD DEPLOYMENT
server = app.server

# ========================================
# STYLING CONSTANTS
# ========================================

app.index_string =  open('page.html').read()

# ========================================
# APP LAYOUT
# ========================================

app.layout = dbc.Container([
    # Store for theme state
    dcc.Store(id='theme-store', data='light'),
    
    # Header Section with Theme Toggle
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H1("Data Quality Dashboard", 
                        id="main-title",
                        style={
                            'margin': 0, 
                            'fontWeight': '800', 
                            'letterSpacing': '-1px',
                            'fontSize': '4rem',
                            'textAlign': 'center',
                            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent',
                            'backgroundClip': 'text'
                        }),
                    html.P("Comprehensive analysis of annotation quality, disagreements, and gold-set recommendations",
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
                        label="Top Disagreement Items", 
                        tab_id="disagreement-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="Per-Annotator Confusion", 
                        tab_id="confusion-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="Suggested Gold-Set Refresh", 
                        tab_id="goldset-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    )
                ], 
                id="main-tabs", 
                active_tab="disagreement-tab",
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
    
    if active_tab == "disagreement-tab":
        return create_disagreement_tab()
    elif active_tab == "confusion-tab":
        return create_confusion_tab()
    elif active_tab == "goldset-tab":
        return create_goldset_tab()
    else:
        return html.Div("Tab content not found")
    

# ========================================
# CALLBACKS
# ========================================

@app.callback(
    Output('max-disagreement-slider', 'min'),
    Input('min-disagreement-slider', 'value')
)
def update_max_slider_min(min_value):
    """Ensure max disagreement is always >= min disagreement."""
    return min_value


@app.callback(
    [Output("disagreement-results-container", "children"),
     Output("disagreement-calculation-progress", "style"),
     Output("disagreement-calculate-btn", "disabled"),
     Output("disagreement-calculate-btn", "children")],
    [Input("disagreement-calculate-btn", "n_clicks")],
    [State("disagreement-annotator-selector", "value"),
     State("disagreement-label-type-selector", "value"),
     State("top-n-selector", "value"),
     State("min-disagreement-slider", "value"),
     State("max-disagreement-slider", "value")]
)
def update_disagreement_analysis(n_clicks, selected_annotators, label_type, top_n,
                                min_disagreement, max_disagreement):
    """Update disagreement analysis based on user selections."""

    if n_clicks is None:
        return (html.Div("Click 'Calculate Disagreement Analysis' to begin"),
                {"visibility": "hidden"}, False, "Calculate Disagreement Analysis")

    print(f"[INFO] Starting disagreement analysis")
    print(f"[INFO] Annotators: {len(selected_annotators)}, Label type: {label_type}")
    print(f"[INFO] Top N: {top_n}, Disagreement range: {min_disagreement:.3f} - {max_disagreement:.3f}")

    # Validate inputs
    if not selected_annotators:
        error_msg = dbc.Alert("Please select at least one annotator", color="warning")
        return error_msg, {"visibility": "hidden"}, False, "Calculate Disagreement Analysis"

    if max_disagreement < min_disagreement:
        error_msg = dbc.Alert("Maximum disagreement must be >= minimum disagreement", color="danger")
        return error_msg, {"visibility": "hidden"}, False, "Calculate Disagreement Analysis"

    try:
        # Show progress and disable button
        progress_style = {"visibility": "visible"}

        # Filter data
        filtered_df = disagreement_calculator.agreement_df[
            disagreement_calculator.agreement_df['annotator'].isin(selected_annotators)
        ].copy()

        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")

        # Calculate document disagreement scores
        disagreement_df = disagreement_calculator.calculate_document_disagreement_scores(
            filtered_df, label_type
        )

        # Filter by disagreement threshold
        threshold_filtered_df = disagreement_calculator.filter_by_disagreement_threshold(
            disagreement_df, min_disagreement, max_disagreement
        )

        # Calculate label confusion matrix
        confusion_matrix, confusion_counts = disagreement_calculator.calculate_label_confusion_matrix(
            filtered_df, label_type
        )

        # Analyze patterns
        patterns = disagreement_calculator.analyze_disagreement_patterns(disagreement_df)

        # Get top disagreement documents
        top_disagreement_data = disagreement_calculator.get_top_disagreement_documents(
            threshold_filtered_df, top_n
        )

        # Create visualizations
        distribution_fig = disagreement_visualizer.create_disagreement_distribution_histogram(disagreement_df)
        confusion_fig = disagreement_visualizer.create_confusion_matrix_heatmap(confusion_matrix)
        complexity_fig = disagreement_visualizer.create_text_complexity_vs_disagreement_scatter(disagreement_df)
        patterns_cards = disagreement_visualizer.create_patterns_summary_cards(patterns)

        # Create top disagreement table
        if top_disagreement_data:
            disagreement_table = dash_table.DataTable(
                data=top_disagreement_data,
                columns=[{"name": i, "id": i} for i in top_disagreement_data[0].keys()],
                style_cell={
                    'textAlign': 'left',
                    'fontSize': 12,
                    'font_family': 'Arial'
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Text Preview'}, 'width': '40%'},
                    {'if': {'column_id': 'Annotator Labels'}, 'width': '30%'},
                ],
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data={'whiteSpace': 'normal', 'height': 'auto'},
                page_size=10,
                sort_action='native',
                filter_action='native'
            )
        else:
            disagreement_table = html.P("No documents found matching the criteria.")

        # Create results layout
        results = html.Div([
            # Summary Section
            html.H4("Disagreement Analysis Summary"),
            patterns_cards,
            html.Hr(),

            # Visualizations
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=distribution_fig)
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=confusion_fig)
                ], md=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=complexity_fig)
                ], md=12)
            ], className="mb-4"),

            # Top disagreement documents
            html.H4(f"Top {min(top_n, len(top_disagreement_data))} Disagreement Samples"),
            html.P(f"Showing documents with disagreement scores between {min_disagreement:.3f} and {max_disagreement:.3f}"),
            disagreement_table
        ])

        return results, {"visibility": "hidden"}, False, "Calculate Disagreement Analysis"

    except Exception as e:
        print(f"[ERROR] Disagreement analysis failed: {str(e)}")
        error_message = html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])
        return error_message, {"visibility": "hidden"}, False, "Calculate Disagreement Analysis"


@app.callback(
    [Output("confusion-results-container", "children"),
     Output("confusion-calculation-progress", "style"),
     Output("confusion-calculate-btn", "disabled"),
     Output("confusion-calculate-btn", "children"),
     Output("matrix-annotator-selector", "options"),
     Output("matrix-annotator-selector", "value")],
    [Input("confusion-calculate-btn", "n_clicks")],
    [State("confusion-annotator-selector", "value"),
     State("confusion-label-type-selector", "value"),
     State("confusion-analysis-mode", "value")]
)
def update_confusion_analysis(n_clicks, selected_annotators, label_type, analysis_mode):
    """Update confusion analysis based on user selections."""

    if n_clicks is None:
        return (html.Div("Click 'Calculate Confusion Analysis' to begin"),
                {"visibility": "hidden"}, False, "Calculate Confusion Analysis",
                [{'label': 'Select after analysis', 'value': 'none'}], 'none')

    print(f"[INFO] Starting confusion analysis")
    print(f"[INFO] Annotators: {len(selected_annotators)}, Mode: {analysis_mode}")
    print(f"[INFO] Label type: {label_type}")

    if not selected_annotators:
        error_msg = dbc.Alert("Please select at least two annotators", color="warning")
        return (error_msg, {"visibility": "hidden"}, False, "Calculate Confusion Analysis",
                [{'label': 'Select after analysis', 'value': 'none'}], 'none')

    try:
        # Show progress and disable button
        progress_style = {"visibility": "visible"}

        # Filter data
        filtered_df = confusion_calculator.agreement_df[
            confusion_calculator.agreement_df['annotator'].isin(selected_annotators)
        ].copy()

        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")

        results_components = []
        matrix_options = [{'label': 'Select an annotator', 'value': 'none'}]
        matrix_value = 'none'

        # Individual performance analysis
        if analysis_mode in ['complete', 'individual']:
            performance_results = confusion_calculator.calculate_all_annotator_performance(
                filtered_df, label_type
            )

            if performance_results:
                # Update matrix selector options
                matrix_options = [{'label': f'{ann.replace("A_", "Annotator ")} (Accuracy: {results["accuracy"]:.1%})', 'value': ann} 
                                for ann, results in performance_results.items()]
                matrix_value = list(performance_results.keys())[0]  # Select first annotator by default

                # Performance ranking chart
                performance_fig = confusion_visualizer.create_annotator_performance_ranking(performance_results)
                results_components.extend([
                    html.H2("Individual Annotator Performance",style={'textAlign': 'center'}),
                    dcc.Graph(figure=performance_fig),
                ])

                # Add container for individual matrix (will be populated by separate callback)
                results_components.extend([
                    html.Hr(),
                    html.H2("Individual Confusion Matrix", id="matrix-title",style={'textAlign': 'center'}),
                    html.Div(
                        id="individual-matrix-container",
                        style={
                            'textAlign': 'center', 
                            'padding': '2rem 0 4rem 0',  # CHANGED: More bottom padding
                            'minHeight': '900px'         # ADD: Minimum height container
                        }
                    )
                ])

                # Store performance results for matrix callback
                results_components.append(
                    dcc.Store(id='performance-results-store', data={
                        ann: {
                            'accuracy': results['accuracy'],
                            'confusion_matrix': results['confusion_matrix'].to_dict()
                        } for ann, results in performance_results.items()
                    })
                )

                # Systematic bias analysis
                global_biases, annotator_biases = confusion_calculator.identify_systematic_biases(performance_results)
                bias_fig = confusion_visualizer.create_bias_pattern_visualization(global_biases)

                # Training recommendations
                recommendations = confusion_calculator.generate_training_recommendations(
                    performance_results, annotator_biases
                )
                recommendations = {key.replace('A_', 'Annotator '): value for key, value in recommendations.items()}

                recommendation_display = confusion_visualizer.create_training_recommendations_display(recommendations)

                results_components.extend([
                    html.Hr(),
                    html.H2("Systematic Bias Analysis",style={'textAlign': 'center'}),
                    dcc.Graph(figure=bias_fig),
                    html.Hr(),
                    html.H2("Training Recommendations",style={'textAlign': 'center'}),
                    recommendation_display
                ])

        # Pairwise analysis
        if analysis_mode in ['complete', 'pairwise']:
            pairwise_agreement = confusion_calculator.calculate_pairwise_annotator_agreement(
                filtered_df, label_type
            )
            pairwise_fig = confusion_visualizer.create_pairwise_agreement_heatmap(pairwise_agreement)

            results_components.extend([
                html.Hr() if analysis_mode == 'complete' else html.Div(),
                html.H2("Pairwise Annotator Agreement",style={'textAlign': 'center'}),
                html.Div([
                    dcc.Graph(figure=pairwise_fig)
                ], style={
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'width': '100%'
                })
            ])

        # Wrap results
        final_results = html.Div(results_components)

        return (final_results, {"visibility": "hidden"}, False, "Calculate Confusion Analysis",
                matrix_options, matrix_value)

    except Exception as e:
        print(f"[ERROR] Confusion analysis failed: {str(e)}")
        error_message = html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])
        return (error_message, {"visibility": "hidden"}, False, "Calculate Confusion Analysis",
                [{'label': 'Select after analysis', 'value': 'none'}], 'none')


@app.callback(
    Output("individual-matrix-container", "children"),
    [Input("matrix-annotator-selector", "value")],
    [State("performance-results-store", "data")]
)
def update_individual_matrix(selected_annotator, performance_data):
    """Update the individual confusion matrix display."""
    
    if not selected_annotator or selected_annotator == 'none' or not performance_data:
        return html.Div([
            html.P("Select an annotator from the dropdown above to view their confusion matrix.",
                  style={'textAlign': 'center', 'color': 'gray', 'fontSize': '1.1rem', 'padding': '2rem'})
        ])

    try:
        # Get the performance data for selected annotator
        annotator_data = performance_data[selected_annotator]
        accuracy = annotator_data['accuracy']
        
        # Reconstruct confusion matrix from stored data
        confusion_dict = annotator_data['confusion_matrix']
        confusion_df = pd.DataFrame(confusion_dict)
        
        # Create large matrix visualization
        matrix_fig = AnnotatorConfusionVisualizer.create_large_individual_confusion_matrix(
            confusion_df, selected_annotator, accuracy
        )
        
        return html.Div([
            html.Div([
                dcc.Graph(
                    figure=matrix_fig,
                    style={'width': '100%', 'height': '100%'}
                )
            ], style={
                'width': '1100px',  # Fixed width to match your matrix
                'margin': '0 auto'  # Center the container
            })
        ], style={
            'display': 'flex', 
            'justifyContent': 'center',
            'alignItems': 'center',
            'marginBottom': '3rem',
            'minHeight': '850px',
            'width': '100%'
        })

    except Exception as e:
        print(f"[ERROR] Matrix display failed: {str(e)}")
        return html.Div([
            dbc.Alert(f"Error displaying matrix: {str(e)}", color="danger")
        ])


@app.callback(
    [Output("goldset-results-container", "children"),
     Output("goldset-calculation-progress", "style"),
     Output("goldset-calculate-btn", "disabled"),
     Output("goldset-calculate-btn", "children")],
    [Input("goldset-calculate-btn", "n_clicks")],
    [State("goldset-annotator-selector", "value"),
     State("goldset-label-type-selector", "value"),
     State("goldset-strategy-selector", "value"),
     State("samples-per-label-slider", "value"),
     State("high-agreement-threshold-slider", "value"),
     State("disagreement-range-slider", "value")]
)
def update_goldset_analysis(n_clicks, selected_annotators, label_type, strategy,
                           samples_per_label, high_agreement_threshold, disagreement_range):
    """Update gold-set analysis based on user selections."""

    if n_clicks is None:
        return (html.Div("Click 'Generate Gold-Set Recommendations' to begin"),
                {"visibility": "hidden"}, False, "Generate Gold-Set Recommendations")

    print(f"[INFO] Starting gold-set analysis")
    print(f"[INFO] Strategy: {strategy}, Samples per label: {samples_per_label}")
    print(f"[INFO] High agreement threshold: {high_agreement_threshold}")
    print(f"[INFO] Disagreement range: {disagreement_range}")

    if not selected_annotators:
        error_msg = dbc.Alert("Please select at least two annotators", color="warning")
        return error_msg, {"visibility": "hidden"}, False, "Generate Gold-Set Recommendations"

    try:
        # Filter data
        filtered_df = goldset_calculator.agreement_df[
            goldset_calculator.agreement_df['annotator'].isin(selected_annotators)
        ].copy()

        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")

        # Calculate document agreement levels
        agreement_df = goldset_calculator.calculate_document_agreement_levels(
            filtered_df, label_type
        )

        # Analyze label coverage
        coverage_df = goldset_calculator.analyze_label_coverage(filtered_df, label_type)

        # Select candidates based on strategy
        high_confidence_candidates = pd.DataFrame()
        disagreement_candidates = pd.DataFrame()

        if strategy in ['high_only', 'mixed']:
            high_confidence_candidates = goldset_calculator.select_high_confidence_candidates(
                agreement_df, coverage_df, high_agreement_threshold, samples_per_label
            )

        if strategy in ['mixed', 'disagreement_focus']:
            disagreement_samples = samples_per_label if strategy == 'disagreement_focus' else max(2, samples_per_label // 2)
            disagreement_candidates = goldset_calculator.select_disagreement_candidates(
                agreement_df, coverage_df, disagreement_range[0], disagreement_range[1], disagreement_samples
            )

        # Identify coverage gaps
        coverage_gaps = goldset_calculator.identify_coverage_gaps(
            coverage_df, high_confidence_candidates, disagreement_candidates
        )

        # Calculate quality metrics
        quality_metrics = goldset_calculator.calculate_gold_set_quality_metrics(
            high_confidence_candidates, disagreement_candidates, coverage_df
        )

        # Create visualizations
        coverage_fig = goldset_visualizer.create_coverage_comparison_chart(coverage_df, quality_metrics)
        candidate_fig = goldset_visualizer.create_candidate_distribution_chart(
            high_confidence_candidates, disagreement_candidates
        )
        agreement_fig = goldset_visualizer.create_agreement_distribution_scatter(
            agreement_df, high_agreement_threshold, disagreement_range[1]
        )
        quality_dashboard = goldset_visualizer.create_quality_metrics_dashboard(quality_metrics)

        # Create candidate tables
        all_candidates = pd.concat([high_confidence_candidates, disagreement_candidates], ignore_index=True)

        if len(all_candidates) > 0:
            # Prepare table data
            table_data = []
            for _, candidate in all_candidates.iterrows():
                table_data.append({
                    'Sample ID': candidate['document_id'],
                    'Label': candidate['label'],
                    'Agreement Rate': f"{candidate['agreement_rate']:.1%}",
                    'Selection Reason': candidate['selection_reason'].replace('_', ' ').title(),
                    'Priority': candidate['priority'].title(),
                    'Text Preview': candidate['text_preview'],
                    'Text Length': candidate['text_length'],
                    'Annotators': candidate['n_annotators']
                })

            candidates_table = dash_table.DataTable(
                data=table_data,
                columns=[{"name": i, "id": i} for i in table_data[0].keys()],
                style_cell={
                    'textAlign': 'left',
                    'fontSize': 12,
                    'font_family': 'Arial'
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Text Preview'}, 'width': '40%'},
                ],
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data={'whiteSpace': 'normal', 'height': 'auto'},
                page_size=10,
                sort_action='native',
                filter_action='native'
            )
        else:
            candidates_table = html.P("No candidates selected with current criteria.")

        # Coverage gaps table
        if len(coverage_gaps) > 0:
            gaps_table_data = coverage_gaps.to_dict('records')
            gaps_table = dash_table.DataTable(
                data=gaps_table_data,
                columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in coverage_gaps.columns],
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=5
            )
        else:
            gaps_table = html.P("No significant coverage gaps identified.")

        # Create results layout
        results = html.Div([
            # Quality Dashboard
            html.H2("Gold-Set Quality Overview", style={'textAlign': 'center'}),
            quality_dashboard,
            html.Hr(),

            # Visualizations
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=coverage_fig)
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=candidate_fig)
                ], md=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=agreement_fig)
                ], md=12)
            ], className="mb-4"),

            # Recommended candidates
            html.H2(f"Recommended Gold-Set Candidates ({len(all_candidates)} samples)", style={'textAlign': 'center'}),
            html.P(f"Strategy: {strategy.replace('_', ' ').title()}, High Agreement ≥ {high_agreement_threshold:.0%}, Useful Disagreement: {disagreement_range[0]:.0%}-{disagreement_range[1]:.0%}"),
            candidates_table,

            html.Hr(),

            # Coverage gaps
            html.H2("Coverage Gap Analysis", style={'textAlign': 'center'}),
            html.P("Labels requiring additional attention for comprehensive gold-set coverage:"),
            gaps_table
        ])

        return results, {"visibility": "hidden"}, False, "Generate Gold-Set Recommendations"

    except Exception as e:
        print(f"[ERROR] Gold-set analysis failed: {str(e)}")
        error_message = html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])
        return error_message, {"visibility": "hidden"}, False, "Generate Gold-Set Recommendations"
    
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

# ###### THE BELOW CODE IS ONLY FOR LOCAL TESTING ######
# if __name__ == '__main__':
#     app.run(debug=True, host='127.0.0.1', port=8062)
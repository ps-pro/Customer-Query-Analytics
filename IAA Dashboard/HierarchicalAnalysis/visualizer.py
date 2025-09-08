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



# Hierarchical Analysis Visualizer
class HierarchicalAnalysisVisualizer:
    """Visualizer for hierarchical analysis."""

    @staticmethod
    def create_level_comparison_chart(level_comparison_results):
        """Create side-by-side comparison of hierarchical levels."""
        print(f"[DEBUG] Creating hierarchical level comparison chart")

        levels = list(level_comparison_results.keys())
        alphas = [level_comparison_results[level]['alpha'] for level in levels]
        n_labels = [level_comparison_results[level]['n_unique_labels'] for level in levels]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Agreement by Hierarchical Level", "Label Complexity by Level"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Alpha comparison
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        fig.add_trace(
            go.Bar(x=levels, y=alphas, name="Krippendorff's Alpha",
                  text=[f'{a:.3f}' if not np.isnan(a) else 'N/A' for a in alphas],
                  textposition='outside', marker_color=colors),
            row=1, col=1
        )

        # Label complexity
        fig.add_trace(
            go.Bar(x=levels, y=n_labels, name="Number of Unique Labels",
                  text=n_labels, textposition='outside', marker_color=colors),
            row=1, col=2
        )

        # Add interpretation lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=1, col=1,
                     annotation_text="Excellent")
        fig.add_hline(y=0.67, line_dash="dash", line_color="orange", row=1, col=1,
                     annotation_text="Good")
        fig.add_hline(y=0.4, line_dash="dash", line_color="red", row=1, col=1,
                     annotation_text="Fair")

        fig.update_layout(
            title={
                'text': "Hierarchical Level Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            showlegend=False
        )

        fig.update_yaxes(title_text="Krippendorff's Alpha", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Number of Labels", row=1, col=2)

        return fig

    @staticmethod
    def create_conditional_agreement_chart(conditional_results):
        """Create chart showing agreement within each parent category."""
        print(f"[DEBUG] Creating conditional agreement chart")

        parents = list(conditional_results.keys())
        l2_alphas = [conditional_results[p]['l2_alpha'] for p in parents]
        full_alphas = [conditional_results[p]['full_alpha'] for p in parents]
        n_annotations = [conditional_results[p]['n_annotations'] for p in parents]

        fig = go.Figure()

        # L2 alphas within parent
        fig.add_trace(go.Bar(
            x=parents, y=l2_alphas, name="L2 Agreement (within parent)",
            text=[f'{a:.3f}' if not np.isnan(a) else 'N/A' for a in l2_alphas],
            textposition='outside', opacity=0.7
        ))

        # Full alphas within parent
        fig.add_trace(go.Bar(
            x=parents, y=full_alphas, name="Full Agreement (within parent)",
            text=[f'{a:.3f}' if not np.isnan(a) else 'N/A' for a in full_alphas],
            textposition='outside', opacity=0.7
        ))

        # Add sample sizes as text annotations
        for i, parent in enumerate(parents):
            fig.add_annotation(
                x=parent, y=-0.05,
                text=f"n={n_annotations[i]}",
                showarrow=False, font=dict(size=10)
            )

        fig.update_layout(
            title={
                'text': "Conditional Agreement Analysis by Parent Category",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Parent Category",
            yaxis_title="Krippendorff's Alpha",
            yaxis=dict(range=[0, 1]),
            barmode='group'
        )

        return fig

    @staticmethod
    def create_parent_child_combination_heatmap(combination_df):
        """Create heatmap of agreement rates for parent-child combinations."""
        print(f"[DEBUG] Creating parent-child combination heatmap")

        if len(combination_df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data available for parent-child combinations",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        # Create pivot table for heatmap
        heatmap_data = combination_df.pivot(index='parent', columns='child', values='agreement_rate')
        heatmap_data = heatmap_data.fillna(0)  # Fill missing combinations with 0

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlBu',
            zmin=0,
            zmax=1,
            text=np.round(heatmap_data.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Agreement Rate")
        ))

        fig.update_layout(
            title={
                'text': "Agreement Rates by Parent-Child Combination",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Child Category",
            yaxis_title="Parent Category",
            height=max(400, len(heatmap_data.index) * 40)
        )

        return fig

    @staticmethod
    def create_consistency_metrics_display(consistency_metrics):
        """Create display for hierarchical consistency metrics."""
        print(f"[DEBUG] Creating consistency metrics display")

        metrics_cards = [
            dbc.Card([
                dbc.CardBody([
                    html.H5("Best Agreement Level", className="card-title"),
                    html.H3(consistency_metrics['best_agreement_level'], className="text-success"),
                    html.P("Highest alpha score")
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Hierarchy Impact", className="card-title"),
                    html.H3(consistency_metrics['hierarchy_impact'],
                           className="text-primary" if consistency_metrics['hierarchy_impact'] == 'Positive' else "text-warning"),
                    html.P("Effect of hierarchical structure")
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Alpha Range", className="card-title"),
                    html.H3(f"{consistency_metrics['alpha_range']:.3f}" if not np.isnan(consistency_metrics['alpha_range']) else "N/A",
                           className="text-info"),
                    html.P("Spread across levels")
                ])
            ])
        ]

        return dbc.Row([dbc.Col(card, md=4) for card in metrics_cards])


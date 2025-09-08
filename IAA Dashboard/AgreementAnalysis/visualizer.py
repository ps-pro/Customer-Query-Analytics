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


class IAAAgreementVisualizer:
    """Class for creating IAA visualizations."""

    @staticmethod
    def create_agreement_heatmap(agreement_matrix, title="Annotator Agreement Matrix"):
        """Create interactive heatmap of annotator agreement."""
        print(f"[DEBUG] Creating heatmap: {title}")

        # Convert annotator names for display
        x_labels = [f'Annotator {col.split("_")[1]}' if col.startswith('A_') else col for col in agreement_matrix.columns]
        y_labels = [f'Annotator {idx.split("_")[1]}' if idx.startswith('A_') else idx for idx in agreement_matrix.index]

        fig = go.Figure(data=go.Heatmap(
            z=agreement_matrix.values,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
            zmin=0,
            zmax=100,
            text=agreement_matrix.round(1),
            texttemplate='%{text}%',
            textfont={"size": 14},
            colorbar=dict(title="Agreement %")
        ))

        fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
            'font_weight': 'bold'
        },
        xaxis_title="Annotator",
        yaxis_title="Annotator",
        width=800,
        height=800
    )
        return fig

    @staticmethod
    def create_alpha_comparison_chart(alpha_results, title="Krippendorff's Alpha Comparison"):
        """Create bar chart showing alpha values with confidence intervals."""
        print(f"[DEBUG] Creating alpha comparison chart: {title}")

        labels = list(alpha_results.keys())
        alphas = [alpha_results[label]['alpha'] for label in labels]
        ci_lower = [alpha_results[label]['ci_lower'] for label in labels]
        ci_upper = [alpha_results[label]['ci_upper'] for label in labels]

        fig = go.Figure()

        # Add bars
        fig.add_trace(go.Bar(
            x=labels,
            y=alphas,
            name="Krippendorff's Alpha",
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ci_upper[i] - alphas[i] for i in range(len(alphas))],
                arrayminus=[alphas[i] - ci_lower[i] for i in range(len(alphas))],
            ),
            text=[f'{alpha:.3f}' for alpha in alphas],
            textposition='outside'
        ))

        # Add interpretation lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                     annotation_text="Excellent (α ≥ 0.8)")
        fig.add_hline(y=0.67, line_dash="dash", line_color="orange",
                     annotation_text="Good (α ≥ 0.67)")
        fig.add_hline(y=0.4, line_dash="dash", line_color="red",
                     annotation_text="Fair (α ≥ 0.4)")

        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Label Type",
            yaxis_title="Krippendorff's Alpha",
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )

        return fig

    @staticmethod
    def create_document_agreement_histogram(doc_agreement_df, title="Sample-Level Agreement Distribution"):
        """Create histogram of Sample-level agreement rates."""
        print(f"[DEBUG] Creating document agreement histogram: {title}")

        perfect_agreement_rate = doc_agreement_df['perfect_agreement'].mean() * 100

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=doc_agreement_df['unique_labels'],
            nbinsx=max(doc_agreement_df['unique_labels']) + 1,
            name="Sample Count",
            textposition='outside'
        ))

        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Number of Unique Labels per Sample",
            yaxis_title="Number of Samples",
            bargap=0.1
        )

        return fig


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


# Frequency Analysis Visualizer
class FrequencyAnalysisVisualizer:
    """Visualizer for frequency-based analysis."""

    @staticmethod
    def create_frequency_distribution_plot(frequency_df, rare_threshold, common_threshold):
        """Create frequency distribution histogram with stratum boundaries."""
        print(f"[DEBUG] Creating frequency distribution plot")

        fig = go.Figure()

        # Create histogram
        fig.add_trace(go.Histogram(
            x=frequency_df['frequency'],
            nbinsx=min(30, len(frequency_df)),
            name="Label Frequency Distribution",
            opacity=0.7
        ))

        # Add threshold lines
        fig.add_vline(x=rare_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Rare threshold: {rare_threshold}")
        fig.add_vline(x=common_threshold, line_dash="dash", line_color="green",
                     annotation_text=f"Common threshold: {common_threshold}")

        fig.update_layout(
            title={
                'text': "Label Frequency Distribution with Stratum Boundaries",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Label Frequency",
            yaxis_title="Number of Labels",
            showlegend=False
        )

        return fig

    @staticmethod
    def create_stratified_agreement_comparison(stratified_results):
        """Create bar chart comparing agreement across frequency strata."""
        print(f"[DEBUG] Creating stratified agreement comparison")

        strata = ['rare', 'moderate', 'common']
        alphas = [stratified_results[s]['alpha'] for s in strata]
        n_labels = [stratified_results[s]['n_labels'] for s in strata]
        n_annotations = [stratified_results[s]['n_annotations'] for s in strata]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Krippendorff's Alpha by Frequency Stratum", "Sample Sizes by Stratum"],
            specs=[[{"secondary_y": False}, {"secondary_y": True}]]
        )

        # Alpha comparison
        fig.add_trace(
            go.Bar(x=strata, y=alphas, name="Alpha",
                  text=[f'{a:.3f}' if not np.isnan(a) else 'N/A' for a in alphas],
                  textposition='outside'),
            row=1, col=1
        )

        # Sample sizes
        fig.add_trace(
            go.Bar(x=strata, y=n_labels, name="# Labels", opacity=0.7),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=strata, y=n_annotations, mode='lines+markers',
                      name="# Annotations", yaxis="y2"),
            row=1, col=2
        )

        fig.update_layout(
            title={
                'text': "Agreement Analysis Across Frequency Strata",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            showlegend=True
        )

        fig.update_yaxes(title_text="Krippendorff's Alpha", row=1, col=1)
        fig.update_yaxes(title_text="Number of Labels", row=1, col=2)
        fig.update_yaxes(title_text="Number of Annotations", secondary_y=True, row=1, col=2)

        return fig

    @staticmethod
    def create_frequency_vs_agreement_scatter(correlation_df, correlation_coef):
        """Create scatter plot of frequency vs agreement rate."""
        print(f"[DEBUG] Creating frequency vs agreement scatter plot")

        if len(correlation_df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for correlation analysis",
                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                            font=dict(size=16))
            fig.update_layout(title="Label Frequency vs Agreement Rate - No Data Available")
            return fig

        # Calculate proper marker sizes based on n_documents
        min_docs = correlation_df['n_documents'].min()
        max_docs = correlation_df['n_documents'].max()
        
        # Scale marker sizes between 8 and 25
        if max_docs > min_docs:
            marker_sizes = 12 + (correlation_df['n_documents'] - min_docs) / (max_docs - min_docs) * 18
        else:
            marker_sizes = [16] * len(correlation_df)  # All same size if no variation

        fig = go.Figure()

        # Create scatter plot with properly sized markers
        fig.add_trace(go.Scatter(
            x=correlation_df['frequency'],
            y=correlation_df['agreement_rate'],
            mode='markers',
            text=[f"<b>{label}</b><br>Documents: {docs}<br>Alpha: {alpha:.3f}" 
                for label, docs, alpha in zip(correlation_df['label'], 
                                            correlation_df['n_documents'],
                                            correlation_df['agreement_rate'])],
            hovertemplate='%{text}<br>Frequency: %{x}<br>Agreement Rate: %{y:.3f}<extra></extra>',
            marker=dict(
                size=marker_sizes,
                color=correlation_df['agreement_rate'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(
                    title="Agreement Rate (α)",
                    x=1.02,  # Move to the right
                    xanchor="left",
                    len=0.8,  # Make it shorter
                    thickness=15  # Make it thinner
                ),
                line=dict(width=2, color='black'),
                sizemode='diameter'
            )
        ))

        # Add trend line if correlation exists
        if not np.isnan(correlation_coef) and len(correlation_df) > 2:
            z = np.polyfit(correlation_df['frequency'], correlation_df['agreement_rate'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(correlation_df['frequency'].min(), correlation_df['frequency'].max(), 100)
            y_trend = p(x_trend)

            fig.add_trace(go.Scatter(
                x=x_trend, y=y_trend,
                mode='lines',
                name=f'Trend (r={correlation_coef:.3f})',
                line=dict(dash='dash', color='red', width=2)
            ))

        fig.update_layout(
            title={
                'text': "Label Frequency vs Agreement Rate (Krippendorff's α)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Label Frequency (Count)",
            yaxis_title="Agreement Rate (Krippendorff's Alpha)",
            showlegend=True,
            height=500,
            margin=dict(r=120)  # Add right margin for colorbar
        )

        return fig


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


# Disagreement Analysis Visualizer
class DisagreementAnalysisVisualizer:
    """Visualizer for disagreement analysis."""

    @staticmethod
    def create_disagreement_distribution_histogram(disagreement_df):
        """Create histogram showing distribution of disagreement scores."""
        print(f"[DEBUG] Creating disagreement distribution histogram")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=disagreement_df['disagreement_score'],
            nbinsx=30,
            name="Sample Count",
            opacity=0.7,
            marker_color='lightblue'
        ))

        # Add vertical lines for key thresholds
        fig.add_vline(x=0.0, line_dash="solid", line_color="green",
                     annotation_text="Perfect Agreement")
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                     annotation_text="High Disagreement")
        fig.add_vline(x=disagreement_df['disagreement_score'].mean(),
                     line_dash="dot", line_color="red",
                     annotation_text=f"Average: {disagreement_df['disagreement_score'].mean():.3f}")

        fig.update_layout(
            title={
                'text': "Distribution of Disagreement Scores Across All Samples",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Disagreement Score (0 = Perfect Agreement, 1 = Maximum Disagreement)",
            yaxis_title="Number of Samples",
            showlegend=False
        )

        return fig

    @staticmethod
    def create_confusion_matrix_heatmap(confusion_matrix, title="Label Confusion Matrix"):
        """Create heatmap showing which labels are most commonly confused."""
        print(f"[DEBUG] Creating confusion matrix heatmap")

        # Remove self-confusion (diagonal) for clarity
        confusion_display = confusion_matrix.copy()
        np.fill_diagonal(confusion_display.values, 0)

        fig = go.Figure(data=go.Heatmap(
            z=confusion_display.values,
            x=confusion_display.columns,
            y=confusion_display.index,
            colorscale='Reds',
            text=confusion_display.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Confusion Count")
        ))

        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Label (Confused With)",
            yaxis_title="Label (Original)",
            height=max(400, len(confusion_matrix) * 30)
        )

        return fig

    @staticmethod
    def create_text_complexity_vs_disagreement_scatter(disagreement_df):
        """Create scatter plot of text characteristics vs disagreement."""
        print(f"[DEBUG] Creating text complexity vs disagreement scatter plot")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Text Length vs Disagreement", "Word Count vs Disagreement"]
        )

        # Text length scatter
        fig.add_trace(
            go.Scatter(
                x=disagreement_df['text_length'],
                y=disagreement_df['disagreement_score'],
                mode='markers',
                text=disagreement_df['document_id'],
                hovertemplate='<b>Doc: %{text}</b><br>Length: %{x}<br>Disagreement: %{y:.3f}<extra></extra>',
                marker=dict(size=6, opacity=0.6, color='blue'),
                name="Text Length"
            ),
            row=1, col=1
        )

        # Word count scatter
        fig.add_trace(
            go.Scatter(
                x=disagreement_df['word_count'],
                y=disagreement_df['disagreement_score'],
                mode='markers',
                text=disagreement_df['document_id'],
                hovertemplate='<b>Doc: %{text}</b><br>Words: %{x}<br>Disagreement: %{y:.3f}<extra></extra>',
                marker=dict(size=6, opacity=0.6, color='red'),
                name="Word Count"
            ),
            row=1, col=2
        )

        # Add trend lines if correlation exists
        if len(disagreement_df) > 1:
            # Text length trend
            text_corr = disagreement_df['text_length'].corr(disagreement_df['disagreement_score'])
            if abs(text_corr) > 0.1:
                z1 = np.polyfit(disagreement_df['text_length'], disagreement_df['disagreement_score'], 1)
                p1 = np.poly1d(z1)
                x_trend1 = np.linspace(disagreement_df['text_length'].min(), disagreement_df['text_length'].max(), 100)
                fig.add_trace(
                    go.Scatter(x=x_trend1, y=p1(x_trend1), mode='lines', name=f'Trend (r={text_corr:.3f})',
                              line=dict(dash='dash', color='blue')), row=1, col=1
                )

            # Word count trend
            word_corr = disagreement_df['word_count'].corr(disagreement_df['disagreement_score'])
            if abs(word_corr) > 0.1:
                z2 = np.polyfit(disagreement_df['word_count'], disagreement_df['disagreement_score'], 1)
                p2 = np.poly1d(z2)
                x_trend2 = np.linspace(disagreement_df['word_count'].min(), disagreement_df['word_count'].max(), 100)
                fig.add_trace(
                    go.Scatter(x=x_trend2, y=p2(x_trend2), mode='lines', name=f'Trend (r={word_corr:.3f})',
                              line=dict(dash='dash', color='red')), row=1, col=2
                )

        fig.update_layout(
            title={
                'text': "Text Characteristics vs Disagreement Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            showlegend=False
        )

        fig.update_yaxes(title_text="Disagreement Score", row=1, col=1)
        fig.update_yaxes(title_text="Disagreement Score", row=1, col=2)
        fig.update_xaxes(title_text="Text Length (characters)", row=1, col=1)
        fig.update_xaxes(title_text="Word Count", row=1, col=2)

        return fig

    @staticmethod
    def create_patterns_summary_cards(patterns):
        """Create summary cards showing disagreement patterns."""
        print(f"[DEBUG] Creating patterns summary cards")

        cards = [
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Samples", className="card-title"),
                    html.H3(f"{patterns['total_documents']:,}", className="text-primary"),
                    html.P(f"Perfect Agreement: {patterns['perfect_agreement_docs']:,}")
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.H5("High Disagreement", className="card-title"),
                    html.H3(f"{patterns['high_disagreement_docs']:,}", className="text-warning"),
                    html.P(f"Score > 0.5: {patterns['high_disagreement_docs']/patterns['total_documents']:.1%}")
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Average Disagreement", className="card-title"),
                    html.H3(f"{patterns['avg_disagreement_score']:.3f}", className="text-info"),
                    html.P(f"Labels per doc: {patterns['avg_unique_labels_per_doc']:.1f}")
                ])
            ])
        ]

        return dbc.Row([dbc.Col(card, md=4) for card in cards])


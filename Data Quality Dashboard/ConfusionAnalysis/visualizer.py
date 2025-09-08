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


# Updated visualizer method for larger matrix
class AnnotatorConfusionVisualizer:
    """Visualizer for annotator confusion analysis."""

    @staticmethod
    def create_large_individual_confusion_matrix(confusion_df, annotator_name, accuracy):
        """Create large, centered confusion matrix heatmap for individual annotator."""
        print(f"[DEBUG] Creating large confusion matrix for {annotator_name}")

        fig = go.Figure(data=go.Heatmap(
            z=confusion_df.values,
            x=confusion_df.columns,
            y=confusion_df.index,
            colorscale='Blues',
            text=confusion_df.values,
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(
                title=dict(text="Count", font=dict(size=16, color="#000000")), 
                tickfont=dict(size=12, color='#000000')  
            )
        ))

        fig.update_layout(
            title={
                'text': f"{annotator_name.replace('A_', 'Annotator ')} vs Majority Vote<br>Accuracy: {accuracy:.1%}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Annotator Label",
            yaxis_title="Majority Vote (True)",
            xaxis=dict(
                title=dict(text="Annotator Label", font=dict(size=16, color='#333333')),
                tickfont=dict(size=12, color='#333333'),
                tickangle=45
            ),
            yaxis=dict(
                title=dict(text="Majority Vote (True)", font=dict(size=16, color='#333333')),
                tickfont=dict(size=12, color='#333333')
            ),
            height=850,
            width=1100,
            margin=dict(l=120, r=120, t=120, b=180),
            paper_bgcolor='white',     # CHANGED: Add white background
            plot_bgcolor='white'       # CHANGED: Add white background
        )
        return fig

    # Keep all other existing methods the same...
    @staticmethod
    def create_annotator_performance_ranking(performance_results):
        """Create bar chart ranking annotator performance."""
        print(f"[DEBUG] Creating annotator performance ranking")

        annotators = list(performance_results.keys())
        annotator_display_names = [ann.replace('A_', 'Annotator ') for ann in annotators]  
        accuracies = [performance_results[ann]['accuracy'] for ann in annotators]
        error_counts = [performance_results[ann]['total_errors'] for ann in annotators]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Accuracy vs Majority Vote", "Total Error Count"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Accuracy chart
        colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in accuracies]
        fig.add_trace(
            go.Bar(x=annotator_display_names, y=accuracies, name="Accuracy",
                  text=[f'{acc:.1%}' for acc in accuracies],
                  textposition='outside', marker_color=colors),
            row=1, col=1
        )

        # Error count chart
        fig.add_trace(
            go.Bar(x=annotator_display_names, y=error_counts, name="Error Count",
                  text=error_counts, textposition='outside', marker_color='lightcoral'),
            row=1, col=2
        )

        # Add benchmark lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=1, col=1,
                     annotation_text="Good Performance")
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", row=1, col=1,
                     annotation_text="Needs Improvement")

        fig.update_layout(
            title={
                'text': "Annotator Performance Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            showlegend=False
        )

        fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Error Count", row=1, col=2)

        return fig

    @staticmethod
    def create_pairwise_agreement_heatmap(pairwise_agreement):
        """Create heatmap showing pairwise agreement between annotators."""
        print(f"[DEBUG] Creating pairwise agreement heatmap")

        # Add this at the start of the method
        display_names = [name.replace('A_', 'Annotator ') for name in pairwise_agreement.index]
        pairwise_agreement_display = pairwise_agreement.copy()
        pairwise_agreement_display.index = display_names
        pairwise_agreement_display.columns = display_names

        fig = go.Figure(data=go.Heatmap(
            z=pairwise_agreement_display.values,
            x=pairwise_agreement_display.columns,
            y=pairwise_agreement_display.index,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            text=np.round(pairwise_agreement.values, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Agreement Rate")
        ))

        fig.update_layout(
            title={
                'text': "Pairwise Annotator Agreement Matrix",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
            xaxis_title="Annotator",
            yaxis_title="Annotator", 
            height=700,                                      
            width=800,                                       
            margin=dict(l=100, r=100, t=80, b=100),        
            paper_bgcolor='white',                           
            plot_bgcolor='white'                             
        )

        return fig

    @staticmethod
    def create_bias_pattern_visualization(global_biases, title="Top Systematic Bias Patterns"):
        """Create visualization of systematic bias patterns."""
        print(f"[DEBUG] Creating bias pattern visualization")

        if not global_biases:
            fig = go.Figure()
            fig.add_annotation(text="No systematic biases detected",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        # Prepare data
        bias_labels = [f"{true_label} → {pred_label}" for (true_label, pred_label), _ in global_biases]
        bias_counts = [count for _, count in global_biases]

        fig = go.Figure(data=go.Bar(
            x=bias_counts,
            y=bias_labels,
            orientation='h',
            text=bias_counts,
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
            xaxis_title="Error Frequency",
            yaxis_title="Confusion Pattern (True → Predicted)",
            height=max(400, len(global_biases) * 30)
        )

        return fig

    @staticmethod
    def create_training_recommendations_display(recommendations):
        """Create display for training recommendations."""
        print(f"[DEBUG] Creating training recommendations display")

        recommendation_cards = []

        for annotator, recs in recommendations.items():
            # Color based on overall priority
            high_priority_count = sum(1 for rec in recs if rec['priority'] == 'high')
            if high_priority_count > 0:
                card_color = "danger"
            elif any(rec['priority'] == 'medium' for rec in recs):
                card_color = "warning"
            else:
                card_color = "success"

            # Create recommendation list
            rec_items = []
            for rec in recs:
                priority_badge = dbc.Badge(rec['priority'].upper(),
                                         color={"high": "danger", "medium": "warning", "low": "info"}[rec['priority']])
                rec_items.append(
                    html.Li([priority_badge, " ", rec['message']], className="mb-2")
                )

            card = dbc.Card([
                dbc.CardHeader(html.H6(annotator.replace('A_', 'Annotator '))),
                dbc.CardBody([
                    html.Ul(rec_items, className="mb-0")
                ])
            ], color=card_color, outline=True, className="mb-3")

            recommendation_cards.append(dbc.Col(card, md=6))

        return dbc.Row(recommendation_cards)




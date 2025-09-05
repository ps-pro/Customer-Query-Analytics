# IAA Dashboard - Complete app.py file
# Inter-Annotator Agreement Analysis Dashboard
# Combines: Overall Agreement + Frequency Analysis + Hierarchical Analysis

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

# ========================================
# PASTE YOUR CALCULATOR CLASSES HERE
# ========================================

# [PASTE] IAAAgreementCalculator class (ENTIRE class from your original code)
class IAAAgreementCalculator:
    """Core class for Inter-Annotator Agreement calculations."""

    def __init__(self, agreement_df):
        """Initialize with agreement dataframe."""
        self.agreement_df = agreement_df
        self.annotators = sorted(agreement_df['annotator'].unique())
        self.documents = sorted(agreement_df['id'].unique())

    def prepare_krippendorff_data(self, df_subset, value_column):
        """Prepare data for Krippendorff's alpha calculation."""
        print(f"[DEBUG] Preparing Krippendorff data for column: {value_column}")
        print(f"[DEBUG] Input data shape: {df_subset.shape}")

        # Encode categorical labels
        le = LabelEncoder()
        df_copy = df_subset.copy()
        df_copy['encoded_labels'] = le.fit_transform(df_copy[value_column])

        # Create pivot table
        pivot_data = df_copy.pivot(index='annotator', columns='id', values='encoded_labels')
        print(f"[DEBUG] Pivot shape: {pivot_data.shape}")
        print(f"[DEBUG] Unique labels encoded: {len(le.classes_)}")

        # Convert to format required by krippendorff
        reliability_data = pivot_data.values.astype(float)

        return reliability_data, le

    def calculate_alpha_with_ci(self, df_subset, value_column, confidence_level=0.95, n_bootstrap=1000):
        """Calculate Krippendorff's alpha with bootstrap confidence intervals."""
        print(f"[DEBUG] Calculating alpha with CI for {value_column}")
        print(f"[DEBUG] Confidence level: {confidence_level}, Bootstrap samples: {n_bootstrap}")

        try:
            # Main calculation
            data_array, label_encoder = self.prepare_krippendorff_data(df_subset, value_column)
            alpha_main = krippendorff.alpha(reliability_data=data_array, level_of_measurement='nominal')

            # Bootstrap for confidence intervals
            alpha_bootstrap = []
            documents = df_subset['id'].unique()

            for i in range(n_bootstrap):
                # Resample documents with replacement
                bootstrap_docs = np.random.choice(documents, size=len(documents), replace=True)
                bootstrap_df = df_subset[df_subset['id'].isin(bootstrap_docs)]

                if len(bootstrap_df['id'].unique()) > 10:  # Minimum sample size check
                    try:
                        boot_data, _ = self.prepare_krippendorff_data(bootstrap_df, value_column)
                        alpha_boot = krippendorff.alpha(reliability_data=boot_data, level_of_measurement='nominal')
                        if not np.isnan(alpha_boot):
                            alpha_bootstrap.append(alpha_boot)
                    except:
                        continue

            # Calculate confidence intervals
            if len(alpha_bootstrap) > 0:
                alpha_ci_lower = np.percentile(alpha_bootstrap, ((1 - confidence_level) / 2) * 100)
                alpha_ci_upper = np.percentile(alpha_bootstrap, (1 - (1 - confidence_level) / 2) * 100)
            else:
                alpha_ci_lower = alpha_ci_upper = np.nan

            print(f"[DEBUG] Alpha calculated: {alpha_main:.4f}")
            print(f"[DEBUG] CI: [{alpha_ci_lower:.4f}, {alpha_ci_upper:.4f}]")

            return {
                'alpha': alpha_main,
                'ci_lower': alpha_ci_lower,
                'ci_upper': alpha_ci_upper,
                'n_bootstrap_valid': len(alpha_bootstrap)
            }

        except Exception as e:
            print(f"[ERROR] Alpha calculation failed: {str(e)}")
            return {
                'alpha': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_bootstrap_valid': 0
            }

    def calculate_pairwise_agreement_matrix(self, df_subset, value_column):
        """Calculate pairwise agreement matrix between annotators."""
        print(f"[DEBUG] Calculating pairwise agreement matrix for {value_column}")

        # Create pivot table
        pivot_data = df_subset.pivot(index='id', columns='annotator', values=value_column)
        annotators = pivot_data.columns

        # Initialize agreement matrix
        agreement_matrix = pd.DataFrame(index=annotators, columns=annotators, dtype=float)

        # Calculate pairwise agreements
        for i, ann1 in enumerate(annotators):
            for j, ann2 in enumerate(annotators):
                if i == j:
                    agreement_matrix.loc[ann1, ann2] = 100.0
                else:
                    # Calculate agreement between two annotators
                    valid_pairs = ~(pivot_data[ann1].isna() | pivot_data[ann2].isna())
                    if valid_pairs.sum() > 0:
                        agreements = (pivot_data.loc[valid_pairs, ann1] == pivot_data.loc[valid_pairs, ann2])
                        percentage = agreements.mean() * 100
                        agreement_matrix.loc[ann1, ann2] = percentage
                    else:
                        agreement_matrix.loc[ann1, ann2] = np.nan

        print(f"[DEBUG] Pairwise matrix calculated for {len(annotators)} annotators")
        return agreement_matrix

    def calculate_document_level_agreement(self, df_subset, value_column):
        """Calculate agreement statistics at document level."""
        print(f"[DEBUG] Calculating document-level agreement for {value_column}")

        # Create pivot table
        pivot_data = df_subset.pivot(index='id', columns='annotator', values=value_column)

        # Calculate per-document agreement
        doc_agreements = []
        for doc_id, row in pivot_data.iterrows():
            valid_annotations = row.dropna()
            if len(valid_annotations) > 1:
                # Calculate perfect agreement rate
                perfect_agreement = len(set(valid_annotations)) == 1
                doc_agreements.append({
                    'document_id': doc_id,
                    'n_annotators': len(valid_annotations),
                    'perfect_agreement': perfect_agreement,
                    'unique_labels': len(set(valid_annotations))
                })

        doc_agreement_df = pd.DataFrame(doc_agreements)

        print(f"[DEBUG] Document-level analysis completed for {len(doc_agreement_df)} documents")
        return doc_agreement_df


# Core Frequency Analysis Calculator
class FrequencyAnalysisCalculator:
    """Calculator for frequency-based IAA analysis."""

    def __init__(self, agreement_df):
        """Initialize with agreement dataframe."""
        self.agreement_df = agreement_df
        self.annotators = sorted(agreement_df['annotator'].unique())
        print(f"[DEBUG] FrequencyAnalysisCalculator initialized with {len(agreement_df)} annotations")

    def calculate_label_frequencies(self, df_subset, value_column):
        """Calculate frequency distribution of labels."""
        print(f"[DEBUG] Calculating label frequencies for column: {value_column}")

        # Count label frequencies across all annotations
        label_counts = Counter(df_subset[value_column])
        frequency_df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['frequency'])
        frequency_df['label'] = frequency_df.index
        frequency_df = frequency_df.sort_values('frequency', ascending=False)

        print(f"[DEBUG] Found {len(frequency_df)} unique labels")
        print(f"[DEBUG] Frequency range: {frequency_df['frequency'].min()} - {frequency_df['frequency'].max()}")

        return frequency_df

    def create_frequency_strata(self, frequency_df, rare_threshold, common_threshold):
        """Create frequency-based strata for labels."""
        print(f"[DEBUG] Creating frequency strata: rare<={rare_threshold}, common>={common_threshold}")

        # Classify labels into frequency strata
        frequency_df['stratum'] = 'moderate'
        frequency_df.loc[frequency_df['frequency'] <= rare_threshold, 'stratum'] = 'rare'
        frequency_df.loc[frequency_df['frequency'] >= common_threshold, 'stratum'] = 'common'

        # Count labels in each stratum
        stratum_counts = frequency_df['stratum'].value_counts()
        print(f"[DEBUG] Stratum distribution: {stratum_counts.to_dict()}")

        return frequency_df

    def prepare_krippendorff_data(self, df_subset, value_column):
        """Prepare data for Krippendorff's alpha calculation."""
        try:
            le = LabelEncoder()
            df_copy = df_subset.copy()
            df_copy['encoded_labels'] = le.fit_transform(df_copy[value_column])
            pivot_data = df_copy.pivot(index='annotator', columns='id', values='encoded_labels')
            reliability_data = pivot_data.values.astype(float)
            return reliability_data, le
        except Exception as e:
            print(f"[ERROR] Failed to prepare Krippendorff data: {str(e)}")
            return None, None

    def calculate_stratified_agreement(self, df_subset, value_column, frequency_strata_df,
                                     rare_threshold, common_threshold):
        """Calculate agreement metrics for each frequency stratum."""
        print(f"[DEBUG] Calculating stratified agreement analysis")

        # Create strata mapping
        strata_mapping = dict(zip(frequency_strata_df['label'], frequency_strata_df['stratum']))

        results = {}

        for stratum in ['rare', 'moderate', 'common']:
            print(f"[DEBUG] Processing stratum: {stratum}")

            # Get labels in this stratum
            stratum_labels = frequency_strata_df[frequency_strata_df['stratum'] == stratum]['label'].tolist()

            if len(stratum_labels) == 0:
                print(f"[WARNING] No labels in {stratum} stratum")
                results[stratum] = {
                    'alpha': np.nan,
                    'n_labels': 0,
                    'n_annotations': 0,
                    'n_documents': 0,
                    'avg_frequency': np.nan,
                    'labels': []
                }
                continue

            # Filter data to this stratum
            stratum_df = df_subset[df_subset[value_column].isin(stratum_labels)].copy()

            if len(stratum_df) < 10:  # Minimum sample size
                print(f"[WARNING] Insufficient data for {stratum} stratum: {len(stratum_df)} annotations")
                results[stratum] = {
                    'alpha': np.nan,
                    'n_labels': len(stratum_labels),
                    'n_annotations': len(stratum_df),
                    'n_documents': stratum_df['id'].nunique(),
                    'avg_frequency': frequency_strata_df[frequency_strata_df['stratum'] == stratum]['frequency'].mean(),
                    'labels': stratum_labels[:5]  # First 5 labels for display
                }
                continue

            # Calculate Krippendorff's alpha for this stratum
            try:
                data_array, _ = self.prepare_krippendorff_data(stratum_df, value_column)
                if data_array is not None:
                    alpha = krippendorff.alpha(reliability_data=data_array, level_of_measurement='nominal')
                else:
                    alpha = np.nan
            except Exception as e:
                print(f"[ERROR] Alpha calculation failed for {stratum}: {str(e)}")
                alpha = np.nan

            # Calculate stratum statistics
            results[stratum] = {
                'alpha': alpha,
                'n_labels': len(stratum_labels),
                'n_annotations': len(stratum_df),
                'n_documents': stratum_df['id'].nunique(),
                'avg_frequency': frequency_strata_df[frequency_strata_df['stratum'] == stratum]['frequency'].mean(),
                'labels': stratum_labels[:5]  # First 5 labels for display
            }

            print(f"[DEBUG] {stratum} stratum results: alpha={alpha:.4f}, n_labels={len(stratum_labels)}")

        return results

    def calculate_frequency_vs_agreement_correlation(self, df_subset, value_column, frequency_df):
        """Calculate correlation between label frequency and agreement."""
        print(f"[DEBUG] Calculating frequency vs agreement correlation")

        correlation_data = []

        for label in frequency_df['label']:
            # Get data for this specific label
            label_df = df_subset[df_subset[value_column] == label].copy()

            if len(label_df) < 5:  # Minimum sample size
                continue

            # Calculate perfect agreement rate for this label
            pivot_data = label_df.pivot(index='id', columns='annotator', values=value_column)
            perfect_agreements = 0
            total_docs = 0

            for doc_id, row in pivot_data.iterrows():
                valid_annotations = row.dropna()
                if len(valid_annotations) > 1:
                    total_docs += 1
                    if len(set(valid_annotations)) == 1:
                        perfect_agreements += 1

            if total_docs > 0:
                agreement_rate = perfect_agreements / total_docs
                correlation_data.append({
                    'label': label,
                    'frequency': frequency_df[frequency_df['label'] == label]['frequency'].iloc[0],
                    'agreement_rate': agreement_rate,
                    'n_documents': total_docs
                })

        correlation_df = pd.DataFrame(correlation_data)

        if len(correlation_df) > 0:
            correlation = np.corrcoef(correlation_df['frequency'], correlation_df['agreement_rate'])[0, 1]
            print(f"[DEBUG] Frequency-agreement correlation: {correlation:.4f}")
        else:
            correlation = np.nan
            print(f"[WARNING] No data for correlation analysis")

        return correlation_df, correlation


# Core Hierarchical Analysis Calculator
class HierarchicalAnalysisCalculator:
    """Calculator for hierarchical IAA analysis."""

    def __init__(self, agreement_df):
        """Initialize with agreement dataframe."""
        self.agreement_df = agreement_df
        self.annotators = sorted(agreement_df['annotator'].unique())
        self.parent_categories = sorted(agreement_df['L1_label'].unique())
        self.child_categories = sorted(agreement_df['L2_label'].unique())
        print(f"[DEBUG] HierarchicalAnalysisCalculator initialized")
        print(f"[DEBUG] Parent categories: {self.parent_categories}")
        print(f"[DEBUG] Child categories: {len(self.child_categories)} unique")

    def prepare_krippendorff_data(self, df_subset, value_column):
        """Prepare data for Krippendorff's alpha calculation."""
        try:
            if len(df_subset) < 10:
                print(f"[WARNING] Insufficient data for {value_column}: {len(df_subset)} rows")
                return None, None

            le = LabelEncoder()
            df_copy = df_subset.copy()
            df_copy['encoded_labels'] = le.fit_transform(df_copy[value_column])
            pivot_data = df_copy.pivot(index='annotator', columns='id', values='encoded_labels')
            reliability_data = pivot_data.values.astype(float)

            print(f"[DEBUG] Prepared data for {value_column}: shape {reliability_data.shape}")
            return reliability_data, le
        except Exception as e:
            print(f"[ERROR] Failed to prepare data for {value_column}: {str(e)}")
            return None, None

    def calculate_hierarchical_level_comparison(self, df_subset):
        """Calculate alpha for L1, L2, and Full labels for comparison."""
        print(f"[DEBUG] Calculating hierarchical level comparison")

        results = {}
        label_types = {
            'L1 (Parent)': 'L1_label',
            'L2 (Child)': 'L2_label',
            'Full Hierarchical': 'full_label'
        }

        for level_name, column in label_types.items():
            try:
                data_array, _ = self.prepare_krippendorff_data(df_subset, column)
                if data_array is not None:
                    alpha = krippendorff.alpha(reliability_data=data_array, level_of_measurement='nominal')
                    n_unique_labels = df_subset[column].nunique()
                    n_annotations = len(df_subset)
                    n_documents = df_subset['id'].nunique()
                else:
                    alpha = np.nan
                    n_unique_labels = 0
                    n_annotations = 0
                    n_documents = 0

                results[level_name] = {
                    'alpha': alpha,
                    'n_unique_labels': n_unique_labels,
                    'n_annotations': n_annotations,
                    'n_documents': n_documents
                }

                print(f"[DEBUG] {level_name}: alpha={alpha:.4f}, labels={n_unique_labels}")

            except Exception as e:
                print(f"[ERROR] Failed to calculate alpha for {level_name}: {str(e)}")
                results[level_name] = {
                    'alpha': np.nan,
                    'n_unique_labels': 0,
                    'n_annotations': 0,
                    'n_documents': 0
                }

        return results

    def calculate_conditional_agreement_by_parent(self, df_subset, selected_parents=None):
        """Calculate agreement within each parent category (conditional analysis)."""
        print(f"[DEBUG] Calculating conditional agreement by parent category")

        if selected_parents is None:
            selected_parents = self.parent_categories

        results = {}

        for parent in selected_parents:
            print(f"[DEBUG] Processing parent category: {parent}")

            # Filter data to this parent category
            parent_df = df_subset[df_subset['L1_label'] == parent].copy()

            if len(parent_df) < 10:
                print(f"[WARNING] Insufficient data for parent '{parent}': {len(parent_df)} annotations")
                results[parent] = {
                    'l2_alpha': np.nan,
                    'full_alpha': np.nan,
                    'n_child_labels': 0,
                    'n_annotations': len(parent_df),
                    'n_documents': parent_df['id'].nunique() if len(parent_df) > 0 else 0,
                    'child_labels': []
                }
                continue

            # Calculate L2 agreement within this parent
            try:
                l2_data, _ = self.prepare_krippendorff_data(parent_df, 'L2_label')
                l2_alpha = krippendorff.alpha(reliability_data=l2_data, level_of_measurement='nominal') if l2_data is not None else np.nan
            except Exception as e:
                print(f"[ERROR] L2 alpha calculation failed for {parent}: {str(e)}")
                l2_alpha = np.nan

            # Calculate Full agreement within this parent
            try:
                full_data, _ = self.prepare_krippendorff_data(parent_df, 'full_label')
                full_alpha = krippendorff.alpha(reliability_data=full_data, level_of_measurement='nominal') if full_data is not None else np.nan
            except Exception as e:
                print(f"[ERROR] Full alpha calculation failed for {parent}: {str(e)}")
                full_alpha = np.nan

            # Get child labels for this parent
            child_labels = sorted(parent_df['L2_label'].unique())

            results[parent] = {
                'l2_alpha': l2_alpha,
                'full_alpha': full_alpha,
                'n_child_labels': len(child_labels),
                'n_annotations': len(parent_df),
                'n_documents': parent_df['id'].nunique(),
                'child_labels': child_labels
            }

            print(f"[DEBUG] {parent}: L2_alpha={l2_alpha:.4f}, Full_alpha={full_alpha:.4f}")

        return results

    def calculate_specific_parent_child_combinations(self, df_subset, selected_parents=None):
        """Calculate agreement for specific parent-child label combinations."""
        print(f"[DEBUG] Calculating specific parent-child combination analysis")

        if selected_parents is None:
            selected_parents = self.parent_categories

        combination_results = []

        for parent in selected_parents:
            parent_df = df_subset[df_subset['L1_label'] == parent].copy()
            child_labels = parent_df['L2_label'].unique()

            for child in child_labels:
                combination_df = parent_df[parent_df['L2_label'] == child].copy()

                if len(combination_df) < 5:  # Minimum threshold for specific combinations
                    continue

                # Calculate simple agreement rate for this specific combination
                pivot_data = combination_df.pivot(index='id', columns='annotator', values='full_label')
                perfect_agreements = 0
                total_docs = 0

                for doc_id, row in pivot_data.iterrows():
                    valid_annotations = row.dropna()
                    if len(valid_annotations) > 1:
                        total_docs += 1
                        if len(set(valid_annotations)) == 1:
                            perfect_agreements += 1

                agreement_rate = perfect_agreements / total_docs if total_docs > 0 else 0

                combination_results.append({
                    'parent': parent,
                    'child': child,
                    'full_label': f"{parent}_{child}",
                    'agreement_rate': agreement_rate,
                    'n_annotations': len(combination_df),
                    'n_documents': total_docs,
                    'perfect_agreements': perfect_agreements
                })

        combination_df = pd.DataFrame(combination_results)
        print(f"[DEBUG] Analyzed {len(combination_df)} parent-child combinations")

        return combination_df

    def calculate_hierarchical_consistency_metrics(self, level_comparison_results):
        """Calculate metrics showing how hierarchy affects agreement."""
        print(f"[DEBUG] Calculating hierarchical consistency metrics")

        l1_alpha = level_comparison_results['L1 (Parent)']['alpha']
        l2_alpha = level_comparison_results['L2 (Child)']['alpha']
        full_alpha = level_comparison_results['Full Hierarchical']['alpha']

        # Hierarchical consistency: how well does full agree compared to components
        if not np.isnan(l1_alpha) and not np.isnan(full_alpha) and l1_alpha > 0:
            l1_consistency = full_alpha / l1_alpha
        else:
            l1_consistency = np.nan

        if not np.isnan(l2_alpha) and not np.isnan(full_alpha) and l2_alpha > 0:
            l2_consistency = full_alpha / l2_alpha
        else:
            l2_consistency = np.nan

        # Agreement hierarchy: which level has highest agreement
        alphas = {'L1': l1_alpha, 'L2': l2_alpha, 'Full': full_alpha}
        valid_alphas = {k: v for k, v in alphas.items() if not np.isnan(v)}

        if valid_alphas:
            best_level = max(valid_alphas, key=valid_alphas.get)
            worst_level = min(valid_alphas, key=valid_alphas.get)
            alpha_range = max(valid_alphas.values()) - min(valid_alphas.values())
        else:
            best_level = worst_level = "N/A"
            alpha_range = np.nan

        consistency_metrics = {
            'l1_consistency': l1_consistency,
            'l2_consistency': l2_consistency,
            'best_agreement_level': best_level,
            'worst_agreement_level': worst_level,
            'alpha_range': alpha_range,
            'hierarchy_impact': 'Positive' if full_alpha > max(l1_alpha, l2_alpha) else 'Negative' if not np.isnan(full_alpha) else 'Unknown'
        }

        print(f"[DEBUG] Consistency metrics calculated: {consistency_metrics}")
        return consistency_metrics
    
# ========================================
# PASTE YOUR VISUALIZER CLASSES HERE  
# ========================================

# [PASTE] IAAAgreementVisualizer class (ENTIRE class from your original code)
class IAAAgreementVisualizer:
    """Class for creating IAA visualizations."""

    @staticmethod
    def create_agreement_heatmap(agreement_matrix, title="Annotator Agreement Matrix"):
        """Create interactive heatmap of annotator agreement."""
        print(f"[DEBUG] Creating heatmap: {title}")

        fig = go.Figure(data=go.Heatmap(
            z=agreement_matrix.values,
            x=agreement_matrix.columns,
            y=agreement_matrix.index,
            colorscale='RdYlBu',
            zmin=0,
            zmax=100,
            text=agreement_matrix.round(1),
            texttemplate='%{text}%',
            textfont={"size": 12},
            colorbar=dict(title="Agreement %")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Annotator",
            yaxis_title="Annotator",
            width=600,
            height=500
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
            title=title,
            xaxis_title="Label Type",
            yaxis_title="Krippendorff's Alpha",
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )

        return fig

    @staticmethod
    def create_document_agreement_histogram(doc_agreement_df, title="Document-Level Agreement Distribution"):
        """Create histogram of document-level agreement rates."""
        print(f"[DEBUG] Creating document agreement histogram: {title}")

        perfect_agreement_rate = doc_agreement_df['perfect_agreement'].mean() * 100

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=doc_agreement_df['unique_labels'],
            nbinsx=max(doc_agreement_df['unique_labels']) + 1,
            name="Document Count",
            textposition='outside'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Number of Unique Labels per Document",
            yaxis_title="Number of Documents",
            bargap=0.1
        )

        return fig

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
            title="Label Frequency Distribution with Stratum Boundaries",
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
            title="Agreement Analysis Across Frequency Strata",
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
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=correlation_df['frequency'],
            y=correlation_df['agreement_rate'],
            mode='markers',
            text=correlation_df['label'],
            hovertemplate='<b>%{text}</b><br>Frequency: %{x}<br>Agreement Rate: %{y:.2%}<extra></extra>',
            marker=dict(
                size=correlation_df['n_documents'],
                sizeref=2 * max(correlation_df['n_documents']) / (17**2),
                sizemin=4,
                color=correlation_df['agreement_rate'],
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title="Agreement Rate")
            )
        ))

        # Add trend line
        if len(correlation_df) > 1:
            z = np.polyfit(correlation_df['frequency'], correlation_df['agreement_rate'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(correlation_df['frequency'].min(), correlation_df['frequency'].max(), 100)
            y_trend = p(x_trend)

            fig.add_trace(go.Scatter(
                x=x_trend, y=y_trend,
                mode='lines',
                name=f'Trend (r={correlation_coef:.3f})',
                line=dict(dash='dash', color='red')
            ))

        fig.update_layout(
            title="Label Frequency vs Agreement Rate",
            xaxis_title="Label Frequency",
            yaxis_title="Agreement Rate",
            showlegend=True
        )

        return fig


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
            title="Hierarchical Level Analysis",
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
            title="Conditional Agreement Analysis by Parent Category",
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
            title="Agreement Rates by Parent-Child Combination",
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

# ========================================
# DATA LOADING
# ========================================

# Load data from CSV file
agreement_df = pd.read_csv('data.csv')
print(f"[INFO] Loaded {len(agreement_df)} annotations from data.csv")

# ========================================
# INITIALIZE COMPONENTS
# ========================================

# [PASTE] Initialize your calculators (copy from your original code)
calculator = IAAAgreementCalculator(agreement_df)
freq_calculator = FrequencyAnalysisCalculator(agreement_df)  
hier_calculator = HierarchicalAnalysisCalculator(agreement_df)

# [PASTE] Initialize your visualizers (copy from your original code)
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

# Add this right after app initialization
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
            }
            
            /* Dark Theme */
            .theme-dark {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --text-primary: #ffffff;
                --text-secondary: #adb5bd;
                --border-color: #495057;
                --card-bg: #2d2d2d;
                --shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.3);
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
            
            #main-title {
                color: var(--text-primary) !important;
                font-size: 2.5rem !important;
            }
            
            #main-subtitle {
                color: var(--text-secondary) !important;
            }
            
            /* Card styling */
            .card {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                box-shadow: var(--shadow) !important;
                border-radius: 0.5rem !important;
            }
            
            .card-header {
                background-color: var(--bg-secondary) !important;
                border-bottom: 1px solid var(--border-color) !important;
                color: var(--text-primary) !important;
            }
            
            .card-body {
                color: var(--text-primary) !important;
            }

            /* Enhanced form controls */
            .Select-control, .dropdown {
                border-radius: 10px !important;
                border: 2px solid rgba(102, 126, 234, 0.2) !important;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1) !important;
                transition: all 0.3s ease !important;
            }

            .Select-control:hover, .dropdown:hover {
                border-color: rgba(102, 126, 234, 0.4) !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
            }

            /* Slider styling */
            .rc-slider-track {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                height: 6px !important;
            }

            .rc-slider-handle {
                border: 3px solid #667eea !important;
                background: white !important;
                width: 20px !important;
                height: 20px !important;
                margin-top: -7px !important;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
            }

            .rc-slider-rail {
                background: rgba(102, 126, 234, 0.2) !important;
                height: 6px !important;
            }

            /* Checkbox styling */
            input[type="checkbox"] {
                accent-color: #667eea !important;
                width: 18px !important;
                height: 18px !important;
            }

            /* Button hover effect */
            .btn:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            }            
            /* Tab styling */
            .nav-tabs {
                border-bottom: 1px solid var(--border-color) !important;
            }

            .nav-tabs .nav-link {
                color: var(--text-secondary) !important;
                border: none !important;
                background: transparent !important;
                font-weight: 600 !important;
                fontSize: 1.1rem !important;
            }

            .nav-tabs .nav-link.active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                background-clip: text !important;
                border: none !important;
                border-bottom: 3px solid transparent !important;
                background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%), linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                background-size: 100% 100%, 100% 3px !important;
                background-position: 0 0, 0 100% !important;
                background-repeat: no-repeat !important;
                font-weight: 700 !important;
            }
            /* Button styling */
            .btn {
                border-radius: 0.375rem !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
            }
            
            .btn-outline-secondary {
                color: var(--text-secondary) !important;
                border-color: var(--border-color) !important;
            }
            
            .btn-outline-secondary:hover {
                background-color: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
            }
            
            /* Plot container styling */
            .js-plotly-plot {
                border-radius: 0.5rem !important;
                box-shadow: var(--shadow) !important;
            }
            
            /* Table styling */
            .dash-table-container {
                background-color: var(--card-bg) !important;
                border-radius: 0.5rem !important;
                overflow: hidden !important;
            }
            
            /* Progress bar */
            .progress {
                background-color: var(--bg-secondary) !important;
                border-radius: 0.5rem !important;
            }
            
            /* Input styling */
            .form-control, .form-select {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                color: var(--text-primary) !important;
            }
            
            .form-control:focus, .form-select:focus {
                background-color: var(--card-bg) !important;
                border-color: #007bff !important;
                color: var(--text-primary) !important;
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
                    html.H1("Inter-Annotator Agreement Dashboard", 
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
# TAB CONTENT FUNCTIONS
# ========================================

def create_overall_tab_content():
    """Create Overall Agreement Analysis tab content."""
    return html.Div([
        html.H4("Overall Agreement Analysis", 
                style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'backgroundClip': 'text',
                    'fontWeight': '700',
                    'fontSize': '2rem',
                    'marginBottom': '2rem',
                    'textAlign': 'center'
                }),
        
        # Controls Section - Redesigned
        dbc.Card([
            dbc.CardHeader([
                html.H5("Analysis Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Annotators
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='overall-annotator-selector',
                                    options=[{'label': ann, 'value': ann} for ann in calculator.annotators],
                                    value=calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=12)
                ], className="mb-4"),
                
                # Row 2: Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='overall-label-type-selector',
                            options=[
                                {'label': 'Full Hierarchical Labels', 'value': 'full_label'},
                                {'label': 'L1 (Parent) Labels', 'value': 'L1_label'},
                                {'label': 'L2 (Child) Labels', 'value': 'L2_label'}
                            ],
                            value='full_label',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Confidence Level:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='overall-confidence-slider',
                                min=0.90,
                                max=0.99,
                                step=0.01,
                                value=0.95,
                                marks={0.90: {'label': '90%', 'style': {'fontSize': '14px'}}, 
                                       0.95: {'label': '95%', 'style': {'fontSize': '14px'}}, 
                                       0.99: {'label': '99%', 'style': {'fontSize': '14px'}}},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=4),
                    dbc.Col([
                        html.Div(style={'height': '2rem'}),  # Spacer
                        dbc.Button("Calculate Agreement Analysis", 
                                 id="overall-calculate-btn",
                                 size="lg", 
                                 style={
                                     'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                     'border': 'none',
                                     'borderRadius': '12px',
                                     'fontWeight': '600',
                                     'fontSize': '1.1rem',
                                     'padding': '12px 30px',
                                     'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)',
                                     'transition': 'all 0.3s ease',
                                     'width': '100%'
                                 })
                    ], md=4)
                ])
            ], style={'padding': '2rem'})
        ], style={
            'border': '1px solid rgba(102, 126, 234, 0.3)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)',
            'marginBottom': '3rem'
        }),

        # Progress indicator
        dbc.Row([
            dbc.Col([
                dbc.Progress(id="overall-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),
            
        # Results container with better spacing
        html.Div(
            id="overall-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})

def create_frequency_tab_content():
    """Create Frequency-Based Analysis tab content."""
    return html.Div([
        html.H4("Frequency-Based Analysis", 
                style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'backgroundClip': 'text',
                    'fontWeight': '700',
                    'fontSize': '2rem',
                    'marginBottom': '2rem',
                    'textAlign': 'center'
                }),

        # Controls Section - Redesigned
        dbc.Card([
            dbc.CardHeader([
                html.H5("Frequency Stratification Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Label Type and Thresholds
                dbc.Row([
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='freq-label-type-selector',
                            options=[
                                {'label': 'Full Hierarchical Labels', 'value': 'full_label'},
                                {'label': 'L1 (Parent) Labels', 'value': 'L1_label'},
                                {'label': 'L2 (Child) Labels', 'value': 'L2_label'}
                            ],
                            value='full_label',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Rare Threshold (≤):", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='rare-threshold-slider',
                                min=50,
                                max=500,
                                step=25,
                                value=200,
                                marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(50, 501, 100)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=4),
                    dbc.Col([
                        html.Label("Common Threshold (≥):", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='common-threshold-slider',
                                min=800,
                                max=2000,
                                step=50,
                                value=1200,
                                marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(800, 2001, 300)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=4)
                ], className="mb-4"),
                
                # Row 2: Annotators
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='freq-annotator-selector',
                                    options=[{'label': ann, 'value': ann} for ann in freq_calculator.annotators],
                                    value=freq_calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=9),
                    dbc.Col([
                        html.Div(style={'height': '3rem'}),  # Spacer
                        dbc.Button("Calculate Frequency Analysis", 
                                 id="freq-calculate-btn",
                                 size="lg", 
                                 style={
                                     'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                     'border': 'none',
                                     'borderRadius': '12px',
                                     'fontWeight': '600',
                                     'fontSize': '1.1rem',
                                     'padding': '12px 30px',
                                     'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)',
                                     'transition': 'all 0.3s ease',
                                     'width': '100%',
                                     'minHeight': '50px'
                                 })
                    ], md=3)
                ])
            ], style={'padding': '2rem'})
        ], style={
            'border': '1px solid rgba(102, 126, 234, 0.3)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)',
            'marginBottom': '3rem'
        }),

        # Threshold Explanation Card
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6("Frequency Stratification Guide", 
                           style={'fontWeight': '600', 'color': '#667eea', 'marginBottom': '1rem'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("Rare Labels", style={'fontWeight': '600', 'color': '#e74c3c'}),
                                html.Span(f" ≤ threshold", style={'color': 'var(--text-secondary)'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Moderate Labels", style={'fontWeight': '600', 'color': '#f39c12'}),
                                html.Span(" between thresholds", style={'color': 'var(--text-secondary)'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Common Labels", style={'fontWeight': '600', 'color': '#27ae60'}),
                                html.Span(" ≥ threshold", style={'color': 'var(--text-secondary)'})
                            ])
                        ], md=4)
                    ])
                ])
            ], style={'padding': '1.5rem'})
        ], style={
            'backgroundColor': 'rgba(102, 126, 234, 0.05)',
            'border': '1px solid rgba(102, 126, 234, 0.2)',
            'borderRadius': '12px',
            'marginBottom': '3rem'
        }),

        # Progress indicator
        dbc.Row([
            dbc.Col([
                dbc.Progress(id="freq-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),
        
        # Results container with better spacing
        html.Div(
            id="freq-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})



def create_hierarchical_tab_content():
    """Create Hierarchical Analysis tab content."""
    return html.Div([
        html.H4("Hierarchical Analysis", 
                style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'backgroundClip': 'text',
                    'fontWeight': '700',
                    'fontSize': '2rem',
                    'marginBottom': '2rem',
                    'textAlign': 'center'
                }),
        
        # Controls Section - Redesigned
        dbc.Card([
            dbc.CardHeader([
                html.H5("Hierarchical Analysis Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Analysis Type and Parent Categories
                dbc.Row([
                    dbc.Col([
                        html.Label("Analysis Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='analysis-type-selector',
                            options=[
                                {'label': 'Complete Analysis', 'value': 'complete'},
                                {'label': 'Level Comparison Only', 'value': 'levels'},
                                {'label': 'Conditional Analysis Only', 'value': 'conditional'}
                            ],
                            value='complete',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Select Parent Categories:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='parent-category-selector',
                                    options=[{'label': cat, 'value': cat} for cat in hier_calculator.parent_categories],
                                    value=hier_calculator.parent_categories,
                                    inline=True,
                                    style={'fontSize': '1rem'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=8)
                ], className="mb-4"),
                
                # Row 2: Annotators and Calculate Button
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='hier-annotator-selector',
                                    options=[{'label': ann, 'value': ann} for ann in hier_calculator.annotators],
                                    value=hier_calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=9),
                    dbc.Col([
                        html.Div(style={'height': '3rem'}),  # Spacer
                        dbc.Button("Calculate Hierarchical Analysis", 
                                 id="hier-calculate-btn",
                                 size="lg", 
                                 style={
                                     'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                     'border': 'none',
                                     'borderRadius': '12px',
                                     'fontWeight': '600',
                                     'fontSize': '1.1rem',
                                     'padding': '12px 30px',
                                     'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)',
                                     'transition': 'all 0.3s ease',
                                     'width': '100%',
                                     'minHeight': '50px'
                                 })
                    ], md=3)
                ])
            ], style={'padding': '2rem'})
        ], style={
            'border': '1px solid rgba(102, 126, 234, 0.3)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)',
            'marginBottom': '3rem'
        }),

        # Analysis Type Explanation Card
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6("Analysis Type Guide", 
                           style={'fontWeight': '600', 'color': '#667eea', 'marginBottom': '1rem'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("Complete Analysis", style={'fontWeight': '600', 'color': '#667eea'}),
                                html.P("Performs both level comparison and conditional analysis for comprehensive insights", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Level Comparison", style={'fontWeight': '600', 'color': '#f39c12'}),
                                html.P("Compares agreement across L1, L2, and full hierarchical labels", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Conditional Analysis", style={'fontWeight': '600', 'color': '#27ae60'}),
                                html.P("Analyzes agreement within each parent category separately", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4)
                    ])
                ])
            ], style={'padding': '1.5rem'})
        ], style={
            'backgroundColor': 'rgba(102, 126, 234, 0.05)',
            'border': '1px solid rgba(102, 126, 234, 0.2)',
            'borderRadius': '12px',
            'marginBottom': '3rem'
        }),

        # Progress indicator
        dbc.Row([
            dbc.Col([
                dbc.Progress(id="hier-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),
        
        # Results container with better spacing
        html.Div(
            id="hier-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})

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
        doc_fig = visualizer.create_document_agreement_histogram(doc_agreement)

        # Create results layout
        results = html.Div([
            # Summary Statistics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Summary Statistics")),
                        dbc.CardBody([
                            html.P(f"Krippendorff's Alpha: {alpha_result['alpha']:.4f}"),
                            html.P(f"{confidence_level*100:.0f}% CI: [{alpha_result['ci_lower']:.4f}, {alpha_result['ci_upper']:.4f}]"),
                            html.P(f"Annotators: {len(selected_annotators)}"),
                            html.P(f"Documents: {len(filtered_df['id'].unique())}"),
                            html.P(f"Bootstrap samples: {alpha_result['n_bootstrap_valid']}")
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Interpretation")),
                        dbc.CardBody([
                            html.P(get_alpha_interpretation(alpha_result['alpha'])),
                            html.P(f"Perfect agreement: {doc_agreement['perfect_agreement'].mean():.1%} of documents"),
                            html.P(f"Average pairwise agreement: {agreement_matrix.values[np.triu_indices_from(agreement_matrix.values, k=1)].mean():.1f}%")
                        ])
                    ])
                ], md=8)
            ], className="mb-4"),

            # Visualizations
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=heatmap_fig)
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=alpha_fig)
                ], md=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=doc_fig)
                ], md=12)
            ])
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
            summary_data.append({
                'Frequency Stratum': stratum.title(),
                'Alpha': f"{result['alpha']:.4f}" if not np.isnan(result['alpha']) else "N/A",
                'Labels': result['n_labels'],
                'Annotations': result['n_annotations'],
                'Documents': result['n_documents'],
                'Avg Frequency': f"{result['avg_frequency']:.1f}" if not np.isnan(result['avg_frequency']) else "N/A"
            })

        summary_table = dash_table.DataTable(
            data=summary_data,
            columns=[{"name": i, "id": i} for i in summary_data[0].keys()],
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )

        # Create results layout
        results = html.Div([
            # Summary Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Frequency Analysis Summary")),
                        dbc.CardBody([
                            summary_table,
                            html.Hr(),
                            html.P(f"Frequency-Agreement Correlation: {correlation_coef:.3f}" if not np.isnan(correlation_coef) else "Correlation: N/A"),
                            html.P(f"Threshold Settings: Rare ≤ {rare_threshold}, Common ≥ {common_threshold}")
                        ])
                    ])
                ])
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
# HELPER FUNCTIONS
# ========================================

# [PASTE] Copy any helper functions used by your callbacks
# Example: get_agreement_interpretation(), get_comparison(), etc.

def get_alpha_interpretation(alpha):
    """Get interpretation text for alpha value."""
    if alpha >= 0.8:
        return "Excellent agreement (α ≥ 0.8)"
    elif alpha >= 0.67:
        return "Good agreement (α ≥ 0.67)"
    elif alpha >= 0.4:
        return "Fair agreement (α ≥ 0.4)"
    else:
        return "Poor agreement (α < 0.4)"

def get_agreement_interpretation(alpha):
    """Get text interpretation of alpha value."""
    if alpha >= 0.8:
        return "Excellent agreement"
    elif alpha >= 0.67:
        return "Good agreement"
    elif alpha >= 0.4:
        return "Fair agreement"
    else:
        return "Poor agreement"

def get_comparison(val1, val2):
    """Get comparison text between two values."""
    if val1 > val2:
        return "higher"
    elif val1 < val2:
        return "lower"
    else:
        return "similar"

# ========================================
# END OF FILE
# ========================================

# ###### THE BELOW CODE IS ONLY FOR LOCAL TESTING ######
# if __name__ == '__main__':
#     app.run(debug=True, host='127.0.0.1', port=8050)
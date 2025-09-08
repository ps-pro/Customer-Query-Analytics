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
    

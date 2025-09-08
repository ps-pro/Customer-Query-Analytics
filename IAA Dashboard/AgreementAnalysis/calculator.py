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

        print(f"[DEBUG] Sample-level analysis completed for {len(doc_agreement_df)} samples")
        return doc_agreement_df


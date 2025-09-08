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


class FrequencyAnalysisCalculator:
    """Calculator for frequency-based IAA analysis."""

    def __init__(self, agreement_df):
        """Initialize with agreement dataframe."""
        self.agreement_df = agreement_df
        self.annotators = sorted(agreement_df['annotator'].unique())
        print(f"[DEBUG] FrequencyAnalysisCalculator initialized with {len(agreement_df)} annotations")

    def debug_agreement_calculation(self, df_subset, value_column):
        """Debug function to check agreement calculations."""
        print(f"[DEBUG] === AGREEMENT CALCULATION DEBUG ===")
        print(f"[DEBUG] Total annotations: {len(df_subset)}")
        print(f"[DEBUG] Unique documents: {df_subset['id'].nunique()}")
        print(f"[DEBUG] Unique annotators: {df_subset['annotator'].nunique()}")
        print(f"[DEBUG] Unique labels: {df_subset[value_column].nunique()}")
        
        # Check label distribution
        label_counts = df_subset[value_column].value_counts()
        print(f"[DEBUG] Label distribution:")
        for label, count in label_counts.head(10).items():
            print(f"[DEBUG]   {label}: {count}")
        
        # Check annotator coverage per document (sample)
        sample_docs = df_subset['id'].unique()[:5]
        for doc in sample_docs:
            doc_data = df_subset[df_subset['id'] == doc]
            print(f"[DEBUG] Doc {doc}: {len(doc_data)} annotators, labels: {set(doc_data[value_column])}")


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
                    'alpha': 0.0,  # Use 0 instead of NaN for empty strata
                    'n_labels': 0,
                    'n_annotations': 0,
                    'n_documents': 0,
                    'avg_frequency': 0.0,  # Use 0 instead of NaN
                    'labels': []
                }
                continue

            # Filter data to this stratum
            stratum_df = df_subset[df_subset[value_column].isin(stratum_labels)].copy()

            if len(stratum_df) < 1:  # Minimum sample size
                print(f"[WARNING] Insufficient data for {stratum} stratum: {len(stratum_df)} annotations")

                results[stratum] = {
                    'alpha': 0.0,  # Use 0 for insufficient data instead of NaN
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
                if data_array is not None and data_array.shape[0] >= 2 and data_array.shape[1] >= 2:
                    # Check if there's enough variation in the data
                    unique_values = np.unique(data_array[~np.isnan(data_array)])
                    if len(unique_values) > 1:
                        alpha = krippendorff.alpha(reliability_data=data_array, level_of_measurement='nominal')
                        if np.isnan(alpha):
                            print(f"[WARNING] Krippendorff returned NaN for {stratum} - calculating simple agreement")
                            # Fallback to simple agreement calculation
                            alpha = self.calculate_simple_agreement_rate(stratum_df, value_column)
                    else:
                        print(f"[WARNING] No variation in {stratum} labels - perfect agreement")
                        alpha = 1.0
                else:
                    alpha = np.nan
                    print(f"[WARNING] Insufficient data structure for {stratum}: shape {data_array.shape if data_array is not None else 'None'}")
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
    
    # Place the function Here
    def calculate_frequency_vs_agreement_correlation(self, df_subset, value_column, frequency_df):
        """Calculate correlation between label frequency and Krippendorff's alpha per label."""
        print(f"[DEBUG] Calculating frequency vs agreement correlation")

        correlation_data = []

        for label in frequency_df['label']:
            # Get all documents that contain this label from any annotator
            docs_with_label = df_subset[df_subset[value_column] == label]['id'].unique()
            
            if len(docs_with_label) < 5:  # Need minimum documents
                print(f"[DEBUG] Skipping {label}: only {len(docs_with_label)} documents")
                continue

            # Create subset for this label analysis
            label_docs_df = df_subset[df_subset['id'].isin(docs_with_label)].copy()
            
            try:
                # Calculate Krippendorff's alpha specifically for documents containing this label
                data_array, _ = self.prepare_krippendorff_data(label_docs_df, value_column)
                
                if data_array is not None and data_array.shape[0] >= 2 and data_array.shape[1] >= 5:
                    alpha = krippendorff.alpha(reliability_data=data_array, level_of_measurement='nominal')
                    
                    if not np.isnan(alpha):
                        frequency_val = frequency_df[frequency_df['label'] == label]['frequency'].iloc[0]
                        correlation_data.append({
                            'label': label,
                            'frequency': frequency_val,
                            'agreement_rate': alpha,
                            'n_documents': len(docs_with_label)
                        })
                        print(f"[DEBUG] {label}: freq={frequency_val}, alpha={alpha:.3f}, docs={len(docs_with_label)}")
                    else:
                        print(f"[DEBUG] {label}: Krippendorff returned NaN")
                else:
                    print(f"[DEBUG] {label}: Insufficient data array shape: {data_array.shape if data_array is not None else 'None'}")
            
            except Exception as e:
                print(f"[ERROR] Failed to process {label}: {str(e)}")
                continue

        correlation_df = pd.DataFrame(correlation_data)
        print(f"[DEBUG] Created correlation dataframe with {len(correlation_df)} labels")

        if len(correlation_df) > 2:
            try:
                # Check variance in both variables
                freq_variance = np.var(correlation_df['frequency'])
                agreement_variance = np.var(correlation_df['agreement_rate'])
                
                print(f"[DEBUG] Frequency variance: {freq_variance:.2f}")
                print(f"[DEBUG] Agreement variance: {agreement_variance:.6f}")
                
                if agreement_variance > 0.001 and freq_variance > 0:
                    correlation = np.corrcoef(correlation_df['frequency'], correlation_df['agreement_rate'])[0, 1]
                    print(f"[DEBUG] Frequency-agreement correlation: {correlation:.4f}")
                else:
                    correlation = 0.0
                    print(f"[DEBUG] Insufficient variance for meaningful correlation")
            except Exception as e:
                print(f"[ERROR] Correlation calculation failed: {str(e)}")
                correlation = np.nan
        else:
            correlation = np.nan
            print(f"[WARNING] Too few labels for correlation: {len(correlation_df)}")

        return correlation_df, correlation



    def calculate_simple_agreement_rate(self, df_subset, value_column):
        """Calculate simple agreement rate as fallback."""
        try:
            pivot_data = df_subset.pivot(index='id', columns='annotator', values=value_column)
            perfect_agreements = 0
            total_docs = 0
            
            for doc_id, row in pivot_data.iterrows():
                valid_annotations = row.dropna()
                if len(valid_annotations) > 1:
                    total_docs += 1
                    if len(set(valid_annotations)) == 1:
                        perfect_agreements += 1
            
            if total_docs > 0:
                return perfect_agreements / total_docs
            else:
                return 0.0
        except Exception as e:
            print(f"[ERROR] Simple agreement calculation failed: {str(e)}")
            return 0.0


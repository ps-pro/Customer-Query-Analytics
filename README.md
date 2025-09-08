# Customer Query Analytics Platform

A comprehensive analytics platform for customer support ticket classification, featuring multi-annotator agreement analysis, data quality assessment, and human-in-the-loop machine learning workflows.

## Live Deployments

<div align="center">

[![IAA Dashboard](https://img.shields.io/badge/IAA%20Dashboard-Live-blue?style=for-the-badge&logo=plotly&logoColor=white)](https://ptc-iaa.plotly.app/)
[![Data Quality Dashboard](https://img.shields.io/badge/Data%20Quality%20Dashboard-Live-green?style=for-the-badge&logo=plotly&logoColor=white)](https://ptc-dq.plotly.app/)
[![HITL Dashboard](https://img.shields.io/badge/Classifier%20HITL%20Dashboard-Live-orange?style=for-the-badge&logo=plotly&logoColor=white)](https://ptc-chitl.plotly.app/)

</div>

## Overview

This platform provides end-to-end analytics for customer support ticket classification systems, from synthetic data generation through advanced disagreement analysis and automated classification improvement. The system simulates realistic annotation scenarios with multiple human annotators exhibiting different behavioral patterns and skill levels.

## Architecture

### Core Components

1. **Dataset Generation Engine**: Synthetic customer support ticket generation with simulated multi-annotator responses
2. **Data Processing Pipeline**: Comprehensive cleaning, standardization, and conversion workflows
3. **Analytics Dashboards**: Three specialized dashboard applications for different analytical perspectives
4. **Business Intelligence Module**: Topic modeling and hierarchical analysis capabilities

### Technical Stack

- **Backend**: Python 3.11+
- **Web Framework**: Dash (Plotly)
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, gensim
- **Statistics**: krippendorff (inter-annotator agreement)
- **Visualization**: Plotly, matplotlib, seaborn
- **Text Processing**: NLTK, TF-IDF
- **Deployment**: Plotly Cloud

## Key Features

### Data Generation & Simulation
- **Hyper-realistic synthetic dataset generation** with 50 customer support tickets across 3 difficulty levels
- **7 distinct annotator personas** with documented behavioral patterns and error tendencies
- **Hierarchical labeling system** (Level 1: Technical Issue/Billing/Account Management, Level 2: specific sub-categories)
- **Deliberate ambiguity injection** to simulate real-world annotation challenges

### Inter-Annotator Agreement Analysis
- **Krippendorff's Alpha calculation** with bootstrap confidence intervals (n=500)
- **Pairwise agreement matrices** between all annotator combinations
- **Hierarchical consistency metrics** comparing L1, L2, and full label agreement
- **Document-level agreement distribution analysis**

### Data Quality Assessment
- **Disagreement scoring algorithm** with confidence-weighted metrics
- **Label confusion matrix analysis** identifying systematic classification errors
- **Gold-set refresh recommendations** based on disagreement patterns
- **Text complexity correlation analysis** with annotation difficulty

### Automated Classification & HITL
- **Rule-based classifier** with boolean logic expressions and keyword matching
- **Fuzzy matching classifier** with character-level and semantic similarity options
- **Performance comparison framework** with comprehensive error pattern analysis
- **CRUD interfaces** for rule management and training example updates
- **Real-time testing environment** for classifier validation

### Business Analytics
- **Latent Dirichlet Allocation (LDA)** topic modeling with 5-topic extraction
- **Interactive sunburst visualizations** for hierarchical label distributions
- **Word frequency analysis** and cloud generation
- **Customer request profiling** with thematic categorization

## Repository Structure

```
Customer-Query-Analytics/
├── Annotation Analytics/
│   ├── Distributions.ipynb              # Label distribution analysis
│   └── Heirarchical_View.ipynb         # Sunburst chart generation
├── Business Analytics/
│   └── Customer_Request_Topic_Profile.ipynb  # LDA topic modeling
├── Data Prepration/
│   ├── Data_Cleaning.ipynb             # Data standardization pipeline
│   ├── Convertor.ipynb                 # JSON to CSV transformation
│   └── Pre-Analysis.ipynb              # Exploratory data analysis
├── IAA Dashboard/
│   ├── app.py                          # Main dashboard application
│   ├── AgreementAnalysis/
│   │   ├── calculator.py               # Krippendorff's Alpha implementation
│   │   ├── visualizer.py              # Agreement visualization components
│   │   └── content.py                 # Dashboard content generation
│   ├── FrequencyAnalysis/
│   │   ├── calculator.py              # Label frequency calculations
│   │   ├── visualizer.py              # Frequency charts
│   │   └── content.py                 # Frequency analysis interface
│   ├── HierarchicalAnalysis/
│   │   ├── calculator.py              # Hierarchical consistency metrics
│   │   ├── visualizer.py              # Hierarchical visualizations
│   │   └── content.py                 # Hierarchical analysis interface
│   └── utils/
│       └── helpers.py                 # Utility functions
├── Data Quality Dashboard/
│   ├── app.py                         # Main dashboard application
│   ├── DisagreementAnalysis/
│   │   ├── calculator.py              # Disagreement scoring algorithms
│   │   ├── visualizer.py              # Disagreement visualizations
│   │   └── content.py                 # Dashboard content management
│   └── utils/
│       └── helpers.py                 # Utility functions
├── Modelling and HITL Dashboard/
│   ├── app.py                         # Main dashboard application
│   ├── RuleBased/
│   │   └── classifier.py              # Boolean logic rule classifier
│   ├── FuzzyMatching/
│   │   └── classifier.py              # Similarity-based classifier
│   ├── HITL/
│   │   └── analyzer.py                # Human-in-the-loop analysis
│   └── utils/
│       └── helper.py                  # Dashboard content generators
├── dataset_preparation_prompt.md      # Synthetic data generation specifications
├── Dataset_Raw.json                   # Generated synthetic dataset
├── Dataset_Clean.json                 # Processed dataset
├── data.csv                          # Analysis-ready format
└── README.md                         # Project documentation
```

## Data Flow Pipeline

1. **Generation**: Synthetic dataset creation using detailed annotator personas
2. **Cleaning**: Label standardization, null removal, duplicate elimination
3. **Conversion**: JSON to CSV transformation for analytics consumption
4. **Analysis**: Multi-dimensional agreement and quality assessment
5. **Classification**: Automated labeling with human feedback integration
6. **Optimization**: Continuous improvement through error pattern analysis

## Installation and Usage

### Prerequisites
```bash
python>=3.11
pandas>=1.5.0
plotly>=5.17.0
dash>=2.14.0
scikit-learn>=1.3.0
krippendorff>=0.6.0
gensim>=4.3.0
nltk>=3.8.0
```

### Local Development
```bash
# Clone repository
git clone https://github.com/your-org/Customer-Query-Analytics.git
cd Customer-Query-Analytics

# Install dependencies
pip install -r requirements.txt

# Run individual dashboards
cd "IAA Dashboard" && python app.py
cd "Data Quality Dashboard" && python app.py
cd "Modelling and HITL Dashboard" && python app.py
```

### Dashboard Customization
```bash
# Each dashboard includes custom HTML templates
IAA Dashboard/page.html                 # Inter-annotator agreement styling
Data Quality Dashboard/page.html        # Data quality assessment styling  
Modelling and HITL Dashboard/page.html  # HITL modeling interface styling

# Features include:
# - Dark/light theme support with client-side persistence
# - Custom CSS with gradient backgrounds and professional typography
# - Bootstrap integration with responsive design
# - Performance optimizations for large dataset rendering
```

### Data Analysis Workflows
```bash
# Business Analytics Pipeline
jupyter notebook "Business Analytics/Customer_Request_Topic_Profile.ipynb"  # Topic modeling
jupyter notebook "Business Analytics/Action_Verb_Analysis.ipynb"           # Intent analysis
jupyter notebook "Annotation Analytics/Distributions.ipynb"               # Distribution analysis
jupyter notebook "Annotation Analytics/Heirarchical_View.ipynb"          # Hierarchy visualization

# Advanced Analytics
jupyter notebook "Data Prepration/Pre-Analysis.ipynb"                     # Exploratory analysis
```

## API Documentation

### IAA Dashboard Components

#### IAAAgreementCalculator
- `calculate_alpha_with_ci(df, label_type, confidence_level, n_bootstrap)`: Krippendorff's Alpha with confidence intervals
- `calculate_pairwise_agreement_matrix(df, value_column)`: Inter-annotator agreement matrix
- `calculate_document_level_agreement(df, value_column)`: Sample-level agreement statistics

#### HierarchicalAnalysisCalculator
- `calculate_hierarchical_level_comparison(df)`: L1 vs L2 vs Full agreement comparison
- `calculate_hierarchical_consistency_metrics(results)`: Consistency ratio calculations

### Data Quality Dashboard Components

#### DisagreementAnalysisCalculator
- `calculate_document_disagreement_scores(df, value_column)`: Document-level disagreement metrics
- `calculate_label_confusion_matrix(df, value_column)`: Label pair confusion analysis

### HITL Dashboard Components

#### BaselineRuleClassifier
- `predict_single(text)`: Single text classification with confidence
- `_evaluate_boolean_rule(expression, text)`: Boolean logic evaluation

#### FuzzyMatchingClassifier
- `set_similarity_method(method)`: Character-level or semantic similarity selection
- `predict(texts)`: Batch text classification

#### HITLAnalyzer
- `compare_classifiers(rule_classifier, fuzzy_classifier)`: Performance comparison
- `identify_error_patterns(results, classifier_name)`: Systematic error detection
- `suggest_improvements(error_patterns, classifier_type)`: Improvement recommendations

## Configuration

### Annotator Personas
The system implements 7 distinct annotator profiles:
- **Tier 1 (Good)**: alex-001 (Idealist), beth-002 (Policy Lawyer), carlos-003 (Double-Checker)
- **Tier 2 (Medium)**: diane-004 (Box-Ticker), eric-005 (Surface Reader)
- **Tier 3 (Bad)**: fatima-006 (Keyword Spotter), george-007 (First-Hitter)

### Labeling Schema
```
Level 1 Categories:
├── Technical Issue
│   ├── Login Issue
│   ├── Feature Bug
│   └── Performance Issue
├── Billing
│   ├── Refund Request
│   ├── Unrecognized Charge
│   └── Invoice Inquiry
└── Account Management
    ├── Close Account
    ├── Update Personal Info
    └── Password Reset
```

## Performance Metrics

### Agreement Analysis
- **Krippendorff's Alpha**: Primary reliability measure with 95% confidence intervals
- **Pairwise Agreement**: Individual annotator consistency assessment
- **Hierarchical Consistency**: Cross-level agreement evaluation

### Classification Performance
- **Accuracy**: Overall prediction correctness
- **Precision/Recall/F1**: Class-specific performance metrics
- **Confidence Distribution**: Prediction certainty analysis

### Data Quality Indicators
- **Disagreement Score**: Sample-level annotation difficulty
- **Confusion Matrix**: Systematic classification errors
- **Text Complexity Correlation**: Length/difficulty relationship analysis

## Contributing

### Development Guidelines
1. Maintain consistent coding style with existing codebase
2. Include comprehensive docstrings for all functions
3. Add unit tests for new analytical components
4. Update documentation for interface changes
5. Follow semantic versioning for releases

### Code Quality Standards
- Type hints for all function parameters and returns
- Error handling with informative logging
- Performance optimization for large dataset processing
- Responsive dashboard design with accessibility considerations

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this platform in academic research, please cite:

```bibtex
@software{customer_query_analytics,
  title={Customer Query Analytics Platform},
  author={PruTech Development Team},
  year={2024},
  url={https://github.com/your-org/Customer-Query-Analytics}
}
```

## Support

For technical support, feature requests, or bug reports, please open an issue in the GitHub repository or contact the development team.
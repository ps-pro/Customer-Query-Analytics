# Customer Query Analytics Platform

A comprehensive analytics platform for customer support ticket classification, featuring multi-annotator agreement analysis, data quality assessment, and human-in-the-loop machine learning workflows with extensive business intelligence capabilities.

## Live Deployments

<div align="center">

[![IAA Dashboard](https://img.shields.io/badge/IAA%20Dashboard-Live-blue?style=for-the-badge&logo=plotly&logoColor=white)](https://ptc-iaa.plotly.app/)
[![Data Quality Dashboard](https://img.shields.io/badge/Data%20Quality%20Dashboard-Live-green?style=for-the-badge&logo=plotly&logoColor=white)](https://ptc-dq.plotly.app/)
[![HITL Dashboard](https://img.shields.io/badge/Classifier%20HITL%20Dashboard-Live-orange?style=for-the-badge&logo=plotly&logoColor=white)](https://ptc-chitl.plotly.app/)

</div>

## Overview

This platform provides end-to-end analytics for customer support ticket classification systems, from synthetic data generation through advanced disagreement analysis and automated classification improvement. The system simulates realistic annotation scenarios with multiple human annotators exhibiting different behavioral patterns and skill levels, supported by comprehensive business analytics and architectural documentation.

## Architecture

### Core Components

1. **Synthetic Data Generation Engine**: Realistic customer support ticket generation with multi-annotator simulation
2. **Business Analytics Suite**: Comprehensive text analysis including topic modeling, sentiment analysis, and NER
3. **Multi-Dashboard Analytics Platform**: Three specialized dashboard applications for different analytical perspectives
4. **Data Quality Assessment Framework**: Advanced disagreement analysis and gold-set optimization
5. **Human-in-the-Loop Classification System**: Automated classifiers with continuous improvement capabilities
6. **Architectural Decision Records**: Comprehensive documentation of design decisions and business use cases

### Technical Stack

- **Backend**: Python 3.11+
- **Web Framework**: Dash (Plotly)
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, gensim, spaCy
- **Statistics**: krippendorff (inter-annotator agreement)
- **Visualization**: Plotly, matplotlib, seaborn
- **Text Processing**: NLTK, TF-IDF, Named Entity Recognition
- **Natural Language Processing**: Sentiment analysis, topic modeling
- **Deployment**: Plotly Cloud

## Repository Structure

```
Customer-Query-Analytics/
├── .gitignore
├── dashboards_data.csv                 # Consolidated dashboard data
├── Dataset_Clean.json                  # Processed synthetic dataset
├── Dataset_Raw.json                    # Original generated dataset
├── LICENSE                             # MIT License
├── README.md                           # Project documentation
│
├── ADRs/                              # Architectural Decision Records
│   ├── pdf/                           # Compiled ADR documents
│   │   ├── ADR_01_Business_Use_Case_Selection.pdf
│   │   ├── ADR_02_Data_Generation_Strategy.pdf
│   │   ├── ADR_03_Label_Distribution_Analysis.pdf
│   │   ├── ADR_04_Hierarchical_Customer_Issue_Landscape.pdf
│   │   ├── ADR_05_Action_Verb_Analysis.pdf
│   │   ├── ADR_06_Common_Customer_Query_Pattern_Analysis.pdf
│   │   ├── ADR_07_Customer_Request_Topic_Profile_via_LDA_Modelling.pdf
│   │   ├── ADR_08_Named_Entity_Recognition_for_Customer_Communication_Analysis.pdf
│   │   ├── ADR_09_Risk_Alert_for_High_Stakes_Customer_Issues.pdf
│   │   └── ADR_10_Customer_Sentiment_Analysis_for_Emotional_Intelligence.pdf
│   └── tex/                           # LaTeX source files
│       ├── ADR-01_Business_Use_Case_Selection.tex
│       ├── ADR-02_Data_Generation_Strategy.tex
│       ├── ADR-03_Label_Distribution_Analysis.tex
│       ├── ADR-04_Hierarchical_Customer_Issue_Landscape.tex
│       ├── ADR-05_Action_Verb_Analysis.tex
│       ├── ADR-06_Common_Customer_Query_Pattern_Analysis.tex
│       ├── ADR-07_Customer_Request_Topic_Profile_via_LDA_Modelling.tex
│       ├── ADR-08_Named_Entity_Recognition_for_Customer_Communication_Analysis.tex
│       ├── ADR-09_Risk_Alert_for_High_Stakes_Customer_Issues.tex
│       └── ADR-10_Customer_Sentiment_Analysis_for_Emotional_Intelligence.tex
│
├── Annotation Analytics/              # Statistical distribution analysis
│   ├── Distributions.ipynb           # Label frequency and distribution analysis
│   └── Heirarchical_View.ipynb      # Interactive sunburst visualizations
│
├── Business Analytics/               # Advanced NLP and business intelligence
│   ├── Action_Verb_Analysis.ipynb   # Customer intent action extraction
│   ├── Common_Customer_Queries.ipynb # Query pattern identification
│   ├── Customer_Request_Topic_Profile.ipynb # LDA topic modeling
│   ├── Named_Entity_Recognition.ipynb # Entity extraction and analysis
│   ├── Risk_Alert.ipynb             # High-priority issue detection
│   └── Sentiment_Analysis.ipynb     # Customer emotion classification
│
├── Dashboards User Guides/          # Comprehensive user documentation
│   ├── pdf/                         # Compiled user guides
│   │   ├── Data_Quality_Dashboard.pdf
│   │   ├── IAA_Dashboard.pdf
│   │   └── Modelling_and_Human_in_the_Loop.pdf
│   └── tex/                         # LaTeX source files
│       ├── Data_Quality_Dashboard.tex
│       ├── IAA_Dashboard.tex
│       └── Modelling_and_Human_in_the_Loop.tex
│
├── Data Prepration/                  # Data processing pipeline
│   ├── Convertor.ipynb              # JSON to CSV transformation
│   ├── Data_Cleaning.ipynb          # Data standardization and cleaning
│   └── Pre-Analysis.ipynb           # Exploratory data analysis
│
├── Data Quality Dashboard/           # Data quality assessment application
│   ├── app.py                       # Main dashboard application
│   ├── data.csv                     # Dashboard data source
│   ├── page.html                    # Custom HTML components
│   ├── requirements.txt             # Python dependencies
│   ├── ConfusionAnalysis/           # Label confusion matrix analysis
│   │   ├── calculator.py            # Confusion metrics computation
│   │   ├── content.py               # Dashboard content generation
│   │   └── visualizer.py            # Confusion visualizations
│   ├── DisagreementAnalysis/        # Annotation disagreement analysis
│   │   ├── calculator.py            # Disagreement scoring algorithms
│   │   ├── content.py               # Content management
│   │   └── visualizer.py            # Disagreement visualizations
│   └── GoldSetAnalysis/             # Gold standard dataset optimization
│       ├── calculator.py            # Gold-set quality metrics
│       ├── content.py               # Interface components
│       └── visualizer.py            # Gold-set visualizations
│
├── IAA/                             # Inter-Annotator Agreement analysis
│   ├── Agreement.ipynb              # Comprehensive IAA analysis
│   ├── IAA_REPORT.pdf               # Detailed IAA report
│   └── iaa_report.tex               # LaTeX source for IAA report
│
├── IAA Dashboard/                   # Inter-Annotator Agreement dashboard
│   ├── app.py                       # Main dashboard application
│   ├── data.csv                     # Dashboard data source
│   ├── page.html                    # Custom HTML components
│   ├── requirements.txt             # Python dependencies
│   ├── AgreementAnalysis/           # Core agreement analysis
│   │   ├── calculator.py            # Krippendorff's Alpha implementation
│   │   ├── content.py               # Dashboard content generation
│   │   └── visualizer.py            # Agreement visualizations
│   ├── FrequencyAnalysis/           # Label frequency analysis
│   │   ├── calculator.py            # Frequency calculations
│   │   ├── content.py               # Frequency interface
│   │   └── visualizer.py            # Frequency charts
│   ├── HierarchicalAnalysis/        # Hierarchical label analysis
│   │   ├── calculator.py            # Hierarchical consistency metrics
│   │   ├── content.py               # Hierarchical interface
│   │   └── visualizer.py            # Hierarchical visualizations
│   └── utils/                       # Utility functions
│       ├── helpers.py               # Common utility functions
│       └── theme.py                 # Dashboard theming
│
└── Modelling and HITL Dashboard/    # Human-in-the-Loop classification
    ├── app.py                       # Main dashboard application
    ├── data.csv                     # Dashboard data source
    ├── page.html                    # Custom HTML components
    ├── requirements.txt             # Python dependencies
    ├── FuzzyMatching/               # Fuzzy matching classifier
    │   └── classifier.py            # Similarity-based classification
    ├── HITL/                        # Human-in-the-loop analysis
    │   └── analyzer.py              # HITL workflow management
    ├── RuleBased/                   # Rule-based classifier
    │   └── classifier.py            # Boolean logic classification
    └── utils/                       # Utility functions
        └── helper.py                # Dashboard content generators
```

## Key Features

### Architectural Decision Records (ADRs)
- **10 comprehensive ADRs** documenting design decisions and business rationales
- **Business use case analysis** with detailed justifications
- **Technical architecture documentation** for each analytical component
- **PDF and LaTeX formats** for professional documentation standards

### Advanced Business Analytics
- **Action Verb Analysis**: Customer intent extraction and action categorization
- **Common Query Pattern Analysis**: Frequent customer communication patterns
- **Topic Modeling**: Latent Dirichlet Allocation for thematic analysis
- **Named Entity Recognition**: Person, organization, and location extraction
- **Risk Alert System**: High-priority issue identification and escalation
- **Sentiment Analysis**: Customer emotion and satisfaction measurement

### Multi-Dimensional Data Quality Assessment
- **Disagreement Analysis**: Document-level annotation difficulty scoring
- **Confusion Analysis**: Label pair misclassification patterns
- **Gold-Set Analysis**: Optimal training data identification and refresh recommendations
- **Text Complexity Correlation**: Annotation difficulty prediction

### Comprehensive Inter-Annotator Agreement
- **Krippendorff's Alpha**: Industry-standard reliability measurement with confidence intervals
- **Frequency Analysis**: Label distribution and usage patterns
- **Hierarchical Analysis**: Multi-level consistency evaluation
- **Pairwise Comparison**: Individual annotator performance assessment

### Human-in-the-Loop Classification System
- **Rule-Based Classification**: Boolean logic with keyword matching
- **Fuzzy Matching**: Character-level and semantic similarity algorithms
- **Performance Comparison**: Automated vs. human classification analysis
- **Continuous Improvement**: Error pattern analysis and suggestion generation
- **CRUD Interfaces**: Real-time rule and example management

### Professional Documentation Suite
- **User Guides**: Comprehensive PDF manuals for each dashboard
- **Technical Documentation**: LaTeX source files for customization
- **API Documentation**: Detailed function and class specifications
- **Installation Instructions**: Complete setup and deployment guidance

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
spacy>=3.7.0
textblob>=0.17.0
```

### Local Development
```bash
# Clone repository
git clone https://github.com/your-org/Customer-Query-Analytics.git
cd Customer-Query-Analytics

# Install dependencies for each dashboard
cd "IAA Dashboard" && pip install -r requirements.txt
cd "../Data Quality Dashboard" && pip install -r requirements.txt
cd "../Modelling and HITL Dashboard" && pip install -r requirements.txt

# Run individual dashboards
cd "IAA Dashboard" && python app.py
cd "Data Quality Dashboard" && python app.py
cd "Modelling and HITL Dashboard" && python app.py
```

### Business Analytics Execution
```bash
# Execute comprehensive business analysis pipeline
jupyter notebook "Business Analytics/Customer_Request_Topic_Profile.ipynb"
jupyter notebook "Business Analytics/Sentiment_Analysis.ipynb"
jupyter notebook "Business Analytics/Named_Entity_Recognition.ipynb"
jupyter notebook "Business Analytics/Action_Verb_Analysis.ipynb"
jupyter notebook "Business Analytics/Risk_Alert.ipynb"
jupyter notebook "Business Analytics/Common_Customer_Queries.ipynb"
```

### Data Processing Pipeline
```bash
# Execute complete data preparation workflow
jupyter notebook "Data Prepration/Data_Cleaning.ipynb"
jupyter notebook "Data Prepration/Convertor.ipynb"
jupyter notebook "Data Prepration/Pre-Analysis.ipynb"
```

## API Documentation

### IAA Dashboard Components

#### IAAAgreementCalculator
- `calculate_alpha_with_ci(df, label_type, confidence_level, n_bootstrap=500)`: Krippendorff's Alpha with bootstrap confidence intervals
- `calculate_pairwise_agreement_matrix(df, value_column)`: Inter-annotator agreement matrix calculation
- `calculate_document_level_agreement(df, value_column)`: Sample-level agreement statistics

#### HierarchicalAnalysisCalculator
- `calculate_hierarchical_level_comparison(df)`: L1 vs L2 vs Full hierarchical agreement comparison
- `calculate_hierarchical_consistency_metrics(results)`: Cross-level consistency ratio calculations

#### FrequencyAnalysisCalculator
- `calculate_label_frequencies(df, label_type)`: Label distribution analysis
- `calculate_annotator_label_distribution(df)`: Per-annotator labeling patterns

### Data Quality Dashboard Components

#### DisagreementAnalysisCalculator
- `calculate_document_disagreement_scores(df, value_column)`: Document-level disagreement metrics
- `calculate_label_confusion_matrix(df, value_column)`: Label pair confusion analysis
- `identify_problematic_samples(df, threshold)`: High-disagreement sample identification

#### ConfusionAnalysisCalculator
- `calculate_per_annotator_confusion(df, annotator_id)`: Individual annotator confusion patterns
- `generate_confusion_heatmaps(df)`: Visual confusion matrix generation

#### GoldSetAnalysisCalculator
- `recommend_gold_set_refresh(df, quality_threshold)`: Training data optimization recommendations
- `calculate_gold_set_quality_metrics(df)`: Quality assessment of training examples

### HITL Dashboard Components

#### BaselineRuleClassifier
- `predict_single(text)`: Single text classification with confidence scoring
- `_evaluate_boolean_rule(expression, text)`: Boolean logic rule evaluation
- `add_rule(label, expression, weight, description)`: Dynamic rule addition

#### FuzzyMatchingClassifier
- `set_similarity_method(method)`: Character-level or semantic similarity selection
- `predict(texts)`: Batch text classification with confidence scores
- `add_training_example(label, text)`: Dynamic training example addition

#### HITLAnalyzer
- `compare_classifiers(rule_classifier, fuzzy_classifier)`: Comprehensive performance comparison
- `identify_error_patterns(results, classifier_name)`: Systematic error pattern detection
- `suggest_improvements(error_patterns, classifier_type)`: Automated improvement recommendations

## Configuration

### Annotator Personas
The system implements 7 distinct annotator behavioral profiles:
- **Tier 1 (High Quality)**: alex-001 (Idealist), beth-002 (Policy Lawyer), carlos-003 (Double-Checker)
- **Tier 2 (Medium Quality)**: diane-004 (Box-Ticker), eric-005 (Surface Reader)
- **Tier 3 (Low Quality)**: fatima-006 (Keyword Spotter), george-007 (First-Hitter)

### Hierarchical Labeling Schema
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

### Business Analytics Configuration
- **Topic Modeling**: 5-topic LDA with optimized hyperparameters
- **Sentiment Analysis**: Multi-class emotion classification (positive, negative, neutral, frustrated, urgent)
- **Named Entity Recognition**: Person, organization, location, and product entity extraction
- **Risk Detection**: Escalation keywords and urgency pattern identification

## Performance Metrics

### Agreement Analysis
- **Krippendorff's Alpha**: Primary reliability measure (α > 0.8 = excellent, 0.67-0.8 = good, 0.4-0.67 = moderate)
- **Pairwise Agreement**: Individual annotator consistency assessment
- **Hierarchical Consistency**: Cross-level agreement evaluation
- **Bootstrap Confidence Intervals**: Statistical significance testing with n=500 samples

### Classification Performance
- **Accuracy**: Overall prediction correctness (target >85%)
- **Precision/Recall/F1**: Class-specific performance metrics
- **Confidence Distribution**: Prediction certainty analysis
- **Error Pattern Analysis**: Systematic misclassification identification

### Data Quality Indicators
- **Disagreement Score**: Sample-level annotation difficulty (0.0-1.0 scale)
- **Confusion Matrix**: Systematic classification error patterns
- **Text Complexity Correlation**: Length and linguistic complexity impact on agreement
- **Gold-Set Quality**: Training data effectiveness measurement

### Business Intelligence Metrics
- **Topic Coherence**: LDA model quality assessment
- **Sentiment Distribution**: Customer emotion classification accuracy
- **Entity Recognition F1**: Named entity extraction performance
- **Risk Alert Precision**: Critical issue identification accuracy

## Architectural Decision Records

The platform includes comprehensive ADR documentation covering:

1. **Business Use Case Selection**: Rationale for customer support focus
2. **Data Generation Strategy**: Synthetic data creation methodology
3. **Label Distribution Analysis**: Category balance and representation
4. **Hierarchical Customer Issue Landscape**: Multi-level classification design
5. **Action Verb Analysis**: Customer intent extraction approach
6. **Common Customer Query Pattern Analysis**: Frequent pattern identification
7. **Customer Request Topic Profile via LDA**: Topic modeling implementation
8. **Named Entity Recognition**: Entity extraction strategy
9. **Risk Alert System**: High-priority issue detection methodology
10. **Customer Sentiment Analysis**: Emotional intelligence integration

## Contributing

### Development Guidelines
1. Maintain consistent coding style with existing codebase
2. Include comprehensive docstrings for all functions
3. Add unit tests for new analytical components
4. Update ADR documentation for architectural changes
5. Follow semantic versioning for releases
6. Maintain professional documentation standards

### Code Quality Standards
- Type hints for all function parameters and returns
- Error handling with informative logging
- Performance optimization for large dataset processing
- Responsive dashboard design with accessibility considerations
- Professional documentation with LaTeX source files

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this platform in academic research, please cite:

```bibtex
@software{customer_query_analytics,
  title={Customer Query Analytics Platform: A Comprehensive System for Multi-Annotator Agreement Analysis and Human-in-the-Loop Classification},
  author={Priyansh Singhal},
  year={2024},
  url={https://github.com/ps-pro/Customer-Query-Analytics},
  note={Advanced analytics platform with comprehensive business intelligence and architectural documentation}
}
```

## Support

For technical support, feature requests, or bug reports, please open an issue in the GitHub repository. Comprehensive user guides are available in the `Dashboards User Guides/` directory, and architectural decisions are documented in the `ADRs/` directory.
# MLOPS-WITH-ML-FLOW

A comprehensive MLOps project demonstrating MLflow integration for machine learning experiment tracking, model logging, and hyperparameter tuning.

## Overview

This repository showcases various MLflow features for MLOps practices, including:
- Manual logging of parameters, metrics, and artifacts
- Automatic logging using MLflow autolog
- Hyperparameter tuning with GridSearchCV and nested runs
- Integration with DagsHub for remote tracking

## Features

### Experiment Tracking
- **Manual Logging** (`src/file1.py`): Demonstrates explicit logging of parameters, metrics, confusion matrices, and models
- **Autolog** (`src/autolog.py`): Shows MLflow's automatic logging capabilities for scikit-learn models
- **Hyperparameter Tuning** (`src/hypertunning.py`): Illustrates logging of GridSearchCV results with nested runs
- **DagsHub Integration** (`src/file2.py`): Remote tracking using DagsHub platform

### Datasets Used
- Wine dataset (classification with 3 classes)
- Breast Cancer dataset (binary classification)

### Models
- RandomForestClassifier with configurable parameters (max_depth, n_estimators)

## Prerequisites

- Python 3.7+
- MLflow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- dagshub (for remote tracking)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/MLOPS-WITH-ML-FLOW.git
cd MLOPS-WITH-ML-FLOW
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local MLflow Tracking

1. Start MLflow tracking server:
```bash
mlflow ui
```

2. Run experiments:
```bash
python src/file1.py          # Manual logging
python src/autolog.py        # Autolog
python src/hypertunning.py   # Hyperparameter tuning
```

### DagsHub Integration

1. Set up DagsHub repository and get your credentials
2. Update the repository owner and name in `src/file2.py`
3. Run the DagsHub example:
```bash
python src/file2.py
```

## Project Structure

```
MLOPS-WITH-ML-FLOW/
├── src/
│   ├── autolog.py           # MLflow autolog demonstration
│   ├── file1.py             # Manual logging example
│   ├── file2.py             # DagsHub integration
│   └── hypertunning.py      # Hyperparameter tuning
├── mlruns/                  # Local MLflow runs
├── mlartifacts/             # MLflow artifacts
├── LICENSE
└── README.md
```

## Key Concepts Demonstrated

### MLflow Components
- **Experiments**: Organizing runs into logical groups
- **Runs**: Individual execution instances
- **Parameters**: Model hyperparameters
- **Metrics**: Performance measures
- **Artifacts**: Files, plots, and models
- **Models**: Serialized machine learning models
- **Tags**: Metadata for runs

### MLOps Best Practices
- Experiment tracking
- Model versioning
- Artifact management
- Remote tracking setup
- Hyperparameter optimization logging

## Results

The experiments demonstrate:
- Random Forest classification on wine dataset achieving ~95% accuracy
- Hyperparameter tuning on breast cancer dataset
- Confusion matrix visualization
- Model serialization and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

KHARVA JENISH H

## Acknowledgments

- MLflow documentation and community
- scikit-learn for machine learning algorithms
- DagsHub for remote MLflow tracking</content>
<parameter name="filePath">c:\Users\jenish\OneDrive\Desktop\MLOPS-WITH-ML-FLOW\README.md
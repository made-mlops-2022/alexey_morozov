Machine learning project for classification problem

# Installation
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
# Usage
## Training
~~~
python model_training.py -config_path=configs/config.yaml
~~~
## Predicting
~~~
python model_predicting.py -config_path=configs/config.yaml
~~~

## Evaluating:
~~~
python model_evaluate.py -config_path=configs/config.yaml
~~~

## Tests
~~~
python -m unittest tests/*
~~~

# Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Directory with raw data for training and saving predictions.
    │   ├── model          <- Directory for saving trained model.
    │   └── tmp            <- Temporal ridectory for logs output.
    │
    ├── configs            <- Configuration files for training and logging
    │
    ├── notebooks          <- Jupyter notebook with EDA and initial model selection,
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── source             <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── model          <- code to load/save model; load and preprocess data; getting classification 
    │   │
    │   └── params         <- code to all dataclass instances for project
    │
    └── model_training.py      <- script for running model training pipeline
    │
    ├── model_predicting.py    <- script for getting predictions on test dataset
    │
    └── model_evaluating.py    <- script for getting classification metrics for trained model

# Technical information
All configurations are stores as `dataclass` objects.

Main configuration file, that is used for training/predicting/evaluating: `configs/config.yaml`.

Logging config: `configs/logging.yaml`
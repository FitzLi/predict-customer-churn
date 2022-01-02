# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to identify credit card customers that are most likely to churn.

The repo contains two major files: `churn_library.py` and `churn_script_logging_and_tests.py`.

`churn_library.py` contains functions for DS processes such as EDA and Model Training.

`churn_script_logging_and_tests.py` contains tests for functions in `churn_library.py`.

## Running Files
Dependencies required to run files are stored in `requirements.txt` file.
To install dependencies, run:
```
$ conda create -n test_env python=3.8
$ conda activate test_env
$ pip install -r requirements.txt
```

To run `churn_library.py` file:
```
$ python churn_library.py
```
Afterwards, figures from EDA and model evaluation are stored in the folder "images", and trained models are stored in the folder "models"

To run `churn_script_logging_and_tests.py` file:
```
$ python churn_script_logging_and_tests.py
```
The resulting test logs are stored in the folder "logs"

## Files in the Repo
- data
    - bank_data.csv
- images
    - eda
        - heatmap_corr.png
        - hist_Churn.png
        - hist_TTC.png
    - results
        - feature_importance.png
        - ROC_curves.png
        - test_logistic_regression_results.png
        - test_random_forest_results.png
        - train_logistic_regression_results.png
        - train_random_forest_results.png
- logs
    - churn_library.log
- models
    - logistic_model.pkl
    - rfc_model.pkl
- churn_library.py
- churn_script_logging_and_tests.py
- LICENSE
- requirements.txt
- README.md

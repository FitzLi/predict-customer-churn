'''
The module contains clean-written libraries for predicting customer churn.

Author: Fitz (Chao Li)
Date: 19.12.2021
'''

import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from pandas.plotting import table
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

# Create folders to save outputs, if they don't exists already
EDA_DIR = './images/eda'
RESULTS_DIR = './images/results'
MODELS_DIR = './models'
for DIR in [EDA_DIR, RESULTS_DIR, MODELS_DIR]:
    Path(DIR).mkdir(parents=True, exist_ok=True)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df_data = pd.read_csv(pth)
        return df_data
    except FileNotFoundError as err:
        print('No file found in the input path')
        raise err


def perform_eda(df_data):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Univariate, categorical plot
    plt.figure(figsize=(20, 10))
    sns.histplot(df_data['Churn'])
    plt.title('Histogram of Churn data')
    plt.savefig(EDA_DIR + '/hist_Churn.png')

    # Univariate, quantitative plot
    plt.figure(figsize=(20, 10))
    sns.histplot(df_data['Total_Trans_Ct'])
    plt.title('Histogram of Total Trans Ct data')
    plt.savefig(EDA_DIR + '/hist_TTC.png')

    # Bivariate plot
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Correlation heatmap between variables')
    plt.savefig(EDA_DIR + '/heatmap_corr.png')


def encoder_helper(df_data, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
                      naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        lst = []
        groups = df_data.groupby(col)[response].mean()

        for val in df_data[col]:
            lst.append(groups.loc[val])

        col_new = '_'.join([col, response])
        df_data[col_new] = lst


def perform_feature_engineering(df_data, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for
                        naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    label = df_data[response]
    df_features = pd.DataFrame()
    df_features[keep_cols] = df_data[keep_cols]
    return train_test_split(df_features, label, test_size=0.3, random_state=42)


def convert_report_to_image(report, dataset, model_name):
    '''
    Convert the report to image and save it in images folder
    input:
            report: classification report
            dataset: the dataset (train or test) that the report is based on
            model_name: name of the prediction model
    output:
            None
    '''
    df_report = pd.DataFrame(report).transpose()
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table(ax, df_report)
    plt.savefig(RESULTS_DIR + '/%s_%s_results.png' % (dataset, model_name))


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    for y_preds, model_name in zip([y_test_preds_lr, y_test_preds_rf], [
            'logistic_regression', 'random_forest']):
        report = classification_report(y_test, y_preds, output_dict=True)
        convert_report_to_image(report, 'test', model_name)

    for y_preds, model_name in zip([y_train_preds_lr, y_train_preds_rf], [
            'logistic_regression', 'random_forest']):
        report = classification_report(y_train, y_preds, output_dict=True)
        convert_report_to_image(report, 'train', model_name)


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 12))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=45)
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr,
        y_test_preds_rf
    )

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(RESULTS_DIR + '/ROC_curves.png')

    joblib.dump(cv_rfc.best_estimator_, MODELS_DIR + '/rfc_model.pkl')
    joblib.dump(lrc, MODELS_DIR + '/logistic_model.pkl')


if __name__ == '__main__':
    df = import_data('data/bank_data.csv')
    RESPONSE = 'Churn'
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio']

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    perform_eda(df)
    encoder_helper(df, cat_columns, RESPONSE)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, RESPONSE)
    train_models(X_train, X_test, y_train, y_test)

    rfc_model = joblib.load(MODELS_DIR + '/rfc_model.pkl')
    feature_importance_plot(
        rfc_model,
        X_train,
        RESULTS_DIR + '/feature_importance.png')

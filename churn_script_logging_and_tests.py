'''
This module tests the functions from `churn_library`, and logs the results.
'''
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df_data


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        perform_eda(DF_DATA)
        logging.info('Testing perform_eda: SUCCESS')
    except Exception as err:
        logging.error(
            'Testing perform_eda: function fail to run due to error %s', err)
        raise err

    try:
        assert os.path.exists('./images/hist_Churn.png')
        assert os.path.exists('./images/dist_TTC.png')
        assert os.path.exists('./images/heatmap_corr.png')
    except AssertionError as err:
        logging.error(
            'Testing perform_eda: figures were not saved to images folder')
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        encoder_helper(DF_DATA, CAT_COLUMNS, RESPONSE)
        logging.info('Testing encoder_helper: SUCCESS')
    except Exception as err:
        logging.error(
            'Testing encoder_helper: function fail to run due to error %s', err)
        raise err

    try:
        for col in CAT_COLUMNS:
            col_new = '_'.join([col, RESPONSE])
            assert col_new in DF_DATA.columns
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper: features are not encoded or saved')
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            DF_DATA, RESPONSE)
        logging.info('Testing perform_feature_engineering: SUCCESS')
    except Exception as err:
        logging.error(
            'Testing perform_feature_engineering: function fail to run due to \
                        error %s', err)
        raise err

    try:
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: resulting data have incorrect dimensions')
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
        logging.info('Testing train_models: SUCCESS')
    except Exception as err:
        logging.error(
            'Testing train_models: function fail to run due to error %s', err)

    try:
        assert os.path.exists('./models/rfc_model.pkl')
        assert os.path.exists('./models/logistic_model.pkl')
    except AssertionError as err:
        logging.error(
            'Testing train_models: trained models are not found in \'./models\'')
        raise err


if __name__ == "__main__":
    DF_DATA = test_import(cls.import_data)
    RESPONSE = 'Churn'
    CAT_COLUMNS = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    DF_DATA[RESPONSE] = DF_DATA['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering)
    test_train_models(cls.train_models)

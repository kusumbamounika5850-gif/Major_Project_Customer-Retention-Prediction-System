import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import seaborn as sns
from seaborn import boxplot
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import logging
from logging_code import setup_logging
logger = setup_logging('categorical_to_numerical')
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def c_t_n(X_train_cat,X_test_cat):
    try:
        logger.info(f'Before applying categorical_to_numeric the X_train categorical columns are: {X_train_cat.columns} and shape is : {X_train_cat.shape}')
        logger.info(f'Before applying categorical_to_numeric the X_test categorical columns are: {X_test_cat.columns} and shape is : {X_test_cat.shape}')

        # CustomerId is dropped here
        X_train_cat = X_train_cat.drop(['customerID'], axis=1)
        X_test_cat = X_test_cat.drop(['customerID'], axis=1)

        # for the nominal columns applying One Hot encoder
        logger.info('The CustomerId column will be dropped from the categorical columns')
        logger.info(f'since The gender, partner and  Dependents are Nominal Columns. therefore OneHotEncoder will be applied ')
        #oneHotEncoder
        one_hot = OneHotEncoder(drop='first')
        one_hot.fit(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'PaymentMethod', 'telecom_partner']])
        val_train = one_hot.transform(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'PaymentMethod', 'telecom_partner']]).toarray()
        val_test = one_hot.transform(X_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'PaymentMethod', 'telecom_partner']]).toarray()

        t1 = pd.DataFrame(val_train)
        t2 = pd.DataFrame(val_test)

        t1.columns = one_hot.get_feature_names_out()
        t2.columns = one_hot.get_feature_names_out()

        X_train_cat.reset_index(drop=True, inplace=True)
        X_test_cat.reset_index(drop=True, inplace=True)

        t1.reset_index(drop=True, inplace=True)
        t2.reset_index(drop=True, inplace=True)

        X_train_cat = pd.concat([X_train_cat,t1],axis=1)
        X_test_cat = pd.concat([X_test_cat,t2],axis=1)

        X_train_cat = X_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'PaymentMethod', 'telecom_partner'], axis=1)
        X_test_cat = X_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'PaymentMethod', 'telecom_partner'], axis=1)

        logger.info(f'After applying One hot Encoding the X_train columns are: {X_train_cat.columns} and shape : {X_train_cat.shape}')
        logger.info(f'After applying One Hot Encoding the X_test columns are: {X_test_cat.columns} and shape : {X_test_cat.shape}')

        logger.info(f'Before applying Ordinal technique----------------------------------------------------------')

        # Ordinal Encoding

        ord = OrdinalEncoder()
        ord.fit(X_train_cat[['Contract']])
        res_train = ord.transform(X_train_cat[['Contract']])
        res_test = ord.transform(X_test_cat[['Contract']])

        p1 = pd.DataFrame(res_train)
        p2 = pd.DataFrame(res_test)

        p1.columns = ord.get_feature_names_out()+"_od"
        p2.columns = ord.get_feature_names_out()+"_od"

        p1.reset_index(drop=True, inplace=True)
        p2.reset_index(drop=True, inplace=True)

        X_train_cat = pd.concat([X_train_cat,p1],axis=1)
        X_test_cat = pd.concat([X_test_cat,p2],axis=1)

        X_train_cat = X_train_cat.drop(['Contract'], axis = 1)
        X_test_cat = X_test_cat.drop(['Contract'], axis = 1)

        logger.info(f'After applying Ordinal Encoding the X_train columns are: {X_train_cat.columns} and shape : {X_train_cat.shape}')
        logger.info(f'After applying Ordinal Encoding the X_test columns are: {X_test_cat.columns} and shape : {X_test_cat.shape}')
        return X_train_cat, X_test_cat


    except Exception as e:
        err_type, err_msg, err_line = sys.exc_info()
        logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

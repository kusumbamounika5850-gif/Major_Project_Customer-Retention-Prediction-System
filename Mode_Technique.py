import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from logging_code import setup_logging
logger = setup_logging('Mode_Technique')


def handle_missing_values(X_train,X_test):
    try:
        logger.info(f'Before handling Null values X_train column names and shape :{X_train.shape} \n :{X_train.columns}: {X_train.isnull().sum()}')
        logger.info(f'Before handling Null values X_test column names and shape :{X_test.shape} \n :{X_test.columns}: {X_test.isnull().sum()}')
        X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'])
        X_test['TotalCharges'] = pd.to_numeric(X_test['TotalCharges'])
        #Fill missing values with mode
        mode_value = X_train['TotalCharges'].mode()[0]

        X_train['TotalCharges_mode'] = X_train['TotalCharges'].fillna(mode_value)
        X_test['TotalCharges_mode'] = X_test['TotalCharges'].fillna(mode_value)
        X_train.drop('TotalCharges', axis=1, inplace=True)
        X_test.drop('TotalCharges', axis=1, inplace=True)

        logger.info(f'After handling Null values X_train column names and shape :{X_train.shape}  \n :{X_train.columns}: {X_train.isnull().sum()}')
        logger.info(f'After handling Null values X_test column names and shape :{X_test.shape}  \n :{X_test.columns}: {X_test.isnull().sum()}')
        return X_train, X_test
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no. {er_line.tb_lineno} due to: {er_msg}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import sys
import logging
from logging_code import setup_logging
logger = setup_logging('var_out')
from sklearn.preprocessing import QuantileTransformer

def vt_outliers(X_train_num, X_test_num):
    try:
        logger.info(f'Before Train column Name: {X_train_num.columns}')
        logger.info(f'Before Test column Name: {X_test_num.columns}')

        for i in X_train_num.columns:

            #  Quantile Transformation
            qt = QuantileTransformer(output_distribution='normal')

            X_train_num[i + '_qt'] = qt.fit_transform(X_train_num[[i]])
            X_test_num[i + '_qt'] = qt.transform(X_test_num[[i]])

            # Trimming (IQR)
            Q1 = X_train_num[i + '_qt'].quantile(0.25)
            Q3 = X_train_num[i + '_qt'].quantile(0.75)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            X_train_num[i + '_trim'] = np.where(
                X_train_num[i + '_qt'] > upper_limit, upper_limit,
                np.where(X_train_num[i + '_qt'] < lower_limit, lower_limit, X_train_num[i + '_qt'])
            )

            X_test_num[i + '_trim'] = np.where(
                X_test_num[i + '_qt'] > upper_limit, upper_limit,
                np.where(X_test_num[i + '_qt'] < lower_limit, lower_limit, X_test_num[i + '_qt'])
            )

        #  Drop unwanted columns
        X_train_num = X_train_num[[col for col in X_train_num.columns if col.endswith('_trim')]]
        X_test_num = X_test_num[[col for col in X_test_num.columns if col.endswith('_trim')]]

        logger.info(f"After Train Column Name : {X_train_num.columns}")
        logger.info(f"After Test Column Name : {X_test_num.columns}")

        return X_train_num, X_test_num

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no. {er_line.tb_lineno} due to: {er_msg}")




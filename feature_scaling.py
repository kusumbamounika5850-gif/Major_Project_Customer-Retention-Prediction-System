import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import logging_code
from logging_code import setup_logging
logger=setup_logging("feature_scaling")
from sklearn.preprocessing import StandardScaler # z_score
from all_models import common
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
import sys

def fs(X_train , y_train , X_test , y_test):
    try:
        logger.info(f"Training data independent size : {X_train.shape}")
        logger.info(f"Training data depdepent size : {y_train.shape}")
        logger.info(f"Testing data independent size : {X_test.shape}")
        logger.info(f"Testing data dependent size : {y_test.shape}")
        logger.info(f"before : {X_train.head(1)}")
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)
        with open('standar_scaler.pkl', 'wb') as f:
            pickle.dump(sc, f)

        logger.info(f"{X_train_sc}")
        common(X_train_sc, y_train, X_test_sc, y_test)

        xgb_reg = XGBClassifier(n_estimators=5)
        xgb_reg.fit(X_train_sc, y_train)

        logger.info(f"Test Accuracy : {accuracy_score(y_test, xgb_reg.predict(X_test_sc))}")
        logger.info(f"Test Confusion Matrix : {confusion_matrix(y_test, xgb_reg.predict(X_test_sc))}")
        logger.info(f"Classification report : {classification_report(y_test, xgb_reg.predict(X_test_sc))}")

        with open('Model.pkl', 'wb') as t:
            pickle.dump(xgb_reg, t)


    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
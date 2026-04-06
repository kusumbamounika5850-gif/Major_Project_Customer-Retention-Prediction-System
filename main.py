'''
In this file i am going to call all related functions for data cleaning and model development
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import os
import sys
from sklearn.model_selection import train_test_split
from Mode_Technique import handle_missing_values
from var_out import vt_outliers
from filter_methods import fm
from categorical_to_num import c_t_n
from imblearn.over_sampling import SMOTE
from feature_scaling import fs
import logging
from logging_code import setup_logging
logger = setup_logging("main")

class CHURN:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)#loading the data in to variable
            # Create new column without changing original
            self.df['telecom_partner'] = self.df['PaymentMethod'].map({
                'Electronic check': 'Reliance Jio',
                'Mailed check': 'Airtel',
                'Bank transfer (automatic)': 'VI-Idea',
                'Credit card (automatic)': 'BSNL'
            })
            logger.info(f'Total data_size :{self.df.shape}')
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', np.nan)
            self.df['TotalCharges']= pd.to_numeric(self.df['TotalCharges'])
            logger.info(f'Total null values :\n{self.df.isnull().sum()}')
            self.X = self.df.drop(['Churn'],axis=1)  # Independent variables
            self.y = self.df['Churn']  # Dependent variable
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            self.y_train = self.y_train.map({'Yes':1,'No':0}).astype(int)
            self.y_test = self.y_test.map({'Yes':1,'No':0}).astype(int)
            logger.info(f'Train data size :{len(self.X_train)}:{len(self.y_train)} \n{self.X_train.shape} : {self.y_train.shape}')
            logger.info(f'Test data size :{len(self.X_test)}:{len(self.y_test)} \n{self.X_test.shape} : {self.y_test.shape}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no. {er_line.tb_lineno} due to: {er_msg}")


    def missing_values(self):
        try:
            logger.info(f'Before Handling NUll values X_train Column names and shape :{self.X_train.shape} :{self.X_train.columns} \n{self.X_train.isnull().sum()}')
            logger.info(f'Before Handling null values X_test column names and shape : {self.X_test.shape} : {self.X_test.columns} \n{self.X_test.isnull().sum()}')

            for i in self.X_train.columns:
                if self.X_train[i].isnull().sum() > 0:
                    logger.info(f'{i} : {self.X_train[i].dtype}')
                    self.X_train['TotalCharges'] = pd.to_numeric(self.X_train['TotalCharges'])
                    self.X_test['TotalCharges'] = pd.to_numeric(self.X_test['TotalCharges'])
            self.X_train, self.X_test = handle_missing_values(self.X_train, self.X_test)
            logger.info(f'After Handling NUll values X_train Column names and shape :{self.X_train.shape} :{self.X_train.columns} \n{self.X_train.isnull().sum()}')
            logger.info(f'After Handling null values X_test column names and shape : {self.X_test.shape} : {self.X_test.columns} \n{self.X_test.isnull().sum()}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no. {er_line.tb_lineno} due to: {er_msg}")
    def data_separation(self):
        try:
            self.X_train_num_cols = self.X_train.select_dtypes(exclude = 'object')
            self.X_test_num_cols = self.X_test.select_dtypes(exclude = 'object')

            self.X_train_cat_cols = self.X_train.select_dtypes(include='object')
            self.X_test_cat_cols = self.X_test.select_dtypes(include='object')

            logger.info(f'X_train_num_cols = {self.X_train_num_cols.columns} : {self.X_train_num_cols.shape}')
            logger.info(f'X_test_num_cols = {self.X_test_num_cols.columns} : {self.X_test_num_cols.shape}')
            logger.info(f'X_train_cat_cols = {self.X_train_cat_cols.columns} : {self.X_train_cat_cols.shape}')
            logger.info(f'X_test_cat_cols = {self.X_test_cat_cols.columns} : {self.X_test_cat_cols.shape}')


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no. {er_line.tb_lineno} due to: {er_msg}")

    def variable_transformation(self):
        try:
            logger.info(f"Before Train Column Name : {self.X_train_num_cols.columns}")
            logger.info(f"Before Test Column Name : {self.X_test_num_cols.columns}")
            self.X_train_num_cols, self.X_test_num_cols = vt_outliers(self.X_train_num_cols, self.X_test_num_cols)
            logger.info(f"After Train Column Name : {self.X_train_num_cols.columns}")
            logger.info(f"After Test Column Name : {self.X_test_num_cols.columns}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no. {er_line.tb_lineno} due to: {er_msg}")

    def feature_selection(self):
        try:
            self.X_train_num_cols, self.X_test_num_cols = fm(self.X_train_num_cols, self.X_test_num_cols, self.y_train,
                                                             self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

    def cat_to_num(self):
        try:
            self.X_train_cat_cols, self.X_test_cat_cols = c_t_n(self.X_train_cat_cols, self.X_test_cat_cols)
            # combine the data
            self.X_train_num_cols.reset_index(drop=True, inplace=True)
            self.X_train_cat_cols.reset_index(drop=True, inplace=True)
            self.X_test_num_cols.reset_index(drop=True, inplace=True)
            self.X_test_cat_cols.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num_cols, self.X_train_cat_cols], axis=1)
            self.testing_data = pd.concat([self.X_test_num_cols, self.X_test_cat_cols], axis=1)

            logger.info(f"===========================================================")
            logger.info(f"Final Trainining data : {self.training_data.shape}")
            logger.info(f"{self.training_data.columns}")
            logger.info(f"{self.training_data.isnull().sum()}")

            logger.info(f"Final Testing data : {self.testing_data.shape}")
            logger.info(f"{self.testing_data.columns}")
            logger.info(f"{self.testing_data.isnull().sum()}")


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

    def data_balancing(self):
        try:
            logger.info(f"Number of Rows for yes Customer {1} : {sum(self.y_train==1)}")
            logger.info(f"Number of Rows for No Customer {0} : {sum(self.y_train == 0)}")
            logger.info(f"Training data size : {self.training_data.shape}")

            sm = SMOTE(random_state=42)

            self.training_data_bal,self.y_train_bal = sm.fit_resample(self.training_data , self.y_train)

            logger.info(f"Number of Rows for Good Customer {1} : {sum(self.y_train_bal == 1)}")
            logger.info(f"Number of Rows for Bad Customer {0} : {sum(self.y_train_bal == 0)}")
            logger.info(f"Training data size : {self.training_data_bal.shape}")

            fs(self.training_data_bal,self.y_train_bal,self.testing_data,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")









if __name__ == '__main__':
    try:
        obj = CHURN('Customer-Churn.csv')
        obj.missing_values()
        obj.data_separation()
        obj.variable_transformation()
        obj.feature_selection()
        obj.cat_to_num()
        obj.data_balancing()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no. {er_line.tb_lineno} due to: {er_msg}")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
import statsmodels.api as sm
import logging

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class LinearRegressionWithFeatureSelection:
    """This class is used build Linear regression models with only the relevant features.
    Author: Sesha Venkata Sriram Erramilli 😊

    references I referred to:-
    reference_1: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
    reference_2: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

    parameters
    ---------------------------------
    x_train: Training data frame containing the independent features.
    y_train: Training dataframe containing the dependent or target feature.
    x_test: Testing dataframe containing the independent features.
    y_test: Testing dataframe containing the dependent or target feature.
    """

    def __init__(self, x_train,
                 y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def backward_elimination_approach(self):
        """Description: This method builds a linear regression model on all the features and
        eliminates each one w.r.t. its p-value if it is above 0.05.
        Else it will be retained.
        Raises an exception if it fails.

        returns
        --------------------------------
        returns the linear regression model, its predictions of both the training and testing dataframes and the relevant features.
        """
        # Logging operation
        logging.info(
            'Entered the backward_elimination_approach method of the LinearRegressionWithFeatureSelection class')
        try:
            cols = list(self.x_train.columns)  # taking all the column names into a list
            pmax = 1
            while len(cols) > 0:
                p = []  # an empty list to be populated by the p values of all the variables
                x_1 = self.x_train[cols]
                x_1 = sm.add_constant(
                    x_1)  # adding constant, as the OLS method from statsmodels by default considers the regression line as a line passing through the origin.
                lr = sm.OLS(self.y_train,
                            x_1).fit()  # fitting using OLS or Ordinary least squares method from statsmodels
                p = pd.Series(lr.pvalues.values[1:], index=cols)  # storing in Pandas series
                pmax = max(p)  # getting the maximum p value
                feature_with_p_max = p.idxmax()  # finding its respective variable
                if pmax > 0.05:
                    cols.remove(feature_with_p_max)  # Feature elimination
                else:
                    break  # breaks the loop once we get all the relevant features
            selected_features_be = cols  # All the relevant columns in a list
            print()
            print("Features selected by the Backward elimination method in Linear regression are ",
                  selected_features_be)
            print()

            x_train_be = self.x_train[selected_features_be]  # considering only the relevant features
            x_test_be = self.x_test[selected_features_be]  # considering only the relevant features

            lr = LinearRegression()  # instantiating linear regression model from LinearRegression of sci-kit learn

            lr.fit(x_train_be, self.y_train)  # fitting

            y_pred_train_be = lr.predict(x_train_be)  # predictions on train data
            y_pred_test_be = lr.predict(x_test_be)  # predictions on test data

            # logging operation
            logging.info('Linear regression model built successfully using Backward Elimination approach.')

            logging.info(
                'Exited the backward_elimination_approach method of the LinearRegressionWithFeatureSelection class')

            return lr, x_train_be, y_pred_train_be, x_test_be, y_pred_test_be, selected_features_be

        except Exception as e:
            # logging operation
            logging.error(
                'Exception occurred in backward_elimination_approach method of the LinearRegressionWithFeatureSelection class. Exception '
                'message:' + str(e))
            logging.info(
                'Backward elimination method unsuccessful. Exited the backward_elimination_approach method of the '
                'LinearRegressionWithFeatureSelection class ')

    def rfe_approach(self):
        """Description: This method uses Recursive Feature Elimination algorithm of sci-kit learn, which ultimately
         selects the most relevant features of the given dataset.
         Raises an exception if it fails.

        returns
        --------------------------------
        returns the linear regression model, its predictions on both the training and testing dataframes and the relevant
        features.
         """
        logging.info(
            'Entered the rfe_approach method of the LinearRegressionWithFeatureSelection class')  # logging operation

        try:
            features = self.x_train.columns.tolist()  # taking all the column names into a list
            nof_list = np.arange(1, len(features) + 1)
            high_score = 0  # variable which stores the highest score among all the variables

            nof = 0  # variable to store the number of optimum features
            score_list = []  # scores of all the variables

            for n in range(len(nof_list)):
                lr = LinearRegression()  # instantiating LinearRegression object, imported from sci-kit learn
                rfe = RFE(lr, nof_list[n])  # instantiating RFE, imported from sci-kit learn
                x_train_rfe = rfe.fit_transform(self.x_train, self.y_train)  # fitting RFE on the train data
                x_test_rfe = rfe.transform(self.x_test)  # transforming the test data using RFE
                lr.fit(x_train_rfe, self.y_train)  # fitting the LinearRegression model
                score = lr.score(x_test_rfe, self.y_test)  # collecting scores and appending to list
                score_list.append(score)
                if score > high_score:
                    high_score = score
                    nof = nof_list[n]

            lr = LinearRegression()  # initiating RFE with optimum features only
            rfe = RFE(lr, nof)
            x_train_rfe = rfe.fit_transform(self.x_train, self.y_train)  # fitting RFE on the train data
            x_test_rfe = rfe.transform(self.x_test)  # transforming the test data using RFE

            lr.fit(x_train_rfe, self.y_train)  # Building Linear regression
            temp = pd.Series(rfe.support_, index=features)  # storing rfe supported columns into a pandas series
            selected_features_rfe = temp[temp == True].index
            print('Features selected by the RFE method in Linear regression are',
                  selected_features_rfe)  # displaying the rfe selected features
            print()

            y_pred_train_rfe = lr.predict(x_train_rfe)  # predictions on the train data
            y_pred_test_rfe = lr.predict(x_test_rfe)  # predictions on the test data

            logging.info('Linear regression model built successfully using RFE approach. ')  # logging operation

            # logging operation
            logging.info('Exited the rfe_approach method of the LinearRegressionWithFeatureSelection class')

            return lr, x_train_rfe, y_pred_train_rfe, x_test_rfe, y_pred_test_rfe, selected_features_rfe

        except Exception as e:
            logging.error(
                'Exception occurred in rfe_approach method of the LinearRegressionWithFeatureSelection class. Exception '
                'message:' + str(e))  # logging operation
            logging.info('RFE method unsuccessful. Exited the rfe_approach method of the '
                         'LinearRegressionWithFeatureSelection class ')  # logging operation


class Lasso:
    """This class is used to train the models using Linear regression with Lasso regularization or L1 regularization.
    Author: Sesha Venkata Sriram Erramilli 😊

    parameters
    ---------------------------------
    x_train: Training data frame containing the independent features.
    y_train: Training dataframe containing the dependent or target feature.
    x_test: Testing dataframe containing the independent features.
    y_test: Testing dataframe containing the dependent or target feature.

    """

    def __init__(self, x_train,
                 y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def lassocv(self):
        """Description: This method uses LassoCV algorithm imported from the sci-kit learn library to build a regression model.
        It does a cross validation with various learning rates, ultimately finds the most relevant features
        and builds a model, i.e., redundant features will be eliminated.
        Raises an exception if it fails

        returns
        ---------------------------------------
        returns the linear regression model,its predictions on both the training and testing dataframes with the features
        selected by the LassoCV.
        """
        logging.info('Entered the lassocv method of the Lasso class.')  # logging operation

        try:

            ls = LassoCV()  # Instantiating LassoCV
            ls.fit(self.x_train, self.y_train)  # fitting on the training data

            coef = pd.Series(ls.coef_, index=self.x_train.columns)  # feature importance by LassoCV

            imp_coef = coef.sort_values(ascending=False)

            print('Feature importance by the LassoCV are: ', imp_coef)
            print()

            y_pred_train_lasso = ls.predict(self.x_train)  # predictions on the train data
            y_pred_test_lasso = ls.predict(self.x_test)  # predictions on the test data

            # logging operation
            logging.info('Linear regression model built successfully using LassoCV approach. ')

            logging.info('Exited the lassocv method of the Lasso class.')  # logging operation

            return ls, self.x_train, y_pred_train_lasso, self.x_test, y_pred_test_lasso

        except Exception as e:
            logging.error('Exception occurred in lassocv method of the Lasso class. Exception '
                          'message:' + str(e))  # logging operation
            logging.info('lassocv method unsuccessful. Exited the lassocv method of the '
                         'Lasso class ')  # logging operation

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import warnings
import logging

warnings.filterwarnings('ignore')  # ignore warnings

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class TreeModelsReg:
    """This class is used to build regression models using different tree and ensemble techniques.
    Author: Sesha Venkata Sriram Erramilli 😊
    References I referred:
    reference 1 - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    reference 2 - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decision%20tree%20regressor#sklearn.tree.DecisionTreeRegressor
    reference 3 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest%20regressor#sklearn.ensemble.RandomForestRegressor
    reference 4 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html?highlight=adaboost%20regressor#sklearn.ensemble.AdaBoostRegressor
    reference 5 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html?highlight=gradient%20boost%20regressor#sklearn.ensemble.GradientBoostingRegressor
    reference 6 - https://xgboost.readthedocs.io/en/latest/get_started.html
    reference 7 - https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

    parameters:
    --------------------------------
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

    def decision_tree_regressor(self):
        """Description: This method builds a model using DecisionTreeRegressor algorithm imported from the sci-kit learn,
        by implementing cross validation technique to choose the best estimator with the best hyper parameters.
        Raises an exception if it fails

        returns
        --------------------------------
        The Decision tree regressor model and prints the importance of each feature
        """

        try:
            dt = DecisionTreeRegressor()  # instantiating DecisionTreeRegressor object

            params = {'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                      'max_depth': [2, 5, 10, 20],
                      'min_samples_split': [2, 4, 8, 12],
                      'min_samples_leaf': [2, 4, 6, 8, 10]}  # parameter grid

            rcv = RandomizedSearchCV(estimator=dt, param_distributions=params, n_iter=10,
                                     scoring='r2', cv=10, verbose=2,
                                     random_state=42, n_jobs=-1,
                                     return_train_score=True)  # Randomized search cross validation object imported from sci-kit learn

            print('Cross validation process for Decision tree regressor')
            rcv.fit(self.x_train, self.y_train)  # fitting on the train data
            print()

            print('The best estimator for Decision tree regressor is ',
                  rcv.best_estimator_)  # display the best estimator

            dt = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final decision tree regressor.

            dt.fit(self.x_train, self.y_train)  # fitting on the train data.

            # Feature importance by the Decision tree regressor
            dt_feature_imp = pd.DataFrame(dt.feature_importances_, index=self.x_train.columns,
                                          columns=['Feature_importance'])
            dt_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the Decision tree regressor: ', dt_feature_imp)
            print()

            logging.info(
                "Successfully built a model using Decision tree regressor with the best hyper parameters")  # logging operation

            logging.info('Exited the decision_tree_regressor method of the TreeModelsReg class')

            return dt

        except Exception as e:

            logging.error('Exception occurred in decision_tree_regressor method of the TreeModelsReg class. Exception '
                          'message:' + str(e))  # logging operation
            logging.info(
                'decision_tree_regressor method unsuccessful. Exited the decision_tree_regressor method of the '
                'TreeModelsReg class ')  # logging operation

    def decision_tree_regressor_post_pruning(self):
        """Description: This method implements the post pruning technique to tackle over-fitting in the decision tree regressor.
        While doing so, we found out the optimum cost complexity pruning or ccp_alpha parameter as 0.8 in the
        'EDA + Model building.ipynb' jupyter notebook using visualization.
         Raises an exception if it fails

         returns
         -------------------------------
         The Decision tree regressor model post pruning
         """

        try:
            dt = DecisionTreeRegressor(random_state=42, ccp_alpha=0.8)  # instantiating the DecisionTreeRegressor object

            dt.fit(self.x_train, self.y_train)  # fitting the model

            logging.info(
                "Successfully built a model using Decision tree regressor with post pruning technique")  # logging operation

            logging.info('Exited the "decision_tree_regressor_post_pruning" method of the TreeModelsReg class')

            return dt
        except Exception as e:

            logging.error(
                'Exception occurred in "decision_tree_regressor_post_pruning" method of the TreeModelsReg class. Exception '
                'message:' + str(e))  # logging operation

            logging.info(
                '"decision_tree_regressor_post_pruning" method unsuccessful. Exited the random_forest_regressor method of the '
                'TreeModelsReg class ')  # logging operation

    def random_forest_regressor(self):
        """Description: This method builds a model using RandomForestRegressor algorithm, a type of ensemble technique
        imported from sci-kit learn library. It uses cross validation technique and chooses the best estimator with the
        best hyper parameters.
        Raises an exception if it fails
        returns
        --------------------------------
        The Random forest regressor model and prints the importance of each feature
        """

        try:
            rf = RandomForestRegressor()  # instantiating the RandomForestRegressor object

            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'criterion': ['mse', 'mae'],
                      'max_depth': [2, 5, 10, 20],
                      'min_samples_split': [2, 4, 8, 12],
                      'min_samples_leaf': [2, 4, 6, 8, 10],
                      'oob_score': [True]}  # parameter grid

            rcv = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=10, scoring='r2', cv=10,
                                     verbose=5,
                                     random_state=42, n_jobs=-1,
                                     return_train_score=True)  # instantiating RandomizedSearchCV

            print('Cross validation process for Random forest regressor')

            rcv.fit(self.x_train, self.y_train)  # Fitting on the train data
            print()

            print('The best estimator for the Random forest regressor is',
                  rcv.best_estimator_)  # displaying the best estimator

            rf = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final random forest regressor.

            rf.fit(self.x_train, self.y_train)  # fitting on the train data

            # Feature importance by the Random Forest regressor
            rf_feature_imp = pd.DataFrame(rf.feature_importances_, index=self.x_train.columns,
                                          columns=['Feature_importance'])
            rf_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the Random Forest regressor: ', rf_feature_imp)
            print()

            logging.info("Successfully built a model using Random Forest regressor ")  # logging operation

            logging.info('Exited the random_forest_regressor method of the TreeModelsReg class')  # logging operation

            return rf

        except Exception as e:

            logging.error('Exception occurred in random_forest_regressor method of the TreeModelsReg class. Exception '
                          'message:' + str(e))  # logging operation

            logging.info(
                'random_forest_regressor method unsuccessful. Exited the random_forest_regressor method of the '
                'TreeModelsReg class ')  # logging operation

    def adaboost_regressor(self):
        """Description: This method builds a model using AdaBoostRegressor algorithm, a type of ensemble technique imported
        from the sci-kit learn library. It uses cross validation technique and chooses the best estimator with the
        best hyper parameters.
        Raises an exception if it fails

        returns
        ----------------------------------
        The Adaboost regressor model and prints the importance of each feature
        """

        try:
            adb = AdaBoostRegressor()  # instantiating the AdaBoostRegressor object

            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                      'loss': ['linear', 'square', 'exponential']
                      }  # parameter grid

            rcv = RandomizedSearchCV(estimator=adb, param_distributions=params, n_iter=10, scoring='r2',
                                     n_jobs=-1, cv=10, verbose=5, random_state=42, return_train_score=True)

            # instantiating RandomizedSearchCV

            print('Cross validation process for the Adaboost regressor')

            rcv.fit(self.x_train, self.y_train)  # fitting on the train data
            print()

            print('The best estimator for the Adaboost regressor is',
                  rcv.best_estimator_)  # displaying the best estimator

            adb = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final Adaboost regressor.

            adb.fit(self.x_train, self.y_train)  # fitting on the train data

            # Feature importance by the Adaboost regressor
            adb_feature_imp = pd.DataFrame(adb.feature_importances_, index=self.x_train.columns,
                                           columns=['Feature_importance'])
            adb_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the Adaboost regressor: ', adb_feature_imp)
            print()

            logging.info("Successfully built a model using Adaboost regressor ")  # logging operation

            logging.info('Exited the adaboost_regressor method of the TreeModelsReg class')  # logging operation

            return adb

        except Exception as e:

            logging.error('Exception occurred in adaboost_regressor method of the TreeModelsReg class. Exception '
                          'message:' + str(e))  # logging operation

            logging.info('adaboost_regressor method unsuccessful. Exited the adaboost_regressor method of the '
                         'TreeModelsReg class ')  # logging operation

    def gradientboosting_regressor(self):
        """Description: This method builds a model using GradientBoostingRegressor algorithm, a type of ensemble technique imported
        from the sci-kit learn library. It uses cross validation technique and chooses the best estimator with the
        best hyper parameters.
        Raises an exception if it fails

        returns
        -------------------------------------
        The Gradientboosting regressor model and prints the importance of each feature
        """

        try:

            gbr = GradientBoostingRegressor()  # instantiating the GradientBoostingRegressor object.

            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'learning_rate': [0.1, 0.2, 0.5, 0.8, 1],
                      'loss': ['lr', 'lad', 'huber'],
                      'subsample': [0.001, 0.009, 0.01, 0.09, 0.1, 0.4, 0.9, 1],
                      'criterion': ['friedman_mse', 'mse'],
                      'min_samples_split': [2, 4, 8, 10],
                      'min_samples_leaf': [1, 10, 20, 50]
                      }  # Parameter grid

            rcv = RandomizedSearchCV(estimator=gbr, param_distributions=params, n_iter=10, scoring='r2', n_jobs=-1,
                                     cv=10, verbose=5, random_state=42,
                                     return_train_score=True)  # instantiating RandomizedSearchCV

            print('Cross validation process for the Gradient Boosting Regressor')

            rcv.fit(self.x_train, self.y_train)  # Fitting on the train data
            print()

            print('The best estimator for the GradientBoosting regressor is',
                  rcv.best_estimator_)  # displaying the best estimator

            gbr = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final Gradient Boosting regressor.

            gbr.fit(self.x_train, self.y_train)  # fitting on the train data

            # Feature importance by the Gradient Boosting regressor
            gbr_feature_imp = pd.DataFrame(gbr.feature_importances_, index=self.x_train.columns,
                                           columns=['Feature_importance'])
            gbr_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the Gradient boosting regressor: ', gbr_feature_imp)
            print()

            logging.info("Successfully built a model using Gradient Boosting regressor ")

            logging.info('Exited the gradientboosting_regressor method of the TreeModelsReg class')

            return gbr

        except Exception as e:

            logging.error(
                'Exception occurred in gradientboosting_regressor method of the TreeModelsReg class. Exception '
                'message:' + str(e))  # logging operation

            logging.info('gradientboosting_regressor method unsuccessful. Exited the gradientboosting_regressor method '
                         'of the TreeModelsReg class ')  # logging operation

    def xgb_regressor(self):
        """Description: This method builds a model using XGBRegressor algorithm, a type of ensemble technique imported from the
        xgboost library.It uses cross validation technique and chooses the best estimator with the
        best hyper parameters.
        Raises an exception if it fails

        returns
        -----------------------------
        The XGBoost regressor model and prints the importance of each feature
        """

        try:
            xgbr = XGBRegressor()  # instantiating the XGBRegressor object

            params = {
                'learning_rate': [0.1, 0.2, 0.5, 0.8, 1],
                'max_depth': [2, 3, 4, 5, 6, 7, 8, 10],
                'subsample': [0.001, 0.009, 0.01, 0.09, 0.1, 0.4, 0.9, 1],
                'min_child_weight': [1, 2, 4, 5, 8],
                'gamma': [0.0, 0.1, 0.2, 0.3],
                'colsample_bytree': [0.3, 0.5, 0.7, 1.0, 1.4],
                'reg_alpha': [0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1, 4, 8, 10, 50, 100],
                'reg_lambda': [1, 4, 5, 10, 20, 50, 100, 200, 500, 800, 1000]
            }  # Parameter grid

            rcv = RandomizedSearchCV(estimator=xgbr, param_distributions=params, n_iter=10,
                                     scoring='r2', cv=10, verbose=2,
                                     random_state=42, n_jobs=-1,
                                     return_train_score=True)  # instantiating RandomizedSearchCV
            print('Cross validation process for the XGBoost regressor')

            rcv.fit(self.x_train, self.y_train)  # Fitting on the train data
            print()

            print('The best estimator for the XGBoost regressor is',
                  rcv.best_estimator_)  # displaying the best estimator

            xgbr = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final XGBoosting regressor.

            xgbr.fit(self.x_train, self.y_train)  # fitting on the train data

            # Feature importance by the XGBoosting regressor
            xgbr_feature_imp = pd.DataFrame(xgbr.feature_importances_, index=self.x_train.columns,
                                            columns=['Feature_importance'])
            xgbr_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the XGBoost regressor: ', xgbr_feature_imp)
            print()

            logging.info("Successfully built a model using XGBoost regressor ")

            logging.info('Exited the xgb_regressor method of the TreeModelsReg class')

            return xgbr

        except Exception as e:

            logging.error('Exception occurred in xgb_regressor method of the TreeModelsReg class. Exception '
                          'message:' + str(e))  # logging operation

            logging.info('xgb_regressor method unsuccessful. Exited the xgb_regressor method of the '
                         'TreeModelsReg class ')  # logging operation

    def model_predict(self, model, X):
        """Description: This method makes predictions using the given model
        raises an exception if it fails

        parameters
        ----------------------------------
        model:- model to be used for making predictions
        X = A pandas dataframe with independent features

        returns
        ----------------------------------
        The predictions of the target variable.
        """

        try:

            pred = model.predict(X)

            return pred

        except Exception as e:

            logging.error('Exception occurred in "model_predict" method of the TreeModelsReg class. Exception '
                          'message:' + str(e))  # logging operation

            logging.info('"model_predict" method unsuccessful. Exited the "model_predict" method of the '
                         'TreeModelsReg class ')  # logging operation

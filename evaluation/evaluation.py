from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import logging

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class Metrics:
    """This class is used to evaluate the models by returning their performance metrics.
    Author: Sesha Venkata Sriram Erramilli 😊
    """

    def __init__(self):
        pass

    def r2_score(self, y_true, y_pred):
        """Description: This method calculates the r2_score of the model, which tells us how much variance our model
        can explain on the given data. This method uses r2_score method imported from the sci-kit learn.
        Raises an exception if it fails

        parameters
        --------------------------------
        y_true: Dataframe containing the actual values of the dependent or the target feature
        y_pred: Dataframe containing the predicted values of the dependent or the target feature

        returns
        --------------------------------
        r2 score of the model
        """
        logging.info('Entered the r2_score method of the Metrics class')  # logging operation

        try:
            score = r2_score(y_true, y_pred)
            logging.info('Calculated r2_score. Exited the r2_score method of the Metrics class')  # logging operation

            return score

        except Exception as e:
            logging.error('Exception occurred in r2_score method of the Metrics class. Exception '
                          'message:' + str(e))  # logging operation
            logging.info(
                'r2_score method unsuccessful. Exited the r2_score method of the Metrics class ')  # logging operation

    def adj_r2_score(self, x, y_true, y_pred):
        """Description: Calculates the adjusted r2_score of the model.
        Raises an exception if it fails.

        parameters
        ---------------------------------
        x: Dataframe containing the independent features
        y_true: Dataframe containing the actual values of the dependent or the target feature
        y_pred: Dataframe containing the predicted values of the dependent or the target feature

        returns
        ---------------------------------
        adjusted r2 score of the model

        """
        logging.info('Entered the adj_r2_score method of the Metrics class')  # logging operation

        try:
            r2 = r2_score(y_true, y_pred)
            n = x.shape[0]
            p = x.shape[1]
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            logging.info(
                'Calculated adj_r2_score. Exited the adj_r2_score method of the Metrics class ')  # logging operation

            return adj_r2

        except Exception as e:
            logging.error('Exception occurred in adj_r2_score method of the Metrics class. Exception '
                          'message:' + str(e))  # logging operation
            logging.info(
                'adj_r2_score method unsuccessful. Exited the adj_r2_score method of the Metrics class ')  # logging operation

    def rmse_score(self, y_true, y_pred):
        """Description: Calculates the root mean square error.
        Raises an exception if it fails

        parameters
        --------------------------------
        y_true: Dataframe containing the actual values of the dependent or the target feature
        y_pred: Dataframe containing the predicted values of the dependent or the target feature

        returns
        --------------------------------
        root mean square error of the model
        """
        logging.info('Entered the rmse_score method of the Metrics class')  # logging operation

        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(
                'Calculated rmse_score. Exited the rmse_score method of the Metrics class')  # logging operation)

            return rmse

        except Exception as e:
            logging.error('Exception occurred in rmse_score method of the Metrics class. Exception '
                          'message:' + str(e))  # logging operation
            logging.info('rmse_score method unsuccessful. Exited the rmse_score method of the Metrics class ')  # logging operation

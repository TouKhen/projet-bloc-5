import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


class Models:
    def __init__(self, df: pd.DataFrame, df_dummies: pd.DataFrame) -> None:
        """
        Initializes a class instance for managing and processing data using provided
        dataframes for operations. The class is designed to handle input datasets and
        their dummified versions, which are later used for machine learning.

        :param df: DataFrame containing the main dataset.
        :param df_dummies: DataFrame containing the dummified version of the dataset.

        """
        self.df = df
        self.df_dummies = df_dummies


    def logistic_regression(self, params=None) -> linear_model.LogisticRegression:
        """
        Perform logistic regression on the dataset, training and testing the model
        with the provided or default parameters.

        :param params: Optional; a dictionary of parameters to be passed to the
                       LogisticRegression model.
                       If None, default parameters are used.
                       Defaults to None.
        :type params: dict, optional
        :return: Trained LogisticRegression model.
        :rtype: linear_model.LogisticRegression
        """
        # Drop Churn row for the logistic regression.
        self.X_clean = self.df_dummies.drop(columns='Churn')
        self.y_clean = self.df['Churn'][self.X_clean.index]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_clean, self.y_clean, test_size=0.3, random_state=42)

        logreg = linear_model.LogisticRegression(max_iter=10000, **params) if params else linear_model.LogisticRegression(max_iter=10000)
        logreg.fit(self.X_train, self.y_train)

        return logreg
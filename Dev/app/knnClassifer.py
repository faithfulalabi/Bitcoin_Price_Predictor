# write all your imports

# create you class with a def init that creates the model


# create function that creates the model


# create function that reads the train data from Postgres

# create prep df function that takes care of spliting X and Y
# Function that split train test split

# Function that scales the data
# Function fixes class imbalance (probably won't use)
# create a function that takes in current model f1 report and compares it to newly trained model f1 report and returns tru
# function that trains and fits the model evaluate on test set returns F1 report,
# function that takes in a trained model and converts it to a pickle file and saves it as a pickle file
# function that will write model name and F1 report to table in postgre
# plot confusion matrix saves it, plot feature importance and saves it

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle


# imports for database
import psycopg2 as pg
from db_connection import connect
from timebasedcv import TimeBasedCV


class Classifier:
    def __init__(self, db_connection, previous_model_score, split_date="2022-03-01"):
        self.db_connection = db_connection
        self.previous_model_score = previous_model_score
        self.split_date = split_date
        self.model = self.create_knn()

    # create knn model
    @staticmethod
    def create_knn():
        return KNeighborsClassifier(n_neighbors=3)

    # read train data from postgre
    @staticmethod
    def read_train_data(db_conn):
        if db_conn is not None:

            con = db_conn
            # read data from db
            sql = "SELECT * FROM processed_bitcoin"
            df = pd.read_sql_query(sql=sql, con=con())
        else:
            print("Error getting DB Connection param while reading train data")
        return df

    # split train and test data by specified split date
    @staticmethod
    def train_test_data(df, split_date):

        if type(split_date) == str:
            split_date = datetime.strptime(split_date, "%Y-%m-%d").date()

        train_df = df.loc[df.prediction_date < split_date]
        test_df = df.loc[df.prediction_date >= split_date]

        return train_df, test_df

    # split train and test data by number of days
    @staticmethod
    def timebased_train_test_split(train_df, time_based_cv):

        tscv = time_based_cv(train_period=30, test_period=7, freq="days")
        index_output = tscv.split(train_df, date_column="prediction_date")
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        for train_index, test_index in index_output:
            X_train = X_train.append(
                train_df.loc[train_index].drop("prediction_date", axis=1)
            )
            X_test = X_test.append(
                train_df.loc[test_index].drop("prediction_date", axis=1)
            )
        # create y_train & drop output from trainset
        y_train = X_train["signal"]
        X_train.drop("signal", inplace=True, axis=1)
        # create y_test and drop output from testset
        y_test = X_test["signal"]
        X_test.drop("signal", inplace=True, axis=1)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def scale_data(X_train, X_test):
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        return scaled_X_train, scaled_X_test, scaler

    def train(self):
        # read in the data
        df = self.read_train_data(self.db_connection)
        # split train and test data
        train_df, test_df = self.train_test_data(df, self.split_date)
        # timebased train test split
        X_train, X_test, y_train, y_test = self.timebased_train_test_split(
            train_df, TimeBasedCV
        )
        # scale the train and test data
        scaled_X_train, scaled_X_test, scaler = self.scale_data(X_train, X_test)
        print("fitting model")
        self.model.fit(scaled_X_train, y_train)
        y_pred = self.model.predict(scaled_X_test)
        model_accuracy = self.model.score(scaled_X_test, y_test)
        # print(f" Model Accuracy is: {model.score(X=scaled_X_test,y=y_test)*100}%")
        # Test on final data
        final_y_test = test_df["signal"]
        test_df = test_df.drop(["prediction_date", "signal"], axis=1)
        scaled_test_data = scaler.transform(test_df)
        final_accuracy = (self.model.score(scaled_test_data, final_y_test)) * 100
        print(f"Final model accuracy is {round(final_accuracy,2)}%")
        if final_accuracy > self.previous_model_score:
            date = datetime.strftime(datetime.today(), "%Y-%m-%d")
            print(f"The model trained on {date} now has the best accuracy ")
            pickle.dump(self.model, open(f"final_model_{date}.pkl", "wb"))
        return scaler
        # df_test = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

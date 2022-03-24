import numpy as np
import pandas as pd
from db_connection import connect
import datetime

# imports for database
import psycopg2 as pg
from joblib import dump, load
import streamlit as st


# SQL Query
sql = """
SELECT * FROM bitcoin
ORDER BY price_date DESC
LIMIT 7
"""

# read the raw bitcoin data from DB
df = pd.read_sql(sql=sql, con=connect())

# save the recent date in a the database
date = df["price_date"][0]
predicted_date = date  # save date for prediction
predicted_date += datetime.timedelta(days=7)  # increment date by 7 days for prediction


# drop the entry_id and date column
df.drop(['entry_id', "price_date"], axis=1, inplace=True)

# create new df and shifts all columns by 7 days
data = pd.DataFrame()  # create new df
for col in df.columns:
    for i in range(6, 0, -1):  # range for num of days
        data[f'{col}-{i}'] = df[col].shift(i)

# drop NA that are created during the shift
data.dropna(axis=0, inplace=True)


# read in final model file
loaded_model = load('final_model.joblib')

# prediction for 7 days from now
prediction = loaded_model.predict_proba(data)

output_message = ""
# checks the output of the prediction
if prediction[0][0] > prediction[0][1]:
    output_message = "SELL NOW!!"
else:
    output_message = "BUY NOW!!"

# write welcome page for users
st.write("""

# Simple Bitcoin Price Predictor App

Below is a prediction of if Bitcoin's price will go up or down 7 days from today.

"""
         )
if output_message == "BUY NOW":
    st.success(f"My model predicts that Bitcoin will increase in price on {predicted_date}. \
    Maybe you should buy. ")
    st.write(f"This prediction was ran on {date}")
else:
    st.error(f"My model predicts that Bitcoin will decrease in price on {predicted_date}. \
    Maybe you should sell.")
    st.write(f"This prediction was ran on {date}")
st.caption("DISCLOSURE: I am not a financial advisor. This is just a fun project. \
You are responsible for your own financial decisions.")

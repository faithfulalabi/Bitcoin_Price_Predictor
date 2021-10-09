import numpy as np
import pandas as pd
from db_connection import connect


def process_data(conn):

    """
    conn: Your DB connection parameters

    We are first reading the data from our DB, dropping primary key, shifting
    all the columns by 7 to make the input the previous 7 days of data, creating
    our output column with the logic that if the price today is greater than 7
    days from now it's a 1, and a 0 else. Lastly we drop all the 7 day data
    because we are only training on the previous 6 days of data. Only kept the
    7th day date to show the prediction date our signal column is predicting.
    """
    # read data from DB
    sql = 'SELECT * FROM bitcoin'
    df = pd.read_sql_query(sql=sql,con=conn)

    # drop the entry_id and date column
    df.drop(['entry_id'],axis=1,inplace=True)

    # create new df and shifts all columns by 7 days
    data = pd.DataFrame()  #create new df
    for col in df.columns:
        for i in range(7, 0, -1): #range for num of days
            data[f'{col}-{i}'] = df[col].shift(i)

    # drop NA that are created during the shift
    data.dropna(axis=0,inplace=True)

    # create signal column labeling 1 if price today is greater than 7days from now, else 0
    data['signal'] = data['close_price-1'].diff(-7).apply(lambda x: 1 if (x > 0) else 0)

    # drop all the 7th day data except for date; only training on previous 6days of data & predicting on 7th.
    data.drop(['price_date-7','price_date-6','price_date-5','price_date-4','price_date-3',
           'price_date-2','open_price-1','high-1','low-1','close_price-1','volume-1'],
          axis=1,inplace=True)
    # convert date object to datetime in order to store right format in DB
    data['price_date-1'] = pd.to_datetime(data['price_date-1'])
    # convert volume to int for DB format
    data[['volume-7','volume-6','volume-5','volume-4','volume-3','volume-2']] = data[['volume-7','volume-6','volume-5',

    print("Data processed successfully ")                                                                              'volume-4','volume-3','volume-2']].astype(int)
    return data

if __name__ == '__main__':

    process_data(conn=connect())

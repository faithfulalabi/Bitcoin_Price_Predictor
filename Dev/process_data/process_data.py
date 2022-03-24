import numpy as np
import pandas as pd
from db_connection import connect
import psycopg2 as pg2
import os


class processedData:

    def __init__(self):
        pass

    def process_data(self, conn):
        """
        conn: Your database connection parameters

        it firsts reads the daily data from the database, then drops primary key, shifts
        all the columns by 7 to make the input the previous 7 days of data, creates
        the output column with the logic that if the price today is greater than 7
        days from now it's a 1(buy), and a 0 (sell) else. Lastly we drop all the 7 day data
        because we are only training on the previous 6 days of data. Only kept the
        7th day date to show the prediction date our signal column is predicting.
        """
        # read data from DB
        sql = 'SELECT * FROM bitcoin'
        df = pd.read_sql_query(sql=sql, con=conn)

        # drop the entry_id and date column
        df.drop(['entry_id'], axis=1, inplace=True)

        # create new df and shifts all columns by 7 days
        data = pd.DataFrame()  # create new df
        for col in df.columns:
            for i in range(7, 0, -1):  # range for num of days
                data[f'{col}-{i}'] = df[col].shift(i)

        # drop NA that are created during the shift
        data.dropna(axis=0, inplace=True)

        # create signal column labeling 1 if price today is greater than 7days from now, else 0
        data['signal'] = data['close_price-1'].diff(-7).apply(lambda x: 0 if (x > 0) else 1)

        # drop all the 7th day data except for date; only training on previous 6days of data & predicting on 7th.
        data.drop(['price_date-7', 'price_date-6', 'price_date-5', 'price_date-4', 'price_date-3',
                   'price_date-2', 'open_price-1', 'high-1', 'low-1', 'close_price-1', 'volume-1'],
                  axis=1, inplace=True)
        # convert date object to datetime in order to store right format in DB
        data['price_date-1'] = pd.to_datetime(data['price_date-1'])
        # convert volume to int for DB format
        data[['volume-7', 'volume-6', 'volume-5', 'volume-4', 'volume-3', 'volume-2']
             ] = data[['volume-7', 'volume-6', 'volume-5', 'volume-4', 'volume-3', 'volume-2']].astype(int)

        print("Successfully processed the data ")
        return data

    def process_data_to_db(self, conn, df, table):
        """
        conn: Your database  connection parameters.
        df: The dataframe you will be inserting into your database.
        table: String table name in the database.

        It drops the table in the database if it already exists,
        then recreate the table with all the columns from our processed Dataframe.
        After, it saves the dataframe on disk as a csv file, load the csv file and
        use copy_from() to copy it to the table.
        """

    # sql query to execute before writing data to db
        statements = ["DROP TABLE IF EXISTS %s;" % (table),
                      """ CREATE TABLE %s(prediction_date DATE, open_price_1 NUMERIC,open_price_2 NUMERIC,
                    open_price_3 NUMERIC,open_price_4 NUMERIC,open_price_5 NUMERIC,open_price_6 NUMERIC,
                    high_1 NUMERIC,high_2 NUMERIC,high_3 NUMERIC,high_4 NUMERIC,high_5 NUMERIC,
                    high_6 NUMERIC,low_1 NUMERIC,low_2 NUMERIC,low_3 NUMERIC,low_4 NUMERIC,
                    low_5 NUMERIC,low_6 NUMERIC,close_price_1 NUMERIC,close_price_2 NUMERIC,
                    close_price_3 NUMERIC,close_price_4 NUMERIC,close_price_5 NUMERIC,close_price_6 NUMERIC,
                    volume_1 BIGINT,volume_2 BIGINT,volume_3 BIGINT,volume_4 BIGINT,volume_5 BIGINT,
                    volume_6 BIGINT,signal SMALLINT);""" % (table)]
        # Save the dataframe to disk
        tmp_df = "./tmp_dataframe.csv"
        df.to_csv(tmp_df, index=False, header=False)
        f = open(tmp_df, 'r')
        cursor = conn.cursor()
        try:
            for statement in statements:
                cursor.execute(statement)
                conn.commit()  # commit on each statement
            cursor.copy_from(f, table, sep=",")
            conn.commit()
        except (Exception, pg2.DatabaseError) as error:
            os.remove(tmp_df)
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
            return 1
        print("Successfully added processed data to the database...")
        cursor.close()
        os.remove(tmp_df)

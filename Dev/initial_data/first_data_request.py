import yfinance as yf
import psycopg2 as pg2
import os


class wholeDataWriteout:

    def __init__(self):
        pass

    def get_data(self):
        """
        Request earilest daily data of bitcoin using Yahoo finance API.
        """

        # request and download bitcoin price from start date till today
        data = yf.download(tickers='BTC-USD', start='2014-12-31', interval='1d', rounding=True)

        # make date a column not an index
        data = data.reset_index()

        # drop adjusted close price, it's the same as close price
        data.drop('Adj Close', axis=1, inplace=True)

        print("Successfully got bitcoin's data...")

        return data

    def whole_data_to_db(self, conn, df, table):
        """
        conn: Your database connection parameters
        df: The dataframe you will be inserting into your DB.
        table: Your table name in your database

        It's going to save the whole historical dataframe on disk as
        a csv file, load the csv file and use copy_from() to copy
        the whole data to the table in your database.
        """
        # Save the dataframe to disk
        tmp_df = "./tmp_dataframe.csv"
        df.to_csv(tmp_df, index_label='id', header=False)
        f = open(tmp_df, 'r')
        cursor = conn.cursor()
        try:
            cursor.copy_from(f, table, sep=",")
            conn.commit()
        except (Exception, pg2.DatabaseError) as error:
            os.remove(tmp_df)
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
            return 1
        print("Successfully added the data to your database..")
        cursor.close()
        os.remove(tmp_df)

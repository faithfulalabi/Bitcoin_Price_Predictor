import yfinance as yf
import psycopg2 as pg2


class dailyRequest:

    def __init__(self):
        pass

    def get_new_data(self, conn):
        """
        conn : Your database connection parameters

        It's going get the latest date and index currently in the
        database, request the data starting on that date, create new indexes starting
        from the lastest index. Lastly, get data in the format needed.
        """

        # SQL query to execute
        sql = """SELECT price_date,entry_id FROM bitcoin
        ORDER BY price_date DESC
        LIMIT 1;"""

        # connect to db and get date & entry_id
        cursor = conn.cursor()
        cursor.execute(sql)
        new = cursor.fetchall()
        cursor.close()

        # store most recent date from your db
        recent_date = new[0][0]
        # store most recent index from your db
        recent_index = new[0][1]
        # convert date to string
        recent_date = str(recent_date)

        # pass the date into data request & get the data
        data = yf.download(tickers='BTC-USD', start=f'{recent_date}', interval='1d', rounding=True)

        # make a list of the new index
        index = []
        for i in range(0, len(data)):
            index.append(recent_index+i)

        # reset the index to make date a column
        data.reset_index(inplace=True)
        # create index column at the front of df
        data.insert(loc=0, column='Index', value=index)
        # drop adjusted close price, it's the same as close price
        data.drop('Adj Close', axis=1, inplace=True)

        data.drop(data.tail(1).index, inplace=True)  # drop last n rows

        print("successfully recieved data...")

        return data

    def daily_data_to_db(self, conn, df, table):
        """
        Conn: Your database connection parameters.
        df: The dataframe you will be inserting into your database.
        table: The name of your table in your database.

        It stores the data as list of Tuples,reupdate the data of the
        latest date in the db data and uses cursor.executemany() to insert
        the data into your DB.
        """
        # Create a list of tuples from the dataframe values
        tuples = [tuple(x) for x in df.to_numpy()]
        # Comma-separated dataframe columns
        cols = ', '.join([f'{col}' for col in ['entry_id', 'price_date', 'open_price', 'high',
                                               'low', 'close_price', 'volume']])
        updates = ', '.join([f'{col} = EXCLUDED.{col}' for col in ['price_date', 'open_price', 'high',
                                                                   'low', 'close_price', 'volume']])

        # SQL query to execute
        query = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s,%%s) ON CONFLICT (entry_id) DO UPDATE SET %s" % (
            table, cols, updates)
        cursor = conn.cursor()
        try:
            cursor.executemany(query, tuples)
            conn.commit()
        except (Exception, pg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
            return 1
        print("Successfully inserted daily data to your database.....")
        cursor.close()

from db_connection import connect
import yfinance as yf

def get_new_data(conn):
    """
    conn : Your DB connection parameters

    Here we are going get the lastest date and index currently in the
    DB, request the data starting on that date, create new indexes starting
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
    data = yf.download(tickers='BTC-USD',start=f'{recent_date}',interval='1d',rounding=True)

    # make a list of the new index
    index = []
    for i in range(0,len(data)):
        index.append(recent_index+i)

    # reset the index to make date a column
    data.reset_index(inplace=True)
    # create index column at the front of df
    data.insert(loc=0,column='Index',value=index)
    #drop adjusted close price, it's the same as close price
    data.drop('Adj Close',axis=1,inplace=True)

    print("successfully recieved data...")

    return data
if __name__ == '__main__':

    get_new_data(connect())

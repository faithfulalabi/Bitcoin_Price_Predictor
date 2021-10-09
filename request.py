import yfinance as yf


def get_data():

    """
    Request earilest data for bitcoin using Yahoo finance API.
    """

    # request and download bitcoin price from start date till today
    data = yf.download(tickers='BTC-USD',start='2014-12-31',interval='1d',rounding=True)

    # make date a column not an index
    data = data.reset_index()

    #drop adjusted close price, it's the same as close price
    data.drop('Adj Close',axis=1,inplace=True)

    print("Successfully got data...")

    return data

if __name__ == '__main__':
    get_data()

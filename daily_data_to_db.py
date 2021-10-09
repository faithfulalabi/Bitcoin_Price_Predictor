# this script can be ran every couple of days to write new data to DB.
import psycopg2 as pg2
from db_connection import connect
from daily_request import get_new_data

def execute_many(conn, df, table):
    """
    Conn: Your DB connection parameters.
    df: The dataframe you will be inserting into your DB.
    table: The name of your table in your DB.

    Here we will store our data as list of Tuples,reupdate the data of the
    latest date in the db data and uses cursor.executemany() to insert
    the data into your DB.
    """
    # Create a list of tuples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols= ', '.join([f'{col}' for col in ['entry_id','price_date','open_price','high',
                                       'low','close_price','volume']])
    updates = ', '.join([f'{col} = EXCLUDED.{col}' for col in ['price_date','open_price','high',
                                       'low','close_price','volume']])

    # SQL query to execute
    query = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s,%%s) ON CONFLICT (entry_id) DO UPDATE SET %s" % (table, cols,updates)
    cursor = conn.cursor()
    try:
        cursor.executemany(query,tuples)
        conn.commit()
    except (Exception, pg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("Data was inserted successfully using execute_many()...")
    cursor.close()

if __name__ == '__main__':

    execute_many(conn= connect(), df= get_new_data(connect()), table= 'bitcoin')

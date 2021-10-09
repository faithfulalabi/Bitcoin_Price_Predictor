# Only run this script once or you will get a duplicate key value violation on constraint error.
from db_connection import connect
from request import get_data
import psycopg2 as pg2
import os


def whole_data_to_db(conn, df, table):
    """
    conn: Your DB connection parameters
    df: The dataframe you will be inserting into your DB.
    table: Your table name in postgresql

    Here we are going save the whole historical dataframe on disk as
    a csv file, load the csv file
    and use copy_from() to copy the whole data to the table in postgresql.
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
    print("Data successfully got added to the DB using whole_data_to_db()...")
    cursor.close()
    os.remove(tmp_df)

if __name__ == '__main__':

    whole_data_to_db(conn=connect(), df=get_data(), table='bitcoin')

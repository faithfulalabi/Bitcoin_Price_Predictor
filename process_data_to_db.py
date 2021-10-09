import psycopg2 as pg2
import os
from db_connection import connect
from data_processing import process_data


def process_data_to_db(conn, df, table):
    """
    conn: Your DB connection parameters
    df: The dataframe you will be inserting into your DB.
    table: Your table name in postgresql

    Here we are going drop the the table in the database if it already exists,
    then recreate the table with all the columns from our processed Dataframe.
    After save the dataframe on disk as a csv file, load the csv file and
    use copy_from() to copy it to the table.
    """
    #sql query to execute before writing data to db
    statements = ["DROP TABLE IF EXISTS %s;" %(table),
                """ CREATE TABLE %s(prediction_date DATE, open_price_1 NUMERIC,open_price_2 NUMERIC,
                open_price_3 NUMERIC,open_price_4 NUMERIC,open_price_5 NUMERIC,open_price_6 NUMERIC,
                high_1 NUMERIC,high_2 NUMERIC,high_3 NUMERIC,high_4 NUMERIC,high_5 NUMERIC,
                high_6 NUMERIC,low_1 NUMERIC,low_2 NUMERIC,low_3 NUMERIC,low_4 NUMERIC,
                low_5 NUMERIC,low_6 NUMERIC,close_price_1 NUMERIC,close_price_2 NUMERIC,
                close_price_3 NUMERIC,close_price_4 NUMERIC,close_price_5 NUMERIC,close_price_6 NUMERIC,
                volume_1 BIGINT,volume_2 BIGINT,volume_3 BIGINT,volume_4 BIGINT,volume_5 BIGINT,
                volume_6 BIGINT,signal SMALLINT);""" %(table)]
    # Save the dataframe to disk
    tmp_df = "./tmp_dataframe.csv"
    df.to_csv(tmp_df, index= False, header=False)
    f = open(tmp_df, 'r')
    cursor = conn.cursor()
    try:
        for statement in statements:
            cursor.execute(statement)
            conn.commit() # commit on each statement
        cursor.copy_from(f, table, sep=",")
        conn.commit()
    except (Exception, pg2.DatabaseError) as error:
        os.remove(tmp_df)
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("Data was added successfully using processed_data_to_db()...")
    cursor.close()
    os.remove(tmp_df)

if __name__ == '__main__':

    process_data_to_db(conn=connect(),df=process_data(connect()),table='processed_bitcoin')

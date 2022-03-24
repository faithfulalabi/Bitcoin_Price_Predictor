from process_data import processedData
from db_connection import connect


def main():
    processed_data = processedData()
    read_data = processed_data.process_data(conn=connect())
    processed_data.process_data_to_db(conn=connect(), df=read_data, table='processed_bitcoin')


if __name__ == '__main__':
    main()

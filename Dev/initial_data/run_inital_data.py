from first_data_request import wholeDataWriteout
from db_connection import connect


def main():
    whole_data_writeout = wholeDataWriteout()
    read_data = whole_data_writeout.get_data()
    whole_data_writeout.whole_data_to_db(conn=connect(), df=read_data, table='bitcoin')


if __name__ == '__main__':
    main()

import sys

print(sys.path)
from daily_request import dailyRequest
import connection.db_connection


def main():
    daily_request = dailyRequest()
    new_data = daily_request.get_new_data(conn=connection.db_connection.connect())
    daily_request.daily_data_to_db(
        conn=connection.db_connection.connect(), df=new_data, table="bitcoin"
    )


if __name__ == "__main__":
    main()

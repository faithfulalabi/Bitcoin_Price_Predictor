from configparser import ConfigParser


def config(
    filename="database.ini",
    section="postgresql",
):
    """
    We read and make items in database.ini file all key value pairs
    """

    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(f"Section {section} not found in the {filename} file")

    return db


# /Users/faithful/Desktop/Data_Sci_Projects/Bitcoin_Price_Predictor/Dev/db_connection/

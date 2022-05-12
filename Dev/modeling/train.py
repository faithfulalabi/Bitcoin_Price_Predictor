from knnClassifer import Classifier
from db_connection import connect

if __name__ == "__main__":
    classifer = Classifier(connect, 58.5)
    classifer.train()

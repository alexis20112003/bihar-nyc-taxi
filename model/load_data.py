import sqlite3
import pandas as pd

import common
DB_PATH = common.CONFIG['paths']['db_path']


def load_train_data():
    print(f"Reading train data from the database: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    data_train = pd.read_sql('SELECT * FROM train', con)
    con.close()
    X = data_train.drop(columns=['target'])
    y = data_train['target']
    return X, y


def load_test_data():
    print(f"Reading test data from the database: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    data_test = pd.read_sql(f'SELECT * FROM test', con)
    con.close()
    X = data_test.drop(columns=['target'])
    y = data_test['target']
    return X, y


def load_random_test_data(n_samples=5):
    print(f"Reading random test data from the database: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    data_test = pd.read_sql(f'SELECT * FROM test ORDER BY RANDOM() LIMIT {n_samples}', con)
    con.close()
    X = data_test.drop(columns=['target'])
    y = data_test['target']
    return X, y

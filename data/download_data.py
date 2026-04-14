import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import sqlite3
import os
import common

DB_PATH = common.CONFIG['paths']['db_path']
RANDOM_STATE = int(common.CONFIG['ml']['random_state'])

def download_data():
    X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=True)
    X.columns = ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']
    data = pd.concat([X,y], axis=1)
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    print(f"Saving train and test data to a database: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as con:
        # cur = con.cursor()
        # cur.execute("DROP TABLE IF EXISTS train")
        # cur.execute("DROP TABLE IF EXISTS test")
        data_train.to_sql(name='train', con=con, if_exists="replace", index=False)
        data_test.to_sql(name='test', con=con, if_exists="replace", index=False)

def test_download_data():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()

        print(f"Reading train data from the database: {DB_PATH}")
        # getting the number of lines in train table
        res = cur.execute("SELECT COUNT(*) FROM train")
        n_rows = res.fetchone()[0]
        # getting the number of columns in train table
        res = cur.execute("SELECT * FROM train LIMIT 1")
        n_cols = len(res.description)
        print(f'Train data: {n_rows} x {n_cols}')
        # show column names
        # # for column in res.description:
        # #     print(column[0])

        print(f"Reading test data from the database: {DB_PATH}")
        # getting the number of lines in test table
        res = cur.execute("SELECT COUNT(*) FROM test")
        n_rows = res.fetchone()[0]
        # getting the number of columns in test table
        res = cur.execute("SELECT * FROM test LIMIT 1")
        n_cols = len(res.description)
        print(f'Test data: {n_rows} x {n_cols}')

if __name__ == "__main__":

    download_data()
    test_download_data()

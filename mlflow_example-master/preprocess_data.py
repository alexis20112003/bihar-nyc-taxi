import pandas as pd
from sklearn.model_selection import train_test_split

import os, pickle
import common as common

DATA_PATH = common.CONFIG['paths']['data']
DATA_PROC_PATH = common.CONFIG['paths']['data_processed']

TARGET = common.CONFIG['ml']['target_name']
RANDOM_STATE = common.CONFIG['ml']['random_state']

def preprocess_data():

    print("Loading wine dataset...")
    data = pd.read_csv(DATA_PATH)
    print(f"{len(data)} objects loaded")

    # Separate features and target
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    print(f"{len(X_train)} objects in train set, {len(X_test)} objects in test set")

    print(f"Preprocessing data...")
    # Do some preprocessing if needed
    print(f"Done")

    print(f"Saving results to {DATA_PROC_PATH}...")
    # Save split results
    model_dir = os.path.dirname(DATA_PROC_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(DATA_PROC_PATH, "wb") as file:
        pickle.dump((X_train, X_test, y_train, y_test), file)
    print(f"Done")

if __name__ == "__main__":

    preprocess_data()
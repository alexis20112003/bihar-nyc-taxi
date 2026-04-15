import sqlite3
from datetime import datetime

import config
DB_PATH = config.CONFIG['paths']['db_path']


def save_prediction(trip: dict, result: int, endpoint: str):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """INSERT INTO predictions
               (timestamp, endpoint, vendor_id, pickup_datetime, passenger_count,
                pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
                store_and_fwd_flag, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), endpoint,
             trip['vendor_id'], trip['pickup_datetime'], trip['passenger_count'],
             trip['pickup_longitude'], trip['pickup_latitude'],
             trip['dropoff_longitude'], trip['dropoff_latitude'],
             trip['store_and_fwd_flag'], result)
        )

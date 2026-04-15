import streamlit as st
import requests
from datetime import datetime, time

import common

API_URL = common.CONFIG['api']['url']


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="NYC Taxi Trip Duration",
    )

    st.write(
        """
        # NYC Taxi Trip Duration
        Predict the duration of a taxi trip in New York City.
        """
    )


def write_user_data(user_input):
    st.write("## Trip data")
    st.write(user_input)


def write_prediction(result):
    minutes = result // 60
    seconds = result % 60
    st.write("## Prediction")
    st.metric("Estimated trip duration", f"{minutes} min {seconds} s")
    st.write(f"({result} seconds)")


def process_side_bar_inputs():
    st.sidebar.header("Trip data")
    user_input = sidebar_input_features()

    st.write("## Trip data")
    st.json(user_input)

    if st.sidebar.button("Predict"):
        try:
            response = requests.post(f"{API_URL}/predict", json=user_input)
            if response.status_code == 200:
                result = response.json()["result"]
                write_prediction(result)
            elif response.status_code == 422:
                st.error(f"Validation error: {response.json()['detail']}")
            else:
                st.error(f"API error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure the API is running on " + API_URL)


def sidebar_input_features():

    vendor_id = st.sidebar.selectbox("Vendor", (1, 2))

    pickup_date = st.sidebar.date_input("Pickup date", value=datetime(2016, 6, 15))
    pickup_time = st.sidebar.time_input("Pickup time", value=time(12, 0))
    pickup_datetime = datetime.combine(pickup_date, pickup_time).strftime("%Y-%m-%d %H:%M:%S")

    passenger_count = st.sidebar.slider(
        "Passenger count",
        min_value=1, max_value=9, value=1, step=1)

    pickup_longitude = st.sidebar.number_input(
        "Pickup longitude", value=-73.9857, format="%.4f")

    pickup_latitude = st.sidebar.number_input(
        "Pickup latitude", value=40.7484, format="%.4f")

    dropoff_longitude = st.sidebar.number_input(
        "Dropoff longitude", value=-73.9856, format="%.4f")

    dropoff_latitude = st.sidebar.number_input(
        "Dropoff latitude", value=40.7489, format="%.4f")

    store_and_fwd_flag = st.sidebar.selectbox("Store and forward", ("N", "Y"))

    return {
        "vendor_id": vendor_id,
        "pickup_datetime": pickup_datetime,
        "passenger_count": passenger_count,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "store_and_fwd_flag": store_and_fwd_flag,
    }


if __name__ == "__main__":
    process_main_page()



def check_longitude(v: float) -> float:
    if not -180 <= v <= 180:
        raise ValueError("Longitude hors limites")
    return v

def check_latitude(v: float) -> float:
    if not -90 <= v <= 90:
        raise ValueError("Latitude hors NYC")
    return v

def check_haversine_distance(trip) -> None:
    from model.train import haversine_array
    distance = haversine_array(
        trip.pickup_latitude, trip.pickup_longitude,
        trip.dropoff_latitude, trip.dropoff_longitude
    )
    if distance < 0.05:
        raise ValueError("Distance pickup-dropoff trop courte (< 50m)")
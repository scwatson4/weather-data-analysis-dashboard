import json
import pytz
import requests
import pandas as pd
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_station_data_for_period_temp(date_input: str, island_name: str, variable: str):
    """
    Fetches station-level climate data for a given island, day/month, and variable.

    Parameters:
    - date_input (str): Either "MM/YYYY" for full month or "MM/DD/YYYY" for a specific day
    - island_name (str): Name of the island (e.g., "Oahu", "Maui", "Lanai", etc.)
    - variable (str): Either "temperature" or "rainfall"

    Returns:
    - pd.DataFrame: Daily station-level data for the given time and island
    """

    hcdp_api_token = os.getenv("OAUTH_TOKEN")
    if not hcdp_api_token:
        raise ValueError("OAUTH_TOKEN is not set in the environment")
    api_base_url = "https://api.hcdp.ikewai.org"
    header = {"Authorization": f"Bearer {hcdp_api_token}"}

    def query_stations(values, name, limit=10000, offset=0):
        params = {"name": name}
        for key in values:
            params[f"value.{key}"] = values[key]
        params = {"q": json.dumps(params), "limit": limit, "offset": offset}
        stations_ep = "/stations"
        url = f"{api_base_url}{stations_ep}"
        res = requests.get(url, params=params, headers=header)
        res.raise_for_status()
        return [item["value"] for item in res.json()["result"]]

    def get_station_metadata():
        res = query_stations({}, name="hcdp_station_metadata")
        return {m[m["id_field"]]: m for m in res}

    def get_station_data(values, metadata=None, limit=10000, offset=0):
        res = query_stations(values, name="hcdp_station_value", limit=limit, offset=offset)
        if metadata:
            return [item | metadata.get(item["station_id"], {}) for item in res]
        return res

    # Define all island polygons
    islands = {
        "Hawaii (Big Island)": Polygon([(-156.1, 18.9), (-154.7, 18.9), (-154.7, 20.3), (-156.1, 20.3)]),
        "Maui": Polygon([(-156.8, 20.5), (-156.2, 20.5), (-156.2, 21.0), (-156.8, 21.0)]),
        "Oahu": Polygon([(-158.3, 21.2), (-157.6, 21.2), (-157.6, 21.8), (-158.3, 21.8)]),
        "Kauai": Polygon([(-159.8, 21.8), (-159.2, 21.8), (-159.2, 22.3), (-159.8, 22.3)]),
        "Molokai": Polygon([(-157.4, 20.5), (-156.7, 20.5), (-156.7, 21.2), (-157.4, 21.2)]),
        "Lānai": Polygon([(-157.1, 20.7), (-156.8, 20.7), (-156.8, 21.0), (-157.1, 21.0)]),
        "Niihau": Polygon([(-160.3, 21.8), (-160.0, 21.8), (-160.0, 22.0), (-160.3, 22.0)]),
        "Kahoolawe": Polygon([(-156.7, 20.5), (-156.5, 20.5), (-156.5, 20.7), (-156.7, 20.7)])
    }

    def get_island(lat, lon):
        point = Point(lon, lat)
        for name, poly in islands.items():
            if poly.contains(point):
                return name
        return "Unknown or offshore"

    # Normalize island input
    island_name = island_name.lower()
    matched_island = None
    for name in islands.keys():
        if island_name in name.lower():
            matched_island = name
            break
    if not matched_island:
        raise ValueError(f"Island '{island_name}' not recognized.")

    # Determine date range
    try:
        if len(date_input) == 7:  # MM/YYYY
            start_date = datetime.strptime("01/" + date_input, "%d/%m/%Y")
            end_date = (start_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        elif len(date_input) == 10:  # MM/DD/YYYY
            start_date = datetime.strptime(date_input, "%m/%d/%Y")
            end_date = start_date
        else:
            raise ValueError("Date input must be in MM/YYYY or MM/DD/YYYY format.")
    except ValueError as e:
        raise ValueError(f"Date parsing failed: {e}")

    date_list = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    metadata = get_station_metadata()
    records = []

    for date in date_list:
        date_str = date.strftime("%Y-%m-%d")
        display_date = date.strftime("%m/%d/%Y")
        all_station_data = {}

        if variable == "temperature":
            values = {
                "datatype": "temperature",
                "aggregation": "max",  # only max-temp now
                "period": "day",
                "date": date_str
            }
            data = get_station_data(values, metadata)
            for item in data:
                if not ("lat" in item and "lng" in item): continue
                lat, lon = float(item["lat"]), float(item["lng"])
                if get_island(lat, lon) != matched_island:
                    continue
                sid = item["station_id"]
                if sid not in all_station_data:
                    all_station_data[sid] = {
                        "Time": display_date,
                        "lat": lat,
                        "lon": lon
                    }
                all_station_data[sid]["max-temp"] = float(item["value"])

        elif variable == "rainfall":
            values = {
                "datatype": "rainfall",
                "production": "new",
                "period": "day",
                "date": date_str
            }
            data = get_station_data(values, metadata)
            for item in data:
                if not ("lat" in item and "lng" in item): continue
                lat, lon = float(item["lat"]), float(item["lng"])
                if get_island(lat, lon) != matched_island:
                    continue
                sid = item["station_id"]
                if sid not in all_station_data:
                    all_station_data[sid] = {
                        "Time": display_date,
                        "lat": lat,
                        "lon": lon
                    }
                all_station_data[sid]["rainfall"] = float(item["value"])

        for station_record in all_station_data.values():
            row = {
                "Time": station_record["Time"],
                "lat": station_record["lat"],
                "lon": station_record["lon"]
            }
            if variable == "temperature":
                row["max-temp"] = station_record.get("max-temp")
                #row["min-temp"] = station_record.get("min-temp")
                #row["avg-temp"] = station_record.get("mean-temp")
            elif variable == "rainfall":
                row["rainfall"] = station_record.get("rainfall")
            records.append(row)

    df = pd.DataFrame(records)
    return df
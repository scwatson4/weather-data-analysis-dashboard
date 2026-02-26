import json
import requests
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from dateutil.relativedelta import relativedelta
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def generate_rainfall_forecast_plot(month: str, latitude: float, longitude: float):
    """
    Generate and display a Plotly chart of actual vs predicted daily rainfall.
    
    Forecast range: Apr 4, 2025 to end of input month.
    Actuals: Dec 2024 to end of input month.
    
    Parameters:
        month (str): "MM/YYYY" format (e.g., "06/2025")
        latitude (float): Latitude of location
        longitude (float): Longitude of location
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
        url = f"{api_base_url}/stations"
        res = requests.get(url, params=params, headers=header)
        res.raise_for_status()
        return [item["value"] for item in res.json()["result"]]

    def get_station_metadata():
        res = query_stations({}, name="hcdp_station_metadata")
        return {m[m["id_field"]]: m for m in res}

    def get_closest_station_id(lat, lon, metadata):
        point = Point(lon, lat)
        closest_station, min_dist = None, float("inf")
        for sid, meta in metadata.items():
            try:
                station_point = Point(float(meta["lng"]), float(meta["lat"]))
                dist = point.distance(station_point)
                if dist < min_dist:
                    min_dist = dist
                    closest_station = sid
            except:
                continue
        return closest_station

    now = datetime(2025, 4, 6)
    target_month = datetime.strptime("01/" + month, "%d/%m/%Y")
    forecast_start = datetime(2025, 4, 4)
    forecast_end = (target_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)

    train_start = forecast_start - relativedelta(months=36)
    train_end = forecast_start - timedelta(days=1)

    actual_start = datetime(2024, 12, 1)
    actual_end = forecast_end

    metadata = get_station_metadata()
    station_id = get_closest_station_id(latitude, longitude, metadata)
    if not station_id:
        raise ValueError("No nearby station found.")

    values_train = {
        "station_id": station_id,
        "datatype": "rainfall",
        "production": "new",
        "period": "day",
        "fill": "partial",
        "date": {
            "$gte": train_start.strftime("%Y-%m-%d"),
            "$lte": train_end.strftime("%Y-%m-%d")
        }
    }
    train_raw = query_stations(values_train, name="hcdp_station_value")
    df_train = pd.DataFrame([
        {
            "date": datetime.strptime(r["date"], "%Y-%m-%d"),
            "day": datetime.strptime(r["date"], "%Y-%m-%d").day,
            "month": datetime.strptime(r["date"], "%Y-%m-%d").month,
            "year": datetime.strptime(r["date"], "%Y-%m-%d").year,
            "rainfall": float(r["value"])
        }
        for r in train_raw if "value" in r
    ])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df_train[["day", "month", "year"]], df_train["rainfall"])

    forecast_dates = [forecast_start + timedelta(days=i) for i in range((forecast_end - forecast_start).days + 1)]
    df_forecast = pd.DataFrame({
        "date": forecast_dates,
        "day": [d.day for d in forecast_dates],
        "month": [d.month for d in forecast_dates],
        "year": [d.year for d in forecast_dates]
    })
    df_forecast["predicted_rainfall"] = model.predict(df_forecast[["day", "month", "year"]])

    values_actual = {
        "station_id": station_id,
        "datatype": "rainfall",
        "production": "new",
        "period": "day",
        "fill": "partial",
        "date": {
            "$gte": actual_start.strftime("%Y-%m-%d"),
            "$lte": actual_end.strftime("%Y-%m-%d")
        }
    }
    actual_raw = query_stations(values_actual, name="hcdp_station_value")
    df_actual = pd.DataFrame([
        {
            "date": datetime.strptime(r["date"], "%Y-%m-%d"),
            "rainfall": float(r["value"])
        }
        for r in actual_raw if "value" in r
    ]).sort_values("date")

    # Plotly Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_actual["date"],
        y=df_actual["rainfall"],
        mode='lines+markers',
        name='Actual Rainfall',
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_forecast["date"],
        y=df_forecast["predicted_rainfall"],
        mode='lines+markers',
        name='Predicted Rainfall',
        line=dict(width=3)
    ))

    fig.update_layout(
        title=dict(
            text="Actual vs Predicted Daily Rainfall",
            font=dict(size=25)
        ),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=24)),
            tickfont=dict(size=20),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="Rainfall (mm)", font=dict(size=24)),
            tickfont=dict(size=20),
            showgrid=False
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            font=dict(size=20)
        ),
        font=dict(size=18),  # Base font size
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig)
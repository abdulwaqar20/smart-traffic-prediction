import pandas as pd
import os
from datetime import datetime

class CSVDataStorage:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_all_data(self):
        """Load and merge all data sources with safety checks"""
        try:
            # File paths
            traffic_path = f"{self.data_dir}/traffic_data.csv"
            weather_path = f"{self.data_dir}/weather_data.csv"
            incidents_path = f"{self.data_dir}/incidents.csv"

            # Ensure all files exist
            if not all(os.path.exists(p) for p in [traffic_path, weather_path, incidents_path]):
                raise FileNotFoundError("❌ One or more CSV files are missing in the data directory.")

            # Load CSVs with encoding for non-ASCII characters
            traffic = pd.read_csv(traffic_path, encoding='utf-8')
            weather = pd.read_csv(weather_path, encoding='utf-8')
            incidents = pd.read_csv(incidents_path, encoding='utf-8')

            # Convert timestamp to datetime early
            traffic['timestamp'] = pd.to_datetime(traffic['timestamp'], errors='coerce')
            weather['timestamp'] = pd.to_datetime(weather['timestamp'], errors='coerce')
            incidents['timestamp'] = pd.to_datetime(incidents['timestamp'], errors='coerce')

            # Merge traffic and weather
            df = pd.merge(traffic, weather, on=['timestamp', 'city'], how='left')

            # Aggregate incident data
            incidents_agg = incidents.groupby(['timestamp', 'city']).agg({
                'severity': ['mean', 'count']
            }).reset_index()
            incidents_agg.columns = ['timestamp', 'city', 'avg_severity', 'incident_count']

            # Merge with incident aggregation
            df = pd.merge(df, incidents_agg, on=['timestamp', 'city'], how='left')

            # Feature engineering
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            # Encode weather_condition using one-hot encoding
            df = pd.get_dummies(df, columns=['weather_condition'], drop_first=True)

            # Compute congestion_level
            df['congestion_level'] = (1 - (df['current_speed'] / df['free_flow_speed'])) * 100
            df['congestion_level'] = df['congestion_level'].clip(lower=0, upper=100)

            # Fill missing values
            df['incident_count'] = df['incident_count'].fillna(0)
            df['avg_severity'] = df['avg_severity'].fillna(0)

            # Drop rows with remaining missing values
            df = df.dropna()

            return df

        except Exception as e:
            print(f"Data loading error: {str(e)}")
            return None

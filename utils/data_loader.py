import pandas as pd
import os

DATA_PATH = 'data/traffic_data.csv'

def load_traffic_data():
    """Load and preprocess traffic data"""
    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        
        # Preprocessing
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_holiday'] = data['is_holiday'].astype(int)
        
        # Handle missing values
        data.fillna({
            'precipitation': 0,
            'temperature': data['temperature'].mean(),
            'incident_reported': 0
        }, inplace=True)
        
        return data
    return None
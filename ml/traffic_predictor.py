import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model(csv_path='data/traffic_data.csv'):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    # Feature and target selection - adjust based on actual dataset
    X = df[['temp', 'rain_1h', 'snow_1h', 'clouds_all']]
    y = df['congestion_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'models/congestion_model.pkl')
    print("Model trained and saved.")

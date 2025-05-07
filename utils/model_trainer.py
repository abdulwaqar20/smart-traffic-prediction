import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class TrafficModelTrainer:
    def __init__(self, model_path="models/traffic_model.pkl", save_interval=10):
        """Initialize the model trainer with a save path"""
        self.model_path = model_path
        self.save_interval = save_interval  # Save the model after every X iterations
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model = None
    
    def train_model(self, data, test_size=0.2, n_estimators=100, random_state=42):
        """Train a traffic congestion prediction model"""
        try:
            # Prepare features and target
            X = data.drop(columns=['congestion_level', 'timestamp', 'city'], errors='ignore')
            y = data['congestion_level']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                max_depth=10,
                min_samples_split=5
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Periodically save the model
            if self.save_interval:
                joblib.dump(self.model, self.model_path)
                print(f"✅ Model saved at interval.")

            # Final save after training
            joblib.dump(self.model, self.model_path)
            print(f"✅ Model saved after training.")
            
            return {
                'model': self.model,
                'metrics': {
                    'MAE': round(mae, 2),
                    'R2': round(r2, 2)
                },
                'features': list(X.columns)
            }
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return None
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            self.model = joblib.load(self.model_path)
            return self.model
        except Exception as e:
            print(f"Loading error: {str(e)}")
            return None
    
    def predict(self, input_data):
        """Make a prediction using the trained model"""
        if self.model is None:
            self.load_model()
        if self.model:
            return self.model.predict(input_data)
        return None

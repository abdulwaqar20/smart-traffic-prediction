import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "AdjhtQTjFRnPR3Ld7PTOlG0HaWk21Vox")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "37e2acaaeecb9c766ca053c351ae2365")
    
    # Cities Configuration
    CITIES = {
        "New York": {"lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York"},
        "London": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"}, 
        "Tokyo": {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
        "Berlin": {"lat": 52.5200, "lon": 13.4050, "timezone": "Europe/Berlin"}
    }
    
    # Data Collection
    COLLECTION_INTERVAL = 300  # 5 minutes in seconds
    MAX_RETRIES = 3
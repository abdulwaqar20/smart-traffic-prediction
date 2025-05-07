import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Added import for gauge chart
from utils.storage import CSVDataStorage
from utils.model_trainer import TrafficModelTrainer
from datetime import datetime

# Initialize services
data_storage = CSVDataStorage(data_dir="data")
model_trainer = TrafficModelTrainer(model_path="models/traffic_model.pkl")

# Streamlit UI Configuration
st.set_page_config(layout="wide", page_title="Traffic Prediction")

# Sidebar - Model Training
st.sidebar.header("Model Configuration")
train_col1, train_col2 = st.sidebar.columns(2)

with train_col1:
    test_size = st.slider("Test Size (%)", 10, 40, 20)
with train_col2:
    n_estimators = st.slider("Number of Trees", 50, 200, 100)

if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        data = data_storage.load_all_data()
        if data is not None:
            model = model_trainer.train_model(data, test_size=test_size/100, n_estimators=n_estimators)
            if model:
                st.sidebar.success(f"Model trained successfully! (Features used: {len(model.feature_names_in_)})")
                st.sidebar.write("Feature Importance:")
                feat_imp = pd.DataFrame({
                    'Feature': model.feature_names_in_,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.sidebar.dataframe(feat_imp, hide_index=True)
            else:
                st.sidebar.error("Training failed")
        else:
            st.sidebar.error("Could not load data")

# Main Prediction Interface
st.header("🚦 Traffic Congestion Prediction")

# Load model or show warning
model = model_trainer.load_model()
if not model:
    st.warning("No trained model available. Please train a model first.")
else:
    # Create prediction form
    with st.form("prediction_form"):
        st.subheader("Input Parameters")
        cols = st.columns(3)
        
        with cols[0]:
            hour = st.slider("Hour of day", 0, 23, 17)  # Default to 5 PM
            day_of_week = st.selectbox("Day of week", 
                                     ["Monday", "Tuesday", "Wednesday", 
                                      "Thursday", "Friday", "Saturday", "Sunday"])
            incidents = st.number_input("Number of incidents", 0, 50, 2)
            
        with cols[1]:
            temperature = st.number_input("Temperature (°C)", value=22.0, min_value=-20.0, max_value=50.0)
            humidity = st.slider("Humidity (%)", 0, 100, 65)
            visibility = st.slider("Visibility (km)", 0, 20, 10)
            
        with cols[2]:
            weather = st.selectbox("Weather condition", 
                                 ["Clear", "Clouds", "Rain", "Snow", "Fog"])
            precipitation = st.slider("Precipitation (mm)", 0.0, 20.0, 0.0)
        
        submitted = st.form_submit_button("Predict Congestion")
        
        if submitted:
            # Prepare input data
            day_mapping = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            weather_mapping = {
                "Clear": 0, "Clouds": 1, "Rain": 2, "Snow": 3, "Fog": 4
            }
            
            input_data = {
                'hour': hour,
                'day_of_week': day_mapping[day_of_week],
                'is_weekend': 1 if day_mapping[day_of_week] in [5,6] else 0,
                'temperature': temperature,
                'weather_condition': weather_mapping[weather],
                'precipitation': precipitation,
                'humidity': humidity,
                'visibility': visibility,
                'incident_count': incidents
            }
            
            # Create dataframe with correct feature order
            input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                
                # Display results
                st.subheader("Prediction Results")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("Predicted Congestion", f"{prediction:.1f}%")
                
                with res_col2:
                    congestion_level = "Low"
                    if prediction > 40: congestion_level = "High"
                    elif prediction > 20: congestion_level = "Moderate"
                    st.metric("Congestion Level", congestion_level)
                
                with res_col3:
                    st.metric("Recommended Action", 
                             "Avoid travel" if prediction > 40 else 
                             "Expect delays" if prediction > 20 else "Normal traffic")
                
                # Create gauge chart using plotly.graph_objects
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Congestion Level"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 40], 'color': "lightyellow"},
                            {'range': [40, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# Data Analysis Section
st.header("📊 Data Analysis")
data = data_storage.load_all_data()

if data is not None:
    tab1, tab2, tab3 = st.tabs(["Trends", "Correlations", "Raw Data"])
    
    with tab1:
        st.subheader("Historical Trends")
        city = st.selectbox("Select City", data['city'].unique(), key='city_select')
        metric = st.selectbox("Select Metric", 
                            ['congestion_level', 'current_speed', 'temperature', 'incident_count'])
        
        city_data = data[data['city'] == city]
        fig = px.line(city_data, x='timestamp', y=metric,
                     title=f"{metric.replace('_', ' ').title()} in {city}")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Correlations")
        numeric_cols = data.select_dtypes(include=['number']).columns
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Raw Data Preview")
        st.dataframe(data.sort_values('timestamp', ascending=False).head(100), 
                    height=400)
else:
    st.warning("No data available for visualization")

# Footer
st.markdown("---")
st.caption("Smart Traffic Prediction System | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
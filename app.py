import base64
import pickle

import pandas as pd
import streamlit as st

st.title("Swiggy Delivery Time Prediction")

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("swiggy_img.jpg")
df = pd.read_csv(r"swiggy_demographic.csv")
model = pickle.load(open('swiggy_model.pkl', 'rb'))

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.text("👤 Delivery Executive Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)

with col2:
    ratings = st.slider("Ratings", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

with col3:
    vehicle_condition = st.selectbox("Vehicle Condition",sorted(df["vehicle_condition"].dropna().unique()))


st.text("🛵 Delivery Information")

col1, col2, col3 = st.columns(3)

with col1:
    type_of_vehicle = st.selectbox("Vehicle Type",df["type_of_vehicle"].dropna().unique())

with col2:
    multiple_deliveries = st.selectbox("Multiple Deliveries",sorted(df["multiple_deliveries"].dropna().unique()))

with col3:
    distance = st.number_input("Distance (km)",min_value=0.0,value=3.0,step=0.1)
    
    
st.text("🍔 Order Details")

col1, col2, col3 = st.columns(3)

with col1:
    type_of_order = st.selectbox("Order Type",df["type_of_order"].dropna().unique())

with col2:
    pickup_time_minutes = st.selectbox("Pickup Time",sorted(df["pickup_time_minutes"].dropna().unique()))

with col3:
    festival = st.selectbox("Festival",df["festival"].dropna().unique())
    

st.text("🌦️ Weather & Traffic")

col1, col2, col3 = st.columns(3)

with col1:
    weather = st.selectbox("Weather",df["weather"].dropna().unique())

with col2:
    traffic = st.selectbox("Traffic",["low", "medium", "high", "jam"])

with col3:
    city_type = st.selectbox("City Type",["semi-urban", "urban", "metropolitian"])
    

st.text("📍 Location Details")

col1, col2, col3 = st.columns(3)

with col1:
    city_name = st.selectbox("City",sorted(df["city_name"].dropna().unique()))

with col2:
    restaurant_latitude = st.number_input("Restaurant Latitude",value=22.745049,format="%.6f")

with col3:
    restaurant_longitude = st.number_input("Restaurant Longitude",value=75.892471,format="%.6f")

col4, col5, col6 = st.columns(3)

with col4:
    delivery_latitude = st.number_input("Delivery Latitude",value=22.765049,format="%.6f")

with col5:
    delivery_longitude = st.number_input("Delivery Longitude",value=75.912471,format="%.6f")
    

st.text("📅 Date Details")

col1, col2, col3 = st.columns(3)

with col1:
    order_day = st.number_input("Order Day",min_value=1,max_value=31,value=15)

with col2:
    order_month = st.number_input("Order Month",min_value=1,max_value=12,value=3)

with col3:
    order_day_of_week = st.selectbox("Day of Week",df["order_day_of_week"].dropna().unique())
    
    
st.text("⏰ Time Details")

col1, col2, col3 = st.columns(3)

with col1:
    order_time_hour = st.number_input("Order Hour",min_value=0,max_value=23,value=12)

with col2:
    order_time_of_day = st.selectbox("Time of Day",["morning", "afternoon", "evening", "night", "after_midnight"])

with col3:
    is_weekend = st.selectbox("Weekend",[0, 1])


predict = st.button("🛵 Predict Delivery Time")

if predict:
    
    if distance <= 0:
        st.error("Distance should be greater than 0.")
        st.stop()

    input_df = pd.DataFrame({
        "age": [age],
        "ratings": [ratings],
        "restaurant_latitude": [restaurant_latitude],
        "restaurant_longitude": [restaurant_longitude],
        "delivery_latitude": [delivery_latitude],
        "delivery_longitude": [delivery_longitude],
        "weather": [weather],
        "traffic": [traffic],
        "vehicle_condition": [vehicle_condition],
        "type_of_order": [type_of_order],
        "type_of_vehicle": [type_of_vehicle],
        "multiple_deliveries": [multiple_deliveries],
        "festival": [festival],
        "city_type": [city_type],
        "city_name": [city_name],
        "order_day": [order_day],
        "order_month": [order_month],
        "order_day_of_week": [order_day_of_week],
        "is_weekend": [is_weekend],
        "pickup_time_minutes": [pickup_time_minutes],
        "order_time_hour": [order_time_hour],
        "order_time_of_day": [order_time_of_day],
        "distance": [distance]
    })
    
    result = model.predict(input_df)
    
    if len(result) > 0:
        st.markdown(
            f"""
            <div style="
                background-color: rgba(0,0,0,0);
                color: white;
                padding: 15px;
                border-radius: 10px;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                border: 2px solid white;
            ">
                📍 Your order is expected to arrive in {result[0]:.0f} minutes.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.snow()
    else:
        st.error("Prediction failed.")
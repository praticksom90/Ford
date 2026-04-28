import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Ford Car Price Predictor", page_icon="🚗", layout="centered")

# Load model, scaler, and training columns
model = joblib.load("ford_price_model.pkl")
scaler = joblib.load("ford_scaler.pkl")
columns = joblib.load("ford_columns.pkl")

st.title("🚗 Ford Car Price Predictor")
st.write("Enter the Ford car details below to estimate the price.")

# Input fields
model_name = st.selectbox(
    "Model",
    [
        "B-MAX", "C-MAX", "EcoSport", "Edge", "Escort", "Fiesta", "Focus",
        "Fusion", "Galaxy", "Grand C-MAX", "Grand Tourneo Connect", "KA",
        "Ka+", "Kuga", "Mondeo", "Mustang", "Puma", "S-MAX", "Streetka",
        "Tourneo Connect", "Tourneo Custom", "Transit Tourneo"
    ]
)

year = st.slider("Year", 1996, 2023, 2018)

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual", "Semi-Auto"]
)

mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=30000, step=1000)

fuelType = st.selectbox(
    "Fuel Type",
    ["Diesel", "Electric", "Hybrid", "Other", "Petrol"]
)

tax = st.number_input("Tax", min_value=0, max_value=1000, value=150, step=10)
mpg = st.number_input("MPG", min_value=0.0, max_value=250.0, value=45.0, step=0.1)
engineSize = st.number_input("Engine Size", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

if st.button("Predict Price"):

    # Base input
    input_df = pd.DataFrame([{
        "year": year,
        "mileage": mileage,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize,
        "model": model_name,
        "transmission": transmission,
        "fuelType": fuelType
    }])

    # Feature engineering (must match notebook exactly)
    input_df["car_age"] = 2026 - input_df["year"]
    input_df["car_age"] = input_df["car_age"].replace(0, 1)

    input_df["mileage_per_year"] = input_df["mileage"] / input_df["car_age"]
    input_df["tax_per_engine"] = input_df["tax"] / (input_df["engineSize"] + 0.1)

    # One-hot encoding (must match notebook exactly)
    input_df = pd.get_dummies(
        input_df,
        columns=["model", "transmission", "fuelType"],
        drop_first=True
    )

    # Match exact training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale only the numeric columns used during training
    num_cols = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize",
        "car_age",
        "mileage_per_year",
        "tax_per_engine"
    ]

    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Prediction
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: £{prediction:,.0f}")

    # Optional debug
    with st.expander("See processed input"):
        st.dataframe(input_df)
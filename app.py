import streamlit as st
import numpy as np
import joblib

data = joblib.load("model/model.pkl")
model = data["model"]
le = data["label_encoder"]

st.title("Care prediction ")

brand = st.selectbox(
    "Car Brand",
    le.classes_.tolist()
)

year = st.number_input("Year of Purchase", 1992, 2020, 2015)
km_driven = st.slider("Kilometers Driven", 0, 806599, 1)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.radio("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
seller=st.selectbox("Type",["Individual","Dealer","Trustmark Dealer"])

fuel_map = {"Petrol":0,"Diesel":1,"CNG":2,"LPG":3,"Electric":4}
trans_map = {"Manual":0,"Automatic":1}
owner_map = {"First Owner":0,"Second Owner":1,"Third Owner":2,"Fourth & Above Owner":3,"Test Drive Car":4}
seller_map = {"Individual":0,"Dealer":1,"Trustmark Dealer":2}

fuel_val = fuel_map[fuel]
trans_val = trans_map[transmission]
owner_val = owner_map[owner]
seller_val = seller_map[seller]
brand_val = le.transform([brand])[0]

import pandas as pd

columns = ['brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
features = pd.DataFrame([[brand_val, year, km_driven, fuel_val, seller_val, trans_val, owner_val]],columns=columns)

prediction = model.predict(features)

if st.button("Predict Price"):
    st.success(f"💰 Predicted Car Price: ₹ {prediction[0]:,.2f}")

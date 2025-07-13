import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load and clean the dataset
df = pd.read_csv("diamonds.csv")
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

df.columns = df.columns.str.strip()
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
df = df.drop_duplicates()

# Encode categorical features
cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

df['cut'] = df['cut'].map(cut_map)
df['color'] = df['color'].map(color_map)
df['clarity'] = df['clarity'].map(clarity_map)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'diamond_price_model.pkl')

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("diamond_price_model.pkl")

st.set_page_config(page_title="💎 Diamond Price Estimator", page_icon="💍")
st.title("💎 Diamond Price Prediction")
st.markdown("Estimate the price of a diamond based on its characteristics.")

st.sidebar.header("📋 Diamond Attributes")
carat = st.sidebar.slider("Carat", 0.2, 5.01, step=0.01)
cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.sidebar.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.sidebar.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
x = st.sidebar.slider("Length (x)", 0.0, 10.74, step=0.01)
y = st.sidebar.slider("Width (y)", 0.0, 58.9, step=0.01)
z = st.sidebar.slider("Depth (z)", 0.0, 31.8, step=0.01)
depth = st.sidebar.slider("Total Depth %", 43.0, 79.0, step=0.1)
table = st.sidebar.slider("Table Width %", 43.0, 95.0, step=0.1)

if st.sidebar.button("💰 Predict Price"):
    cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
    clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

    input_df = pd.DataFrame([{
        'carat': carat,
        'cut': cut_map[cut],
        'color': color_map[color],
        'clarity': clarity_map[clarity],
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }])

    # Make sure column order matches training
    input_df = input_df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]

    predicted_price = model.predict(input_df)[0]
    st.success(f"🎯 Estimated Diamond Price: *${predicted_price:,.2f}*")
    st.balloons()
else:
    st.info("Click 'Predict Price' to estimate the value.")

st.markdown("---")
st.markdown("Made with ❤ by [Group 4]")

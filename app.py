import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="India Phillips Curve Predictor",
    layout="centered"
)

model = joblib.load("phillips_curve_model.pkl")
//hist = pd.read_csv("historical_data.csv", index_col="year")
# Load without setting the index yet
hist = pd.read_csv("historical_data.csv")

# Print the columns to your app so you can see exactly how they are spelled
st.write("Available columns:", hist.columns.tolist())

st.title("India Inflation Predictor")

st.write(
    "An expectations-augmented Phillips Curve model "
    "(Random Forest, 5-fold CV R² ≈ 0.49) trained on "
    "India's 1991–2025 macro data."
)

st.sidebar.header("Adjust the scenario")

unemployment = st.sidebar.slider(
    "Unemployment (%)",
    10.0,
    30.0,
    float(hist["unemployment"].iloc[-1])
)

expected_inflation = st.sidebar.slider(
    "Expected Inflation (%, last year's rate)",
    0.0,
    15.0,
    float(hist["inflation"].iloc[-1])
)

exchange_rate = st.sidebar.slider(
    "USD/INR Exchange Rate",
    40.0,
    100.0,
    float(hist["exchange_rate"].iloc[-1])
)

oil_price = st.sidebar.slider(
    "Brent Oil Price ($)",
    20.0,
    130.0,
    float(hist["oil_price"].iloc[-1])
)

call_rate = st.sidebar.slider(
    "RBI Call Money Rate (%)",
    3.0,
    13.0,
    float(hist["call_rate"].iloc[-1])
)

input_df = pd.DataFrame(
    [[
        unemployment,
        expected_inflation,
        exchange_rate,
        oil_price,
        call_rate
    ]],
    columns=[
        "unemployment",
        "expected_inflation",
        "exchange_rate",
        "oil_price",
        "call_rate"
    ]
)

prediction = model.predict(input_df)[0]

st.metric(
    "Predicted Inflation",
    f"{prediction:.2f}%"
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    hist.index,
    hist["inflation"],
    marker="o",
    label="Historical Inflation"
)

ax.axhline(
    prediction,
    color="red",
    linestyle="--",
    label="Your Prediction"
)

ax.set_xlabel("Year")
ax.set_ylabel("Inflation (%)")
ax.legend()

st.pyplot(fig)

st.caption(
    "Note: unemployment consistently ranks as the weakest predictor "
    "in this model — the classical Phillips Curve trade-off does not "
    "hold well in India's data. Expected inflation, exchange rate, "
    "and interest rates matter far more."
)

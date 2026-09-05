import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="India Phillips Curve", layout="centered")

bundle = joblib.load('model_bundle.pkl')
hist = pd.read_csv('historical_data.csv')
hist = hist.set_index('year')

st.title("India Inflation: Testing the Phillips Curve")
st.write("Compare three models built on India's 1991–2025 macro data: the classical Phillips Curve, an expectations-augmented version, and a Random Forest.")

tab1, tab2 = st.tabs(["Predictor", "Why the Classic Curve Fails"])

with tab1:
    st.sidebar.header("Adjust the scenario")
    unemployment = st.sidebar.slider("Unemployment (%)", 10.0, 30.0, float(hist['unemployment'].iloc[-1]))
    expected_inflation = st.sidebar.slider("Expected Inflation (%, last year's rate)", 0.0, 15.0, float(hist['inflation'].iloc[-1]))
    exchange_rate = st.sidebar.slider("USD/INR Exchange Rate", 40.0, 100.0, float(hist['exchange_rate'].iloc[-1]))
    oil_price = st.sidebar.slider("Brent Oil Price ($)", 20.0, 130.0, float(hist['oil_price'].iloc[-1]))
    call_rate = st.sidebar.slider("RBI Call Money Rate (%)", 3.0, 13.0, float(hist['call_rate'].iloc[-1]))

    model_choice = st.radio("Model", ["Classic Phillips Curve", "Expectations-Augmented", "Random Forest"], horizontal=True)

    if model_choice == "Classic Phillips Curve":
        c = bundle['basic_coefs']
        prediction = c['const'] + c['unemployment'] * unemployment
        st.caption("Uses only unemployment. In our data this model has R² ≈ 0.003 — essentially no predictive power.")
    elif model_choice == "Expectations-Augmented":
        c = bundle['augmented_coefs']
        prediction = c['const'] + c['unemployment'] * unemployment + c['expected_inflation'] * expected_inflation
        st.caption("Adds last year's inflation. R² ≈ 0.374 — expected inflation does almost all the work here.")
    else:
        input_df = pd.DataFrame([[unemployment, expected_inflation, exchange_rate, oil_price, call_rate]],
                                 columns=['unemployment', 'expected_inflation', 'exchange_rate', 'oil_price', 'call_rate'])
        prediction = bundle['rf_model'].predict(input_df)[0]
        st.caption(f"Uses all 5 features. 5-fold cross-validated R² ≈ {bundle['rf_cv_r2']:.2f} — the best-performing model.")

    st.metric("Predicted Inflation", f"{prediction:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist.index, hist['inflation'], marker='o', label='Historical Inflation', color='steelblue')
    ax.axhline(prediction, color='red', linestyle='--', label='Your Prediction')

    # Annotate real events using actual values already in the data (no invented numbers)
    if 2022 in hist.index:
        ax.annotate('2022: oil price shock\n(Ukraine war)', xy=(2022, hist.loc[2022, 'inflation']),
                    xytext=(2022 - 8, hist.loc[2022, 'inflation'] + 3),
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=8)
    if 2020 in hist.index:
        ax.annotate('2020: COVID', xy=(2020, hist.loc[2020, 'inflation']),
                    xytext=(2020 - 8, hist.loc[2020, 'inflation'] - 2.5),
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=8)

    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation (%)")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Unemployment vs. Inflation: No Clear Trade-off")
    st.write("If the classical Phillips Curve held, this scatter should trend downward-left to upper-right (or vice versa). Instead, it's a flat cloud — this is the actual data behind our R² ≈ 0.003 finding.")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(hist['unemployment'], hist['inflation'], color='steelblue')
    ax2.set_xlabel("Unemployment (%)")
    ax2.set_ylabel("Inflation (%)")
    st.pyplot(fig2)

    st.write("Instead, expected inflation, exchange rate, and interest rates matter far more — consistent with the Friedman-Phelps expectations-augmented view rather than the original Keynesian Phillips Curve.")
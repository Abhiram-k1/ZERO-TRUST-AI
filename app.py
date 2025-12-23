import streamlit as st
import pandas as pd
import joblib

# Load ML artifacts
model = joblib.load("zero_trust_model.pkl")
scaler = joblib.load("scaler.pkl")
le_role = joblib.load("le_role.pkl")
le_device = joblib.load("le_device.pkl")
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="Zero Trust AI", layout="centered")

st.title("🔐 AI-Enabled Zero Trust Access Control")

st.write("Enter access request details to evaluate risk using the trained ML model.")

# ---------- INPUTS ----------
role = st.selectbox("Role", ["admin", "employee", "user"])
device_trusted = st.selectbox("Device Trusted", ["yes", "no"])
fail_count = st.number_input("Failed Login Count", min_value=0, step=1)
req_count = st.number_input("Request Count", min_value=0, step=1)
ip_risk_score = st.slider("IP Risk Score", 0, 100, 25)
hour = st.slider("Access Hour", 0, 23, 12)
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)

# ---------- PREDICTION ----------
if st.button("Request Access"):
    input_df = pd.DataFrame([{
        "role": role,
        "device_trusted": device_trusted,
        "fail_count": fail_count,
        "req_count": req_count,
        "ip_risk_score": ip_risk_score,
        "hour": hour,
        "day": day,
        "month": month
    }])

    # Encode categorical variables
    input_df["role"] = le_role.transform(input_df["role"])
    input_df["device_trusted"] = le_device.transform(input_df["device_trusted"])

    # Reorder features
    input_df = input_df[feature_order]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    risk_prob = model.predict_proba(input_scaled)[0][1]
    decision = "DENY" if risk_prob >= 0.65 else "ALLOW"

    st.subheader("Assessment Result")
    st.metric("Risk Probability", f"{risk_prob:.2f}")

    if decision == "ALLOW":
        st.success("✅ ACCESS ALLOWED")
    else:
        st.error("❌ ACCESS DENIED")

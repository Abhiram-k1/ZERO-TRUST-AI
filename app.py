import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained artifacts
# -----------------------------
model = joblib.load("zero_trust_model.pkl")
scaler = joblib.load("scaler.pkl")
le_role = joblib.load("le_role.pkl")
le_device = joblib.load("le_device.pkl")
feature_order = joblib.load("feature_order.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Zero Trust AI", layout="centered")

st.title("AI-Enabled Zero Trust Access Control")
st.write("Enter request details. Access risk is evaluated using multi-factor Zero Trust logic.")

# -----------------------------
# User Inputs
# -----------------------------
role = st.selectbox("Role", ["admin", "employee", "user"])
fail_count = st.number_input("Failed Login Count", min_value=0, step=1)
req_count = st.number_input("Request Count", min_value=0, step=1)
ip_risk_score = st.slider("IP Risk Score", 0, 100, 25)
hour = st.slider("Access Hour", 0, 23, 12)
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)

# -----------------------------
# Helper: infer device trust
# -----------------------------
def infer_device_trust(fail_count, ip_risk_score):
    """
    Device trust is inferred implicitly instead of being user-provided.
    """
    if fail_count >= 3 or ip_risk_score > 60:
        return "no"
    return "yes"


# -----------------------------
# Composite risk logic
# -----------------------------
def compute_final_risk(
    role,
    fail_count,
    req_count,
    ip_risk_score,
    hour,
    model_prob
):
    # Identity risk
    role_risk_map = {
        "admin": 0.1,
        "employee": 0.3,
        "user": 0.6
    }
    identity_risk = role_risk_map.get(role, 0.5)

    # Authentication behavior
    auth_risk = min(fail_count / 5, 1.0)

    # Request behavior
    behavior_risk = min(req_count / 20, 1.0)

    # Network risk
    network_risk = ip_risk_score / 100

    # Contextual (time-based) risk
    if hour < 6 or hour > 22:
        context_risk = 0.7
    else:
        context_risk = 0.2

    # Weighted aggregation
    final_risk = (
        0.20 * identity_risk +
        0.20 * auth_risk +
        0.20 * behavior_risk +
        0.20 * network_risk +
        0.20 * model_prob
    )

    return round(final_risk * 100, 2)


def zero_trust_decision(score):
    if score < 30:
        return "ALLOW"
    elif score < 60:
        return "ALLOW (MONITOR)"
    elif score < 80:
        return "DENY"
    else:
        return "DENY – HIGH RISK"


# -----------------------------
# Prediction
# -----------------------------
if st.button("Request Access"):

    # Infer device trust internally
    device_trusted = infer_device_trust(fail_count, ip_risk_score)

    # Build input dataframe
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

    # Encode categorical fields
    input_df["role"] = le_role.transform(input_df["role"])
    input_df["device_trusted"] = le_device.transform(input_df["device_trusted"])

    # Match training feature order
    input_df = input_df[feature_order]

    # Scale
    input_scaled = scaler.transform(input_df)

    # ML probability
    model_prob = model.predict_proba(input_scaled)[0][1]

    # Final composite risk
    final_risk = compute_final_risk(
        role=role,
        fail_count=fail_count,
        req_count=req_count,
        ip_risk_score=ip_risk_score,
        hour=hour,
        model_prob=model_prob
    )

    decision = zero_trust_decision(final_risk)

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("Assessment Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("ML Risk Probability", f"{model_prob:.2f}")
    col2.metric("Final Risk Score", f"{final_risk:.2f} / 100")
    col3.metric("Decision", decision)

    if "ALLOW" in decision:
        st.success(f"✅ ACCESS {decision}")
    else:
        st.error(f"❌ ACCESS {decision}")

    st.caption(
        "Risk score is computed using identity, behavior, network, contextual factors "
        "and ML-based probability, following Zero Trust principles."
    )

# --- START OF FILE checkfraud_enhanced_rules.py ---

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import math # Needed for isclose

# --- 1. Page Setup and Title ---
st.set_page_config(page_title="Fraud & Anomaly Detection Tool", layout="wide")
st.title("Fraud & Anomaly Detection Tool 🕵️")
st.markdown("""
This tool uses a Decision Tree model combined with **enhanced logical rule checks**
to predict potential fraud or data anomalies based on transaction details.
*(Model Accuracy on Test Set: ~99.97%)*
""")
st.divider()

# --- 2. User Input Section ---
st.header("Enter Transaction Details:")

col1, col2 = st.columns(2)

with col1:
    transcationtype_str = st.selectbox(
        "Select Transaction Type:",
        options=["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
        index=3 # Default to TRANSFER for easy testing
    )
    # Use a default amount that might trigger rules
    amount = st.number_input("Transaction Amount:", min_value=0.0, value=1000.0, step=100.0, format="%.2f")

with col2:
    oldbalanceOrg = st.number_input("Old Balance (Origin Account):", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
    # Example default to test Rule 1/3: 5000 - 1000 != 16000, and 16000 > 5000
    newbalanceOrg = st.number_input("New Balance (Origin Account):", min_value=0.0, value=16000.0, step=100.0, format="%.2f")


st.divider()

# --- 3. Input Processing ---

type_mapping = {
    "CASH_OUT": 1,
    "PAYMENT": 2,
    "CASH_IN": 3,
    "TRANSFER": 4,
    "DEBIT": 5
}
type_numeric = type_mapping[transcationtype_str]

# Prepare input for the ML model (still needed even if rules override)
input_features = np.array([[
    float(type_numeric),
    float(amount),
    float(oldbalanceOrg),
    float(newbalanceOrg)
]])

# --- 4. Model Loading ---
try:
    with open('decision_tree_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: The model file ('decision_tree_model.pkl') was not found. Make sure it's in the same directory as the script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- 5. Prediction, Comprehensive Rule Check, and Result Display ---
button = st.button("Check Transaction for Fraud / Anomalies", type="primary")

if button:
    try:
        # --- Get Model Prediction ---
        model_prediction = loaded_model.predict(input_features)[0]

        # --- Initialize Variables for Final Verdict & Rule Tracking ---
        final_verdict = model_prediction # Start with the model's opinion
        triggered_rules = [] # Store descriptions of triggered rules
        # Tolerance for balance checks (e.g., allow for $1 fee/rounding)
        # Use math.isclose relative tolerance for better handling of large/small numbers
        # abs_tol still useful for near-zero checks
        rel_tolerance = 0.01 # e.g., 1% tolerance
        abs_tolerance = 1.01 # Absolute tolerance minimum

        # --- Apply Comprehensive Logical Rule Checks ---

        # --- Checks for OUTGOING Transactions ---
        if type_numeric in [1, 2, 4, 5]: # CASH_OUT, PAYMENT, TRANSFER, DEBIT
            # Rule 1: Balance Increased After Outgoing Tx
            # Use a small absolute tolerance to avoid flagging tiny positive changes
            if newbalanceOrg > oldbalanceOrg + abs_tolerance:
                triggered_rules.append(f"Rule 1 Violation: Origin balance increased significantly (from {oldbalanceOrg:.2f} to {newbalanceOrg:.2f}) after an outgoing transaction.")

            # Rule 2: Insufficient Funds (Reported)
            # Check if amount is strictly greater than starting balance
            if amount > oldbalanceOrg:
                 triggered_rules.append(f"Rule 2 Violation: Transaction amount ({amount:.2f}) exceeds starting origin balance ({oldbalanceOrg:.2f}).")

            # Rule 3: Incorrect Balance Update (Outgoing)
            expected_change = -amount # Expect balance to decrease by amount
            actual_change = newbalanceOrg - oldbalanceOrg
            # Check if the actual change is NOT close to the expected change
            if not math.isclose(actual_change, expected_change, rel_tol=rel_tolerance, abs_tol=abs_tolerance):
                # Avoid double-reporting if Rule 1 already caught a positive increase
                 is_already_flagged_increase = any("Rule 1" in rule for rule in triggered_rules)
                 if not is_already_flagged_increase:
                     triggered_rules.append(f"Rule 3 Violation: Origin balance change is incorrect. Expected change ~{expected_change:.2f}, Actual change: {actual_change:.2f} (from {oldbalanceOrg:.2f} to {newbalanceOrg:.2f}).")

        # --- Checks for INCOMING Transactions ---
        elif type_numeric == 3: # CASH_IN
             # Rule 4: Balance Decreased or Unchanged After Incoming Tx
             if newbalanceOrg <= oldbalanceOrg:
                 triggered_rules.append(f"Rule 4 Violation: Origin balance did not increase (from {oldbalanceOrg:.2f} to {newbalanceOrg:.2f}) after an incoming transaction (CASH_IN).")

             # Rule 5: Incorrect Balance Update (Incoming)
             expected_change = amount # Expect balance to increase by amount
             actual_change = newbalanceOrg - oldbalanceOrg
             # Check if the actual change is NOT close to the expected change
             if not math.isclose(actual_change, expected_change, rel_tol=rel_tolerance, abs_tol=abs_tolerance):
                 # Avoid double-reporting if Rule 4 already caught decrease/no change
                 is_already_flagged_decrease = any("Rule 4" in rule for rule in triggered_rules)
                 if not is_already_flagged_decrease:
                    triggered_rules.append(f"Rule 5 Violation: Origin balance change is incorrect. Expected change ~{expected_change:.2f}, Actual change: {actual_change:.2f} (from {oldbalanceOrg:.2f} to {newbalanceOrg:.2f}).")


        # --- Determine Final Verdict based on Rules ---
        is_logically_suspicious = len(triggered_rules) > 0
        if is_logically_suspicious:
            final_verdict = "Fraud" # Logical rules override model's 'No Fraud'

        # --- Display Results ---
        with st.spinner('Analyzing Transaction...'):
            time.sleep(1) # Simulate processing time

        st.subheader("Analysis Result:")

        if is_logically_suspicious:
            st.warning("⚠️ Logical Anomaly Detected! Flagged as SUSPICIOUS/FRAUD based on rule violation(s):", icon="⚠️")
            st.markdown("---")
            for i, rule in enumerate(triggered_rules):
                st.markdown(f"**{i+1}. {rule}**")
            st.markdown("---")
            st.info("Note: Even if the ML model predicted 'No Fraud', these logical inconsistencies require attention.")

        elif final_verdict == "Fraud":
             st.error("🚨 Transaction predicted as FRAUDULENT by the ML model.", icon="🚨")
             st.info("Reasoning: The ML model identified patterns (based on type, amount, balances) consistent with known fraudulent activities in its training data. It also passed basic logical checks.")
        elif final_verdict == "No Fraud":
             st.success("✅ Transaction predicted as NOT Fraudulent by the ML model and passed all logical checks.", icon="✅")
        else:
             # Fallback for unexpected prediction strings
             st.warning(f"Unexpected prediction output from model: {final_verdict}")

    except Exception as e:
        st.error(f"An error occurred during prediction or rule check: {e}")
        import traceback
        st.error(traceback.format_exc()) # Print detailed traceback for debugging

st.divider()

# --- 6. Additional Information Section ---
st.header("Understanding Transaction Types")
data = {
    "Original Term": ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
    "Common Meaning / Alternative": [
        "Withdrawing cash from an account (e.g., ATM)",
        "Paying a bill or merchant",
        "Depositing cash into an account",
        "Moving money electronically between accounts",
        "A charge or withdrawal from an account (often POS)"
    ]
}
df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

st.caption("Disclaimer: This tool provides predictions based on an ML model and rule-based checks. It's for informational purposes and doesn't guarantee the actual status. Logical anomalies might indicate data errors or fraud.")

# --- END OF FILE checkfraud_enhanced_rules.py ---
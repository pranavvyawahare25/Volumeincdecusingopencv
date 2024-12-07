import streamlit as st
import pandas as pd
import numpy as np

# Set the title and description
st.title("Fraud Detection System (Preview)")
st.write("Please enter the transaction details below to see the predicted fraud status and explanation.")

# Step 1: Input Transaction Details (using relevant features)
st.header("Enter Transaction Details")
transaction_id = st.text_input("Transaction ID")
account_number = st.text_input("Account Number")
name_sender = st.text_input("Name (Sender)")
date = st.date_input("Transaction Date")
time = st.time_input("Transaction Time")
transaction_type = st.selectbox("Transaction Type", ["Credit", "Debit"])
amount = st.number_input("Amount (INR)", min_value=0.0, step=0.01)
before_balance = st.number_input("Before Balance (INR)", min_value=0.0, step=0.01)
after_balance = st.number_input("After Balance (INR)", min_value=0.0, step=0.01)
location = st.text_input("Transaction Location")
merchant_category = st.selectbox("Merchant Category", ["Retail", "Food", "Entertainment", "Others"])
recipient_name = st.text_input("Recipient Name")
recipient_account_id = st.text_input("Recipient Account/ID")
ip_address = st.text_input("IP Address")
device_info = st.text_input("Device Info")
payment_method = st.selectbox("Payment Method", ["Card", "UPI", "Net Banking", "Wallet"])
card_number = st.text_input("Card Number (masked)", type="password")
authentication_method = st.selectbox("Authentication Method", ["OTP", "Biometric", "PIN", "None"])
payment_status = st.selectbox("Payment Status", ["Success", "Failure"])
fraud_indicator = st.selectbox("Fraud Indicator (Simulated)", ["Yes", "No"])

# Initialize fraud_risk_score outside the button's scope
fraud_risk_score = 0

# Step 2: Simulate Fraud Detection Logic (Simple Rule-Based Approach)
if st.button("Check Fraud Status"):
    # Example rule-based fraud detection logic (can be adjusted)
    if amount > 50000:
        fraud_risk_score += 1  # High amount
    if location.lower() not in ['new york', 'los angeles', 'mumbai']:  # Unusual location
        fraud_risk_score += 1
    if payment_status == "Failure":
        fraud_risk_score += 1  # Failed transactions might be fraud
    if authentication_method == "None":
        fraud_risk_score += 1  # Lack of authentication could signal fraud
    if fraud_indicator == "Yes":
        fraud_risk_score += 1  # Simulate fraud indicator being "Yes"

    # Predict based on fraud risk score
    if fraud_risk_score >= 2:
        st.write("### Fraud Prediction: **Fraudulent**")
        st.write("The transaction is flagged as fraudulent due to suspicious patterns.")
    else:
        st.write("### Fraud Prediction: **Safe**")
        st.write("The transaction appears to be legitimate based on the provided details.")

# Step 3: Provide Model Explanation
st.header("Fraud Detection Model Explanation")
if fraud_risk_score >= 2:
    explanation = """
    - The transaction amount exceeds typical thresholds, signaling possible fraud.
    - The transaction location is unusual for the user or merchant.
    - The payment failed, suggesting a potential fraud attempt.
    - Lack of authentication method (such as OTP or biometric) raises red flags.
    - Fraud indicator is marked as "Yes" based on historical patterns.
    """
else:
    explanation = """
    - The transaction amount is within a typical range for this account.
    - The location and payment method appear consistent with user behavior.
    - Successful transaction with proper authentication and no fraud indicators.
    """

st.write("### Explanation for Fraud Prediction:")
st.write(explanation)

# Optional: Add flowchart to illustrate the process
st.header("Fraud Detection Process Flow")
st.image("path_to_your_flowchart_image.png", caption="Fraud Detection Process Flowchart")  # Replace with your actual flowchart image

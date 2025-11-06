import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mysql.connector
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="Financial Penetration Predictor",
    page_icon="💰",
    layout="wide"
)

# -------------------------------------------------------
# DATABASE CONNECTION (MySQL via XAMPP)
# -------------------------------------------------------
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",          # default XAMPP user
        password="",          # leave blank if you haven't set a password
        database="financial_db"
    )

# Function to insert data into MySQL
def insert_record(data):
    conn = create_connection()
    cursor = conn.cursor()
    query = """INSERT INTO predictions 
               (location_type, cellphone_access, household_size, educational_level, job_type, 
                probability, prediction, confidence, timestamp)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    cursor.execute(query, data)
    conn.commit()
    cursor.close()
    conn.close()

# Function to fetch stored data
def fetch_history():
    conn = create_connection()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
loaded_model = joblib.load('C:/Users/CASOR UNN MEDIA/Desktop/IT IS WELL/Dona Project/financial.pkl')

# -------------------------------------------------------
# DEFINE FEATURES
# -------------------------------------------------------
feature_names = ['location_type', 'cellphone_access', 'household_size', 'educational_level', 'job_type']
categorical_features = feature_names

# -------------------------------------------------------
# DEFINE CATEGORIES
# -------------------------------------------------------
dummy_categories = [
    ['Urban', 'Rural'],
    ['Yes', 'No'],
    ['Low', 'Average', 'High'],
    ['Low Formal Education', 'Other', 'Primary Education', 'Secondary Education', 'Tertiary Education', 'Vocational/Specialize Training'],
    ['Not Specified', 'Farming and fishing', 'Formally Government Employment', 'Formally Capitalist Employment',
     'Government Dependent', 'Informal Employment', 'No Employment', 'Other Sources', 'Remittance Dependent', 'Self Employment']
]

# -------------------------------------------------------
# ENCODER SETUP
# -------------------------------------------------------
encoder = OrdinalEncoder(categories=dummy_categories, handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(pd.DataFrame([['Urban', 'Yes', 'Low', 'Low Formal Education', 'Not Specified']], columns=feature_names))

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.title("💰 AI-Driven System For Financial Institution Penetration ")
st.write("Analyze and predict the potential of financial institution penetration in any region.")

# -------------------------------------------------------
# VISUAL OVERVIEW
# -------------------------------------------------------
st.markdown("### 📊 Regional Financial Insights")
col1, col2 = st.columns(2)

sample_data = pd.DataFrame({
    'Region Type': ['Urban', 'Rural'],
    'Financial Access (%)': [82, 46]
})

with col1:
    st.bar_chart(sample_data.set_index('Region Type'))

with col2:
    fig, ax = plt.subplots()
    ax.pie(sample_data['Financial Access (%)'], labels=sample_data['Region Type'], autopct='%1.1f%%', startangle=90)
    ax.set_title("Urban vs Rural Financial Access")
    st.pyplot(fig)

st.markdown("---")

# -------------------------------------------------------
# INPUT FORM
# -------------------------------------------------------
st.markdown("### 🧾 Predict Regional Financial Penetration")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        population = st.selectbox("Population Size", ["Low", "Average", "High"])
        employment = st.selectbox("Location Type", ["Urban", "Rural"])
        education = st.selectbox("Education Level", dummy_categories[3])
    with col2:
        occupation = st.selectbox("Prevailant Regional Occupation", dummy_categories[4])
        mobile = st.selectbox("Internet/Mobile Access", ["Yes", "No"])

    submitted = st.form_submit_button("🔮 Predict Bank Penetration")

    if submitted:
        # Prepare input
        input_data = pd.DataFrame({
            'location_type': [employment],
            'cellphone_access': [mobile],
            'household_size': [population],
            'educational_level': [education],
            'job_type': [occupation]
        })[feature_names]

        # Encode and predict
        input_encoded = encoder.transform(input_data[categorical_features])
        prob = loaded_model.predict_proba(input_encoded)[0][1]
        threshold = 0.4
        prediction = "Financial Services Will Thrive" if prob >= threshold else "Financial Services Might Not Penetrate"
        confidence = 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'

        # Save record to MySQL
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_record((employment, mobile, population, education, occupation, float(prob), prediction, confidence, timestamp))

        # Display Results
        st.markdown("### 🎯 Prediction Result")
        st.metric(label="Prediction", value=prediction)
        st.progress(prob)
        st.write(f"**Probability of Financial Institution Penetration:** {prob:.2%}")
        st.write(f"**Model Confidence:** {'🟢 High' if prob > 0.7 else '🟡 Medium' if prob > 0.4 else '🔴 Low'}")

        if prob > 0.7:
            st.balloons()

        # Recommendations
        st.markdown("### 💡 Recommendations")
        if prediction == "Financial Services Might Not Penetrate":
            if employment == "Rural":
                st.info("🏦 Improve mobile banking access or create local banking points in rural communities.")
            if education == "Low Formal Education":
                st.info("📘 Invest and Implement financial literacy programs for low-education populations.")
            if mobile == "No":
                st.info("📱 Expand mobile Money and internet coverage.")
        else:
            st.success("✅ Region already shows high potential for financial services.  Consider introducing advanced banking services like credit, loans, or digital payments.")

# -------------------------------------------------------
# SHOW HISTORY FROM DATABASE
# -------------------------------------------------------
st.markdown("---")
st.subheader("📋 Stored Predictions (MySQL)")

try:
    df_history = fetch_history()
    if not df_history.empty:
        st.dataframe(df_history)
        st.bar_chart(df_history['probability'])
    else:
        st.info("No prediction data yet — make your first prediction above.")
except Exception as e:
    st.error(f"Error fetching data: {e}")

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.header("📚 About the App")
st.sidebar.info("""
This predictive tool helps policymakers and financial institutions understand
how regional demographic characteristics influence **financial penetration**.

It uses a trained ML model to predict how well financial institutions
might penetrate a region, based on its social and economic attributes.

**Key Inputs:**
- Population Size  
- Education Level  
- Employment Rate  
- Mobile Access  
- Prevailing Regional Occupation
""")

st.sidebar.markdown("---")
st.sidebar.header("💡 Quick Facts")
st.sidebar.success("""
- Urban areas show ~82% bank access.  
- Rural access averages 46%.  
- Education & mobile access boost banking likelihood.
""")

st.sidebar.markdown("---")
st.sidebar.caption("💻 Developed by **Uzor Donatus** | Powered by **Streamlit + MySQL + scikit-learn** ⚙️")
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import plotly.express as px
import hashlib
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Default admin login
# -------------------------
DEFAULT_ADMIN_USER = "uzordona1999@gmail.com"
DEFAULT_ADMIN_PASS = "admin123"
DEFAULT_ADMIN_ROLE = "admin"
USERS_FILE = "users.csv"

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed: str):
    return hash_password(password) == hashed

def load_users():
    if not os.path.exists(USERS_FILE):
        return pd.DataFrame(columns=["username", "password", "role"])
    try:
        return pd.read_csv(USERS_FILE, usecols=["username", "password", "role"], on_bad_lines='skip')
    except:
        return pd.DataFrame(columns=["username", "password", "role"])

def login(username, password):
    # Check default admin first
    if username == DEFAULT_ADMIN_USER and password == DEFAULT_ADMIN_PASS:
        return True, DEFAULT_ADMIN_ROLE
    # Check CSV users
    df = load_users()
    user = df[df["username"] == username]
    if user.empty:
        return False, None
    if not verify_password(password, user.iloc[0]["password"]):
        return False, None
    return True, user.iloc[0]["role"]

def register_user(username, password, role="user"):
    df = load_users()
    if username in df["username"].tolist():
        return False
    hashed = hash_password(password)
    df.loc[len(df)] = [username, hashed, role]
    df.to_csv(USERS_FILE, index=False)
    return True

# -------------------------
# Theme toggle
# -------------------------
def set_theme(dark_mode: bool):
    if dark_mode:
        st.markdown("""
        <style>
        .reportview-container { background: #0e1117; color: #FFFFFF; }
        .stButton>button { background-color: #2b6cb0; color: white; }
        .stMetric { color: #FFFFFF; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .reportview-container { background: #FFFFFF; color: #111111; }
        .stButton>button { background-color: #0d6efd; color: white; }
        </style>
        """, unsafe_allow_html=True)

# -------------------------
# Load model
# -------------------------
def load_model(path="financial.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        return None

# -------------------------
# Encoder setup
# -------------------------
feature_names = ['location_type', 'cellphone_access', 'household_size', 'educational_level', 'job_type']
dummy_categories = [
    ['Urban', 'Rural'],
    ['Yes', 'No'],
    ['Low', 'Average', 'High'],
    ['Low Formal Education', 'Other', 'Primary Education', 'Secondary Education', 'Tertiary Education', 'Vocational/Specialize Training'],
    ['Not Specified', 'Farming and fishing', 'Formally Government Employment', 'Formally Capitalist Employment',
     'Government Dependent', 'Informal Employment', 'No Employment', 'Other Sources', 'Remittance Dependent', 'Self Employment']
]
encoder = OrdinalEncoder(categories=dummy_categories, handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(pd.DataFrame([['Urban', 'Yes', 'Low', 'Low Formal Education', 'Not Specified']], columns=feature_names))

# -------------------------
# App layout
# -------------------------
st.set_page_config(page_title="Financial Penetration Suite", page_icon="💰", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Predict", "Dashboard", "Model Comparison", "History", "Admin"])
dark = st.sidebar.checkbox("Dark mode", value=False)
set_theme(dark)

# -------------------------
# Authentication status
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

if not st.session_state.logged_in:
    st.sidebar.subheader("🔐 Login")
    u = st.sidebar.text_input("Username", key="login_user")
    p = st.sidebar.text_input("Password", type="password", key="login_pass")
    if st.sidebar.button("Login", key="login_btn"):
        success, role = login(u, p)
        if success:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.role = role
            st.success(f"Welcome {u}! Role: {role}")
        else:
            st.sidebar.error("Invalid username or password.")
else:
    st.sidebar.info(f"Logged in as: {st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None

is_admin = st.session_state.logged_in and st.session_state.role == "admin"

# -------------------------
# Save prediction
# -------------------------
def save_prediction(record, csv_file="predictions.csv"):
    df_record = pd.DataFrame([record])
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_all = pd.concat([df_existing, df_record], ignore_index=True)
    else:
        df_all = df_record
    df_all.to_csv(csv_file, index=False)
    return df_all

# -------------------------
# Load model for predictions
# -------------------------
loaded_model = load_model()

# -------------------------
# Pages
# -------------------------
if page == "Home":
    st.title("💰 AI-Driven System For Financial Institution Penetration")
    st.write("Welcome — use the sidebar to navigate.")
    st.markdown("---")
    st.info("This application predicts and analyzes regional financial institution penetration. It includes dashboards, mapping, model comparison, and admin retraining tools.")

    # -------------------------
    # Visual Overview
    # -------------------------
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

elif page == "Predict":
    st.title("🔮 Predict Regional Financial Penetration")
    if not st.session_state.logged_in:
        st.warning("Please log in to access prediction features.")
    else:
        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            region_name = st.text_input("Region Name", placeholder="Enter the region name, e.g., Nsukka, Enugu North...")
            with col1:
                population = st.selectbox("Population Size", ["Low", "Average", "High"])
                employment = st.selectbox("Location Type", ["Urban", "Rural"])
                education = st.selectbox("Education Level", dummy_categories[3])
            with col2:
                occupation = st.selectbox("Prevailing Regional Occupation", dummy_categories[4])
                mobile = st.selectbox("Internet/Mobile Access", ["Yes", "No"])
            submitted = st.form_submit_button("🔮 Predict")

        if submitted:
            input_data = pd.DataFrame({
                'location_type': [employment],
                'cellphone_access': [mobile],
                'household_size': [population],
                'educational_level': [education],
                'job_type': [occupation]
            })[feature_names]

            input_encoded = encoder.transform(input_data)
            if loaded_model is None:
                st.error("Model not found. Please upload/retrain via Admin page.")
            else:
                prob = loaded_model.predict_proba(input_encoded)[0][1]
                threshold = 0.4
                prediction = "Financial Services Will Thrive" if prob >= threshold else "Financial Services Might Not Penetrate"

                # -------------------------
                # Display results & recommendations
                # -------------------------
                st.markdown("### 🎯 Prediction Result")
                st.metric(label="Prediction", value=prediction)
                st.progress(prob)
                st.write(f"**Region Name:** {region_name if region_name else 'Unnamed Region'}")
                st.write(f"**Probability of Financial Institution Penetration:** {prob:.2%}")
                st.write(f"**Model Confidence:** {'🟢 High' if prob > 0.7 else '🟡 Medium' if prob > 0.4 else '🔴 Low'}")

                if prob > 0.7:
                    st.balloons()

                st.markdown("### 💡 Recommendations")
                if prediction == "Financial Services Might Not Penetrate":
                    if employment == "Rural":
                        st.info("🏦 Improve mobile banking access or create local banking points in rural communities.")
                    if education == "Low Formal Education":
                        st.info("📘 Implement financial literacy programs for low-education populations.")
                    if mobile == "No":
                        st.info("📱 Expand mobile money and internet coverage.")
                else:
                    st.success("✅ Region already shows high potential for financial services. Consider introducing advanced banking services like credit, loans, or digital payments.")

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                record = {
                    "region_name": region_name or "Unnamed Region",
                    "location_type": employment,
                    "cellphone_access": mobile,
                    "household_size": population,
                    "educational_level": education,
                    "job_type": occupation,
                    "probability": float(prob),
                    "prediction": prediction,
                    "confidence": 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low',
                    "timestamp": timestamp
                }
                save_prediction(record)

elif page == "Dashboard":
    st.title("📊 Dashboard")
    if not st.session_state.logged_in:
        st.warning("Please log in to view dashboards.")
    else:
        if os.path.exists("predictions.csv"):
            df = pd.read_csv("predictions.csv")
            st.markdown("### Prediction Summary")
            st.dataframe(df.tail(10))

            # Fix histogram: numeric values for bar_chart
            hist_counts = pd.cut(df['probability'], bins=10).value_counts().sort_index()
            st.bar_chart(hist_counts.values)

            # Pie chart
            fig = px.pie(df, names='prediction', title="Prediction Distribution")
            st.plotly_chart(fig)
        else:
            st.info("No prediction data available.")

elif page == "Model Comparison":
    st.title("⚖️ Model Comparison")
    if not st.session_state.logged_in:
        st.warning("Please log in to access model comparison.")
    else:
        st.markdown("### Model Performance Metrics")
        metrics = {
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [0.82, 0.89],
            "Precision": [0.8, 0.87],
            "Recall": [0.78, 0.88],
            "F1-Score": [0.79, 0.87]
        }
        df_metrics = pd.DataFrame(metrics)
        st.dataframe(df_metrics)
        st.bar_chart(df_metrics.set_index("Model")["Accuracy"])

elif page == "History":
    st.title("📜 Prediction History")
    if not st.session_state.logged_in:
        st.warning("Please log in to view prediction history.")
    else:
        if os.path.exists("predictions.csv"):
            df = pd.read_csv("predictions.csv")
            st.dataframe(df)
        else:
            st.info("No prediction history available.")

elif page == "Admin":
    st.title("🔧 Admin Panel")
    if not is_admin:
        st.warning("You must log in as admin to access admin actions.")
    else:
        st.success("Admin access granted.")
        st.markdown("### 🛠️ Register New User")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["user", "admin"])
        if st.button("Create Account"):
            if register_user(new_user, new_pass, role):
                st.success(f"User '{new_user}' registered successfully!")
            else:
                st.error("Username already exists!")

        st.markdown("---")
        st.markdown("### Model Upload")
        uploaded = st.file_uploader("Upload trained model (.pkl)", type=["pkl"])
        if uploaded:
            with open("financial.pkl", "wb") as f:
                f.write(uploaded.getbuffer())
            st.success("Uploaded financial.pkl")

        st.markdown("---")
        st.markdown("### Maintenance")
        if st.button("🧹 Clear predictions.csv"):
            if os.path.exists("predictions.csv"):
                os.remove("predictions.csv")
                st.success("predictions.csv removed.")
            else:
                st.info("predictions.csv not found.")

st.sidebar.caption("💻 Developed by **Uzor Donatus** | Powered by **Streamlit + scikit-learn** ⚙️")
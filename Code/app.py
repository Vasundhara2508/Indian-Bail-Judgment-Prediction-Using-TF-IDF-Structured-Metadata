# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="JusticeLens: Bail Prediction", layout="wide")

# ------------------------------------------------------------
# 📂 LOAD & PREPROCESS DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("indian_bail_judgments.csv")

    # Clean and encode labels
    df['bail_outcome'] = df['bail_outcome'].astype(str).str.strip().str.lower()
    df['label'] = df['bail_outcome'].apply(lambda x: 1 if x == 'granted' else 0)

    # Keep Male/Female only for simplicity
    df = df[df['accused_gender'].isin(['Male', 'Female'])].copy()
    df['gender_encoded'] = df['accused_gender'].map({'Male': 0, 'Female': 1})

    # Crime type
    df['crime_type'] = df['crime_type'].fillna('Unknown')
    df['crime_type_encoded'] = df['crime_type'].astype('category').cat.codes

    # Prior cases
    df['prior_cases'] = df['prior_cases'].replace({'Yes': 1, 'No': 0, 'Unknown': 0})
    df['prior_cases'] = df['prior_cases'].fillna(0).astype(int)

    # Derived features
    df['risk_score'] = df['prior_cases'] * df['gender_encoded']
    df['gender_crime_interaction'] = df['gender_encoded'] * df['crime_type_encoded']

    # Create text column for TF-IDF
    df['summary'] = df.get('summary', pd.Series([""] * len(df)))
    return df

df = load_data()

# ------------------------------------------------------------
# 🔹 Prepare TF-IDF + feature matrix (same as training pipeline)
# ------------------------------------------------------------
@st.cache_resource
def prepare_features_and_train():
    # TF-IDF vectorizer trained on summary
    tfidf = TfidfVectorizer(max_features=500)
    text_matrix = tfidf.fit_transform(df['summary'].fillna("")).toarray()

    structured = df[['prior_cases', 'gender_encoded', 'crime_type_encoded',
                     'risk_score', 'gender_crime_interaction']].values
    X = np.hstack([structured, text_matrix])
    y = df['label'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost model
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        colsample_bytree=0.7,
        gamma=1,
        learning_rate=0.1,
        max_depth=7,
        min_child_weight=1,
        n_estimators=300,
        subsample=0.9
    )
    model.fit(X_train, y_train)

    return tfidf, model, X_train, X_test, y_train, y_test

tfidf, model, X_train, X_test, y_train, y_test = prepare_features_and_train()

# ------------------------------------------------------------
# 🧭 Streamlit UI
# ------------------------------------------------------------
st.title("⚖️ JusticeLens — Bail Judgment Prediction")
st.write("Enter case details (structured + free text) and get bail prediction.")

st.markdown("---")
# User inputs
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Accused Gender", ["Male", "Female","other"])
with col2:
    prior_cases = st.selectbox("Any Prior Cases?", ["No", "Yes"])
with col3:
    crime_type = st.selectbox("Crime Type", sorted(df['crime_type'].unique().tolist()))

summary = st.text_area("Case Summary / Facts", height=150,
                       placeholder="Enter case summary (facts, circumstances, mitigation, etc.)")

# Map crime_type to category code
crime_cat = list(df['crime_type'].astype('category').cat.categories)
crime_encoded = int(np.where(np.array(crime_cat) == crime_type)[0]) if crime_type in crime_cat else 0

# Button to run prediction
if st.button("🔮 Predict Bail Outcome"):
    if summary.strip() == "":
        st.warning("Please enter a case summary to analyze.")
    else:
        # Build structured vector
        gender_encoded = 1 if gender == "Female" else 0
        prior_encoded = 1 if prior_cases == "Yes" else 0
        risk_score = prior_encoded * gender_encoded
        gender_crime_interaction = gender_encoded * crime_encoded

        structured_input = np.array([[prior_encoded, gender_encoded, crime_encoded,
                                      risk_score, gender_crime_interaction]])
        # TF-IDF vector for summary
        text_vec = tfidf.transform([summary]).toarray()
        user_input = np.hstack([structured_input, text_vec])  # shape (1, n_features)

        # Predict
        prob = model.predict_proba(user_input)[0][1]
        pred_label = "✅ Bail Granted" if prob > 0.8 else "❌ Bail Denied"

        st.subheader("Prediction")
        st.write(f"**Outcome:** {pred_label}")
        st.write(f"**Confidence (probability of grant):** {prob:.3f}")

# ------------------------------------------------------------
# 📘 Sidebar info
# ------------------------------------------------------------
st.sidebar.header("Dataset & Model Info")
st.sidebar.write("Model: XGBoost (sklearn wrapper)")

st.markdown("---")
st.caption("Built with XGBoost + Streamlit. Data and model consistent with training pipeline.")

import streamlit as st
import joblib
import numpy as np
import os

# ✅ Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Load models safely from 'models' folder in the project root
tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder4.joblib"))
model = joblib.load(os.path.join(BASE_DIR, "models", "random_forest.joblib")) # you can change model here

# -------------------- Streamlit Page Config --------------------
st.set_page_config(
    page_title="🩺 Symptom-Based Disease Prediction Chatbot",
    layout="wide",
    page_icon="💊",
)

# -------------------- Custom CSS --------------------
st.markdown("""
    <style>
        body {
            background-color: #f8fafc;
        }
        .main-title {
            font-size: 35px;
            color: #0078D7;
            text-align: center;
            font-weight: 800;
        }
        .subtext {
            color: #555;
            text-align: center;
            font-size: 18px;
        }
        .stTextArea textarea {
            border-radius: 10px !important;
            border: 2px solid #0078D7 !important;
            font-size: 16px !important;
        }
        .prediction-box {
            background-color: #e8f0fe;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .result {
            font-size: 18px;
            color: #000;
        }
        .highlight {
            color: #0078D7;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Main Page --------------------
st.markdown('<h1 class="main-title">🩺 Symptom-Based Disease Prediction Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Enter your symptoms below and get the top 3 possible diseases with confidence scores.</p>', unsafe_allow_html=True)

# Input Section
user_input = st.text_area("Describe your symptoms (e.g., fever, headache, runny nose):", height=150)

if st.button("🔍 Predict Disease"):
    if user_input.strip():
        # Preprocess & Predict
        X = tfidf.transform([user_input])
        probs = model.predict_proba(X)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_diseases = label_encoder.inverse_transform(top3_idx)
        top3_scores = probs[top3_idx] * 100

        # Display results
        st.markdown("### 🧾 Prediction Results")
        for i in range(3):
            st.markdown(
                f"<div class='prediction-box'>"
                f"<p class='result'> <b>{i+1}. {top3_diseases[i]}</b> — "
                f"<span class='highlight'>{top3_scores[i]:.2f}% match</span></p>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter your symptoms before predicting.")

# -------------------- Sidebar --------------------
st.sidebar.title("🧩 Model Information")
st.sidebar.info("""
**Model Used:** Random Forest  
**Feature Extraction:** TF-IDF Vectorizer  
**Trained On:** Cleaned symptom-disease dataset  
**Label Encoder:** Encoded disease names
""")

st.sidebar.title("📊 Accuracy Report")
st.sidebar.success("""
✅ Logistic Regression: 87%  
✅ SVM (Linear): 89%  
✅ Random Forest: **92%**
""")

st.sidebar.title("💡 About Project")
st.sidebar.write("""
This chatbot uses **Natural Language Processing (NLP)** to analyze your described
symptoms and predict the most probable diseases using machine learning models.
""")
st.sidebar.markdown("---")
st.sidebar.caption("Developed by **Aaman Inamdar**  | Streamlit + NLP + ML")



import streamlit as st
import re
import nltk
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')

# Initialize stemmer and stopwords
porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ------------------ Text Preprocessing ------------------
def stemming_fast(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    stemmed_words = [porter_stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(stemmed_words)

# ------------------ Load Model & Vectorizer ------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")           # Pre-trained Logistic Regression
    vectorizer = joblib.load("vectorizer.pkl") # Pre-fitted TF-IDF vectorizer
    return model, vectorizer

model, vectorizer = load_model()

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Add Custom Background with CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #f0f4c3, #c8e6c9);
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffccbc, #ffe0b2);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #FF5722;'>📰 Fake News Detection App</h1>
    <p style='text-align: center; color: #4CAF50;'>Check if a news article is <b>Real</b> or <b>Fake</b> instantly!</p>
    <hr style="border:1px solid #FF5722">
    """,
    unsafe_allow_html=True
)

# Sidebar Info
st.sidebar.title("📊 App Info")
st.sidebar.info("This app uses **Logistic Regression** with TF-IDF to classify news as Fake or Real.")
st.sidebar.success("✅ Pre-trained model loaded instantly")

# Input box for user
user_input = st.text_area("✍️ Enter News Text Here", height=200, placeholder="Paste the news article text...")

# Predict button
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Preprocess
        input_data = stemming_fast(user_input)
        vectorized_input = vectorizer.transform([input_data])
        prediction = model.predict(vectorized_input)

        if prediction[0] == 0:
            st.success("✅ The News is Real")
            st.markdown("<h3 style='color:#4CAF50;'>The model predicts this article is Real ✅</h3>", unsafe_allow_html=True)
        else:
            st.error("❌ The News is Fake")
            st.markdown("<h3 style='color:#FF5722;'>The model predicts this article is Fake ❌</h3>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: grey;'>Made with ❤️ using Streamlit</p>
    """,
    unsafe_allow_html=True
)

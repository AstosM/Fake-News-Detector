import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Add Custom Background
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

# Sidebar
st.sidebar.title("📊 App Info")
st.sidebar.info("This app trains a Logistic Regression model on a CSV dataset and classifies news as Real or Fake.")
st.sidebar.markdown("CSV must have columns: id, title, author, text, label (0=Real, 1=Fake)")

# ------------------ Load CSV directly ------------------
# 🔹 Replace this filename with the one you upload (must be in same folder as app.py)
csv_file = "Fake news dataset.csv"

news_dataset = pd.read_csv(csv_file)

# Fill missing text and drop NaN labels
news_dataset['text'] = news_dataset['text'].fillna('')
news_dataset = news_dataset.dropna(subset=['label'])

# Apply stemming
news_dataset['text'] = news_dataset['text'].astype(str).apply(stemming_fast)

# Prepare data
X = news_dataset['text'].values
Y = news_dataset['label'].values

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

st.sidebar.success(f"✅ Training Accuracy: {train_acc:.2f}")
st.sidebar.success(f"✅ Test Accuracy: {test_acc:.2f}")

# ------------------ Prediction ------------------
st.markdown("## 🔍 Predict News Article")
user_input = st.text_area("✍️ Enter News Text Here", height=200, placeholder="Paste the news article text...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
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

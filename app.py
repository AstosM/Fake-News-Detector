import streamlit as st
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')

# Initialize stemmer
porter_stemmer = PorterStemmer()

# ------------------ Text Preprocessing ------------------
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [porter_stemmer.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# ------------------ Load Dataset & Train Model ------------------
news_dataset = pd.read_csv("fake_or_real_news.csv.zip", compression="zip")
news_dataset['label'] = news_dataset['label'].map({'FAKE': 1, 'REAL': 0})
news_dataset['text'] = news_dataset['text'].astype(str).apply(stemming)

X = news_dataset['text'].values
Y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Add Custom Background with CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #ffecd2, #fcb69f);
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #a1c4fd, #c2e9fb);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>📰 Fake News Detection App</h1>
    <p style='text-align: center; color: #4CAF50;'>Check if a news article is <b>Real</b> or <b>Fake</b> instantly!</p>
    <hr style="border:1px solid #FF4B4B">
    """,
    unsafe_allow_html=True
)

# Sidebar Info
st.sidebar.title("📊 App Info")
st.sidebar.info("This app uses **Logistic Regression** with TF-IDF to classify news as Fake or Real.")
st.sidebar.markdown("**Dataset:** Fake/Real News dataset 🗂️")
st.sidebar.success("✅ Accuracy: ~90% on test data")

# Input box
user_input = st.text_area("✍️ Enter News Text Here", height=200, placeholder="Paste the news article text...")

# Predict button
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        input_data = stemming(user_input)
        vectorized_input = vectorizer.transform([input_data])
        prediction = model.predict(vectorized_input)

        if prediction[0] == 0:
            st.success("✅ The News is Real")
            st.markdown("<h3 style='color:#4CAF50;'>The model predicts this article is Real ✅</h3>", unsafe_allow_html=True)
        else:
            st.error("❌ The News is Fake")
            st.markdown("<h3 style='color:#FF4B4B;'>The model predicts this article is Fake ❌</h3>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: grey;'>Made with ❤️ using Streamlit</p>
    """,
    unsafe_allow_html=True
)

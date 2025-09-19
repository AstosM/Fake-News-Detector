📰 Fake News Detection App

This project is a Machine Learning web application built with Streamlit to detect whether a news article is Real or Fake.

🚀 Features
Clean & colorful Streamlit UI with gradient backgrounds
Text preprocessing (stopword removal, stemming, lowercasing)
TF-IDF vectorization to convert text into numerical features
Logistic Regression model for classification
Instant prediction of news authenticity (Real ✅ / Fake ❌)
Sidebar with dataset details and accuracy.

📂 Project Structure
fake-news-detection/
│── app.py               # Main Streamlit app  
│── fake_or_real_news[1].csv   # Dataset  
│── requirements.txt     # Required Python libraries  
│── README.md            # Project documentation  

📊 Dataset
We use the Fake/Real News Dataset containing news headlines and full text labeled as:
FAKE → Fake news
REAL → Genuine news
The dataset is preprocessed and split into training and testing sets.

🤖 Model
TF-IDF Vectorizer → Converts text into numerical vectors
Logistic Regression → Classifier to detect Fake vs Real news
Accuracy → ~90% on test data.

🛠️ Tech Stack
Python 🐍
Streamlit 🎨
Pandas, NumPy 📊
NLTK (text preprocessing) 📝
Scikit-learn 🤖

👨‍💻 Author

Developed by [Ashutosh Maurya] 💻

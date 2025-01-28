import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and preprocess data
@st.cache
def load_data(filepath):
    data = pd.read_csv(filepath)
    data = data[['tweet', 'sentiment']]  # Ensure required columns exist
    return data

# Train a simple sentiment model
@st.cache
def train_model(data):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['tweet'])
    y = data['sentiment']
    
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

# Streamlit app
def main():
    st.title("Twitter Sentiment Analysis")
    
    # Sidebar
    st.sidebar.header("Upload CSV Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        data = load_data(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
        
        # Split data and train model
        vectorizer, model = train_model(data)
        
        # Input for new tweet
        st.subheader("Test New Tweet")
        tweet_input = st.text_input("Enter a tweet:")
        
        if tweet_input:
            tweet_vector = vectorizer.transform([tweet_input])
            prediction = model.predict(tweet_vector)[0]
            st.write(f"Sentiment Prediction: **{prediction.capitalize()}**")
    else:
        st.info("Please upload a dataset to begin.")

if __name__ == "__main__":
    main()

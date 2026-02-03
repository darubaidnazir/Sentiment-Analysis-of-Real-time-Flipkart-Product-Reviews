import streamlit as st
import joblib
from preprocessing import clean_text

# Load model
model = joblib.load("model/sentiment_model.pkl")
tfidf = joblib.load("model/tfidf.pkl")

st.set_page_config(page_title="Multi-Product Sentiment Analysis")

st.title("🛒 Flipkart Multi-Product Review Sentiment Analyzer")

product = st.selectbox(
    "Select Product",
    ["badminton", "Tawa", "Tea"]
)

review = st.text_area("✍️ Enter customer review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_clean = clean_text(review)
        vector = tfidf.transform([review_clean])
        prediction = model.predict(vector)[0]

        st.subheader(f"Product: {product}")

        if prediction == 1:
            st.success("✅ Positive Review 😊")
        else:
            st.error("❌ Negative Review 😞")

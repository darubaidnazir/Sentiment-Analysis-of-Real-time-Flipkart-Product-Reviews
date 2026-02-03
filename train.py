import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from preprocessing import clean_text

# Load all product datasets
df1 = pd.read_csv("data/data_badminton.csv")
df2 = pd.read_csv("data/data_tawa.csv")
df3 = pd.read_csv("data/data_tea.csv")

# Add product labels
df1["product"] = "badminton"
df2["product"] = "tawa"
df3["product"] = "tea"

# Combine datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

# Create sentiment label
df["sentiment"] = df["Ratings"].apply(lambda x: 1 if x >= 4 else 0)

# Clean reviews
df["clean_review"] = df["Review text"].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(df["clean_review"])
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("F1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")

print("✅ Model trained on all three products and saved.")

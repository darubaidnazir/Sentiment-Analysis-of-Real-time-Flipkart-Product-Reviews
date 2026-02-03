print("RUNNING UPDATED PAIN POINTS FILE")

import pandas as pd
from collections import Counter
from preprocessing import clean_text

files = {
    "badminton": "data/data_badminton.csv",
    "tawa": "data/data_tawa.csv",
    "tea": "data/data_tea.csv"
}

for product, path in files.items():
    df = pd.read_csv(path)

    # 🔥 Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # 🔥 STANDARDIZE column names explicitly
    df = df.rename(columns={
        "ratings": "rating",        # plural → singular
        "review_text": "review_text"
    })

    print(f"\nColumns in {product} dataset:", df.columns.tolist())

    # ✅ Now this WILL work
    df["sentiment"] = df["rating"].apply(lambda x: 1 if float(x) >= 4 else 0)
    df["clean_review"] = df["review_text"].apply(clean_text)

    negative_reviews = df[df["sentiment"] == 0]["clean_review"]
    words = " ".join(negative_reviews).split()

    print(f"\n🔴 Pain Points for {product}:")
    for word, count in Counter(words).most_common(10):
        print(f"{word}: {count}")

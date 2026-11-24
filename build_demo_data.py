import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

reviews = pd.read_csv(DATA / "sample_reviews.csv")
reviews["review_date"] = pd.to_datetime(reviews["review_date"])
reviews["review_text"] = reviews["review_text"].fillna("")

aspect_keywords = {
    "Food Quality": ["food", "taste", "delicious", "cold", "fresh", "meal"],
    "Delivery": ["delivery", "delivered", "late", "slow", "order", "arrive"],
    "Staff & Service": ["staff", "service", "rude", "friendly", "helpful"],
    "Packaging": ["packaging", "package", "container", "damaged", "leaking"],
    "Pricing": ["price", "expensive", "cheap", "cost", "overpriced"]
}

def sentiment_for(text, rating):
    text = text.lower()
    negative_words = ["late", "cold", "rude", "terrible", "damaged",
                      "leaking", "slow", "poor", "unhelpful", "expensive"]
    positive_words = ["excellent", "friendly", "amazing", "delicious",
                      "great", "good", "helpful"]
    if any(word in text for word in negative_words):
        return "Negative"
    if any(word in text for word in positive_words):
        return "Positive"
    if rating >= 4:
        return "Positive"
    if rating <= 2:
        return "Negative"
    return "Neutral"

aspect_rows = []

for _, row in reviews.iterrows():
    text = row["review_text"].lower()
    matched = [
        aspect for aspect, words in aspect_keywords.items()
        if any(word in text for word in words)
    ]
    if not matched:
        matched = ["General"]

    for aspect in matched:
        sentiment = sentiment_for(text, row["rating"])
        score = {"Positive": 0.6, "Neutral": 0.0, "Negative": -0.6}[sentiment]

        aspect_rows.append({
            "review_id": row["review_id"],
            "business_id": row["business_id"],
            "location": row["location"],
            "review_date": row["review_date"],
            "rating": row["rating"],
            "review_text": row["review_text"],
            "aspect": aspect,
            "sentiment_score": score,
            "sentiment": sentiment
        })

aspects = pd.DataFrame(aspect_rows)
aspects.to_csv(OUT / "aspect_sentiment.csv", index=False)

reviews.to_csv(OUT / "cleaned_reviews.csv", index=False)

negative = aspects[aspects["sentiment"] == "Negative"].copy()
negative["cluster_id"] = negative["aspect"].astype("category").cat.codes
negative["cluster_keywords"] = negative["aspect"]
negative.to_csv(OUT / "complaint_clusters.csv", index=False)

aspects["month"] = aspects["review_date"].dt.to_period("M").astype(str)
monthly = (
    aspects.groupby(["month", "aspect"])
    .agg(
        total_reviews=("review_id", "nunique"),
        negative_reviews=("sentiment", lambda x: (x == "Negative").sum())
    )
    .reset_index()
)
monthly["negative_rate"] = (
    monthly["negative_reviews"] / monthly["total_reviews"]
)
monthly["baseline_rate"] = monthly.groupby("aspect")["negative_rate"].transform(
    lambda x: x.shift(1).rolling(2, min_periods=1).mean()
)
monthly["potential_alert"] = (
    (monthly["negative_reviews"] >= 2)
    & (monthly["negative_rate"] >= monthly["baseline_rate"] + 0.10)
    & monthly["baseline_rate"].notna()
)
monthly["month"] = pd.to_datetime(monthly["month"])
monthly.to_csv(OUT / "emerging_complaints.csv", index=False)

recommendations = (
    negative.groupby(["business_id", "location", "aspect"])
    .agg(
        total_reviews=("review_id", "nunique"),
        negative_reviews=("review_id", "nunique")
    )
    .reset_index()
)
recommendations["negative_rate"] = (
    recommendations["negative_reviews"] / recommendations["total_reviews"]
)
recommendations["priority"] = "Medium"
recommendations.loc[
    recommendations["negative_reviews"] >= 2, "priority"
] = "High"
recommendations["recommended_action"] = recommendations["aspect"].map({
    "Food Quality": "Review food preparation and quality checks.",
    "Delivery": "Review delivery timing and order handoff processes.",
    "Staff & Service": "Provide staff training and review service procedures.",
    "Packaging": "Inspect packaging materials and sealing procedures.",
    "Pricing": "Review pricing and communicate value clearly.",
    "General": "Review customer feedback and investigate recurring concerns."
})
recommendations["priority_score"] = (
    recommendations["negative_rate"] * recommendations["negative_reviews"]
)
recommendations.to_csv(OUT / "business_recommendations.csv", index=False)

print("Synthetic demo datasets created successfully.")
print("Aspect records:", len(aspects))
print("Unique reviews:", reviews["review_id"].nunique())
print("Negative complaint records:", len(negative))

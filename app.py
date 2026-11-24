
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Revio | Feedback Intelligence",
    page_icon="\U0001F4CA",
    layout="wide"
)

DATA = Path("data/processed")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    aspects = pd.read_csv(DATA / "aspect_sentiment.csv")
    reviews = pd.read_csv(DATA / "cleaned_reviews.csv")
    clusters = pd.read_csv(DATA / "complaint_clusters.csv")
    alerts = pd.read_csv(DATA / "emerging_complaints.csv")
    recommendations = pd.read_csv(DATA / "business_recommendations.csv")

    aspects["review_date"] = pd.to_datetime(aspects["review_date"])
    alerts["month"] = pd.to_datetime(alerts["month"])

    return aspects, reviews, clusters, alerts, recommendations


try:
    aspects, reviews, clusters, alerts, recommendations = load_data()
except Exception as e:
    st.error(f"Could not load project datasets: {e}")
    st.stop()

# ---------------- HEADER ----------------

st.title("Revio")
st.subheader("AI-Powered Customer Feedback Intelligence")
st.caption(
    "Restaurant feedback analysis using sentiment analysis, "
    "complaint discovery and actionable recommendations."
)

st.info(
    "Research prototype using historical Yelp Open Dataset reviews. "
    "This is not live Revio customer data."
)

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.title("Dashboard Filters")

locations = sorted(aspects["location"].dropna().unique())

selected_locations = st.sidebar.multiselect(
    "Select locations",
    locations,
    default=locations
)

available_aspects = sorted(aspects["aspect"].dropna().unique())

selected_aspects = st.sidebar.multiselect(
    "Select aspects",
    available_aspects,
    default=available_aspects
)

filtered = aspects[
    aspects["location"].isin(selected_locations)
    & aspects["aspect"].isin(selected_aspects)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ---------------- KPI CARDS ----------------

total_reviews = filtered["review_id"].nunique()
positive = (filtered["sentiment"] == "Positive").sum()
negative = (filtered["sentiment"] == "Negative").sum()
neutral = (filtered["sentiment"] == "Neutral").sum()

negative_rate = negative / len(filtered) * 100

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Unique Reviews", f"{total_reviews:,}")
c2.metric("Aspect Records", f"{len(filtered):,}")
c3.metric("Positive", f"{positive:,}")
c4.metric("Negative", f"{negative:,}")
c5.metric("Negative Rate", f"{negative_rate:.1f}%")

st.divider()

# ---------------- SENTIMENT OVERVIEW ----------------

st.header("1. Sentiment Overview")

left, right = st.columns(2)

with left:
    sentiment_counts = (
        filtered["sentiment"]
        .value_counts()
        .reset_index()
    )
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=0.45,
        title="Overall Sentiment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    aspect_sentiment = (
        filtered.groupby(["aspect", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        aspect_sentiment,
        x="aspect",
        y="count",
        color="sentiment",
        barmode="group",
        title="Sentiment by Aspect"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- ASPECT ANALYSIS ----------------

st.header("2. Aspect-Based Analysis")

aspect_summary = (
    filtered.groupby("aspect")
    .agg(
        total_records=("review_id", "count"),
        unique_reviews=("review_id", "nunique"),
        average_score=("sentiment_score", "mean"),
        negative_records=(
            "sentiment",
            lambda x: (x == "Negative").sum()
        )
    )
    .reset_index()
)

aspect_summary["negative_rate"] = (
    aspect_summary["negative_records"]
    / aspect_summary["total_records"] * 100
)

st.dataframe(
    aspect_summary.round(2),
    use_container_width=True,
    hide_index=True
)

# ---------------- COMPLAINT CLUSTERS ----------------

st.header("3. Complaint Discovery")

st.caption(
    "Exploratory TF-IDF and KMeans clusters. "
    "Themes may overlap and require human interpretation."
)

if not clusters.empty:
    cluster_summary = (
        clusters.groupby("cluster_id")
        .agg(
            complaint_records=("review_id", "count"),
            unique_reviews=("review_id", "nunique"),
            keywords=("cluster_keywords", "first")
        )
        .reset_index()
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True,
        hide_index=True
    )

    selected_cluster = st.selectbox(
        "Explore complaint cluster",
        sorted(clusters["cluster_id"].unique())
    )

    cluster_reviews = clusters[
        clusters["cluster_id"] == selected_cluster
    ]

    for _, row in cluster_reviews.head(5).iterrows():
        with st.expander(
            f"{row['location']} | {row['aspect']} | Rating {row['rating']}"
        ):
            st.write(row["review_text"])
            st.caption(f"Keywords: {row['cluster_keywords']}")

# ---------------- EMERGING COMPLAINTS ----------------

st.header("4. Emerging Complaint Alerts")

alert_data = alerts[
    (alerts["potential_alert"] == True)
    & (alerts["aspect"].isin(selected_aspects))
].copy()

if alert_data.empty:
    st.info("No potential alerts found for the selected aspects.")
else:
    st.metric("Historical Potential Alerts", len(alert_data))

    fig = px.line(
        alert_data.sort_values("month"),
        x="month",
        y="negative_rate",
        color="aspect",
        markers=True,
        title="Negative Feedback Rate During Potential Alert Months"
    )

    st.plotly_chart(fig, use_container_width=True)

    display_alerts = alert_data.sort_values(
        "negative_rate",
        ascending=False
    )

    st.dataframe(
        display_alerts.round(3),
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Alerts are historical aggregate signals, not real-time notifications. "
    "The detector compares against previous observed months."
)

# ---------------- BUSINESS RECOMMENDATIONS ----------------

st.header("5. Business Recommendations")

if not recommendations.empty:
    priority_filter = st.multiselect(
        "Filter by priority",
        sorted(recommendations["priority"].dropna().unique()),
        default=sorted(recommendations["priority"].dropna().unique())
    )

    recommendation_data = recommendations[
        recommendations["priority"].isin(priority_filter)
    ].sort_values("priority_score", ascending=False)

    st.dataframe(
        recommendation_data[
            [
                "location",
                "aspect",
                "total_reviews",
                "negative_rate",
                "priority",
                "recommended_action"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# ---------------- REVIEW EXPLORER ----------------

st.header("6. Customer Review Explorer")

search = st.text_input(
    "Search reviews",
    placeholder="Enter a keyword such as slow, food, service..."
)

review_data = reviews[
    reviews["location"].isin(selected_locations)
].copy()

if search:
    review_data = review_data[
        review_data["review_text"].str.contains(
            search,
            case=False,
            na=False
        )
    ]

st.write(f"Matching reviews: {len(review_data):,}")

st.dataframe(
    review_data[
        ["location", "rating", "review_date", "review_text"]
    ].head(100),
    use_container_width=True,
    hide_index=True
)

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Revio AI Intelligence | Cynbit AI/ML Engineer Internship Prototype"
)
st.caption(
    "Dataset: Synthetic demonstration data | Methods: Rule-based sentiment, aspect grouping and exploratory complaint analysis"
)



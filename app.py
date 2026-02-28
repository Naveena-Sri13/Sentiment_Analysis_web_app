"""
Product Review Sentiment Analyzer
==================================
A Streamlit web application that analyzes product review sentiment
using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).

Author: [Your Name]
Date: 2026
Course: Computer Science Internship Project

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, List
import io


# ─── NLTK Setup ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_vader():
    """Download and cache the VADER lexicon for sentiment analysis."""
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


# ─── Sentiment Analysis Functions ─────────────────────────────────────────────
def classify_sentiment(compound_score: float) -> Tuple[str, str]:
    """
    Classify a VADER compound score into a sentiment category.

    Args:
        compound_score: VADER compound score ranging from -1 to +1

    Returns:
        Tuple of (sentiment_label, emoji)

    Classification thresholds:
        - Positive: compound > 0.05
        - Negative: compound < -0.05
        - Neutral:  -0.05 <= compound <= 0.05
    """
    if compound_score > 0.05:
        return "Positive", "😊"
    elif compound_score < -0.05:
        return "Negative", "😞"
    else:
        return "Neutral", "😐"


def analyze_review(text: str, analyzer: SentimentIntensityAnalyzer) -> Dict:
    """
    Analyze a single product review and return sentiment metrics.

    Args:
        text: The review text to analyze
        analyzer: Pre-loaded VADER SentimentIntensityAnalyzer instance

    Returns:
        Dictionary containing sentiment label, emoji, compound score,
        confidence, and individual polarity scores
    """
    if not text or not text.strip():
        return {
            "label": "Neutral",
            "emoji": "😐",
            "compound": 0.0,
            "confidence": 0.0,
            "pos": 0.0,
            "neu": 0.0,
            "neg": 0.0,
        }

    scores = analyzer.polarity_scores(text)
    label, emoji = classify_sentiment(scores["compound"])
    confidence = abs(scores["compound"])

    return {
        "label": label,
        "emoji": emoji,
        "compound": round(scores["compound"], 4),
        "confidence": round(confidence, 4),
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
    }


def analyze_batch(texts: List[str], analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Perform batch sentiment analysis on a list of review texts.

    Args:
        texts: List of review strings
        analyzer: Pre-loaded VADER SentimentIntensityAnalyzer instance

    Returns:
        DataFrame with columns: review, label, emoji, compound, confidence, pos, neu, neg
    """
    results = []
    for text in texts:
        if text and str(text).strip():
            result = analyze_review(str(text), analyzer)
            result["review"] = str(text).strip()
            results.append(result)
    return pd.DataFrame(results)


# ─── Visualization Functions ──────────────────────────────────────────────────
def create_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing the distribution of sentiment categories.

    Args:
        df: DataFrame with a 'label' column containing sentiment labels

    Returns:
        Plotly Figure object
    """
    counts = df["label"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )

    colors = {"Positive": "#2dd4bf", "Neutral": "#f59e0b", "Negative": "#ef4444"}

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map=colors,
        labels={"x": "Sentiment", "y": "Count"},
        title="Sentiment Distribution",
    )
    fig.update_layout(showlegend=False, template="plotly_dark")
    return fig


def create_pie_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing sentiment proportions.

    Args:
        df: DataFrame with a 'label' column

    Returns:
        Plotly Figure object
    """
    counts = df["label"].value_counts()
    colors = {"Positive": "#2dd4bf", "Neutral": "#f59e0b", "Negative": "#ef4444"}

    fig = px.pie(
        names=counts.index,
        values=counts.values,
        color=counts.index,
        color_discrete_map=colors,
        title="Sentiment Proportions",
        hole=0.4,
    )
    fig.update_layout(template="plotly_dark")
    return fig


# ─── CSV Processing ───────────────────────────────────────────────────────────
def process_csv(uploaded_file, text_column: str, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Process an uploaded CSV file and perform batch sentiment analysis.

    Args:
        uploaded_file: Streamlit UploadedFile object
        text_column: Name of the column containing review text
        analyzer: Pre-loaded VADER SentimentIntensityAnalyzer instance

    Returns:
        DataFrame with sentiment analysis results

    Raises:
        ValueError: If the specified column is not found in the CSV
    """
    df = pd.read_csv(uploaded_file)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")

    texts = df[text_column].dropna().astype(str).tolist()
    return analyze_batch(texts, analyzer)


# ─── Main Application ─────────────────────────────────────────────────────────
def main():
    """Main entry point for the Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="Product Review Sentiment Analyzer",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Product Review Sentiment Analyzer")
    st.markdown(
        "Analyze product reviews using **VADER NLP** to determine "
        "sentiment polarity (Positive, Negative, or Neutral)."
    )

    # Load the VADER analyzer
    analyzer = load_vader()

    # ── Tabs for Single / Batch Analysis ──────────────────────────────────
    tab1, tab2 = st.tabs(["📝 Single Review", "📁 Batch CSV Upload"])

    # ── Single Review Tab ─────────────────────────────────────────────────
    with tab1:
        review_text = st.text_area(
            "Enter a product review:",
            placeholder="Type or paste a product review here...",
            height=120,
        )

        if st.button("Analyze Sentiment", type="primary"):
            if not review_text.strip():
                st.warning("⚠️ Please enter a review to analyze.")
            else:
                result = analyze_review(review_text, analyzer)

                col1, col2, col3 = st.columns(3)
                col1.metric("Sentiment", f"{result['emoji']} {result['label']}")
                col2.metric("Compound Score", f"{result['compound']:.4f}")
                col3.metric("Confidence", f"{result['confidence']:.1%}")

                with st.expander("Detailed Scores"):
                    st.json(result)

    # ── Batch CSV Tab ─────────────────────────────────────────────────────
    with tab2:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded is not None:
            try:
                preview_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                columns = preview_df.columns.tolist()

                text_col = st.selectbox("Select the review text column:", columns)

                if st.button("Analyze All Reviews", type="primary"):
                    uploaded.seek(0)
                    results_df = process_csv(uploaded, text_col, analyzer)

                    if results_df.empty:
                        st.warning("No valid reviews found in the selected column.")
                    else:
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        total = len(results_df)
                        col1.metric("Total Reviews", total)
                        pos_pct = (results_df["label"] == "Positive").mean()
                        col2.metric("Positive", f"{pos_pct:.1%}")
                        neg_pct = (results_df["label"] == "Negative").mean()
                        col3.metric("Negative", f"{neg_pct:.1%}")
                        avg_score = results_df["compound"].mean()
                        col4.metric("Avg Score", f"{avg_score:.4f}")

                        # Charts
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            st.plotly_chart(
                                create_distribution_chart(results_df),
                                use_container_width=True,
                            )
                        with chart_col2:
                            st.plotly_chart(
                                create_pie_chart(results_df),
                                use_container_width=True,
                            )

                        # Results table
                        st.dataframe(
                            results_df[["review", "label", "emoji", "compound", "confidence"]],
                            use_container_width=True,
                        )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit & NLTK VADER | CS Internship Project 2026")


if __name__ == "__main__":
    main()

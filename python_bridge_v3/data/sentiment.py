"""
=============================================================
  Python ML Bridge - Sentiment Analysis Module
  Uses HuggingFace FinBERT (ProsusAI/finbert) for financial
  news sentiment analysis from RSS feeds.
=============================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import requests
import feedparser

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SentimentConfig


class SentimentAnalyzer:
    """Financial news sentiment analysis using FinBERT."""

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._pipeline = None
        self._sentiment_history: List[Dict] = []

    def _load_model(self):
        """Lazy-load the FinBERT model pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.config.model_name,
                    tokenizer=self.config.model_name,
                    top_k=None
                )
            except Exception as e:
                print(f"[Sentiment] Failed to load FinBERT model: {e}")
                self._pipeline = None

    def fetch_news(self) -> List[Dict]:
        """
        Fetch financial news articles from RSS feeds.

        Returns:
            List of dicts with 'title', 'summary', 'published' keys
        """
        articles = []
        for feed_url in self.config.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:self.config.max_articles]:
                    article = {
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "published": entry.get("published", ""),
                        "source": feed_url,
                    }
                    articles.append(article)
            except Exception as e:
                print(f"[Sentiment] Error fetching feed {feed_url}: {e}")
                continue

        return articles[:self.config.max_articles]

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using FinBERT.

        Args:
            text: Input text to analyze

        Returns:
            Dict with 'positive', 'negative', 'neutral' scores
        """
        self._load_model()
        if self._pipeline is None:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        try:
            # Truncate text to model max length
            text = text[:512]
            results = self._pipeline(text)
            scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            if results and isinstance(results[0], list):
                for item in results[0]:
                    scores[item["label"]] = item["score"]
            elif results:
                for item in results:
                    scores[item["label"]] = item["score"]
            return scores
        except Exception as e:
            print(f"[Sentiment] Error analyzing text: {e}")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    def analyze_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment of multiple articles.

        Args:
            articles: List of article dicts from fetch_news()

        Returns:
            Articles with added 'sentiment' scores
        """
        analyzed = []
        for article in articles:
            text = f"{article['title']}. {article['summary']}"
            sentiment = self.analyze_text(text)
            article["sentiment"] = sentiment
            article["compound_score"] = (
                sentiment["positive"] - sentiment["negative"]
            )
            analyzed.append(article)

        return analyzed

    def compute_sentiment_index(self) -> Dict[str, float]:
        """
        Compute rolling sentiment index for trading signals.

        Returns:
            Dict with 'score' (-1 to 1), 'confidence', 'num_articles',
            'bullish_pct', 'bearish_pct'
        """
        articles = self.fetch_news()
        if not articles:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "num_articles": 0,
                "bullish_pct": 0.0,
                "bearish_pct": 0.0,
            }

        analyzed = self.analyze_articles(articles)

        # Store in history
        self._sentiment_history.extend(analyzed)

        # Keep only recent articles within rolling window
        cutoff = datetime.now() - timedelta(hours=self.config.rolling_window)
        self._sentiment_history = [
            a for a in self._sentiment_history
            if a.get("published", "") != ""
        ][-self.config.max_articles * 3:]

        # Compute aggregate scores
        scores = [a["compound_score"] for a in analyzed]
        if not scores:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "num_articles": 0,
                "bullish_pct": 0.0,
                "bearish_pct": 0.0,
            }

        avg_score = np.mean(scores)
        confidence = min(1.0, len(scores) / 10.0)  # More articles = higher confidence
        bullish = sum(1 for s in scores if s > 0.2) / len(scores)
        bearish = sum(1 for s in scores if s < -0.2) / len(scores)

        return {
            "score": float(np.clip(avg_score, -1.0, 1.0)),
            "confidence": float(confidence),
            "num_articles": len(scores),
            "bullish_pct": float(bullish),
            "bearish_pct": float(bearish),
        }

    def get_sentiment_features(self) -> np.ndarray:
        """
        Get sentiment features as numpy array for model input.

        Returns:
            Array of shape (5,) with sentiment metrics
        """
        index = self.compute_sentiment_index()
        return np.array([
            index["score"],
            index["confidence"],
            index["num_articles"] / 50.0,  # Normalize
            index["bullish_pct"],
            index["bearish_pct"],
        ], dtype=np.float32)

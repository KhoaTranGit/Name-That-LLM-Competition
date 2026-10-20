import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline as pipe
from sklearn.svm import LinearSVC

PUNCTUATIONS = [
  "-", "–", "—", ",", 
  '"', ";", ":", "?", 
  "!", "...", "("
]


class StylometricFeatures(BaseEstimator, TransformerMixin):
    """
    Example usage: 
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("char_tfidf", char_vectorizer, "text"),
            ("stylometry", pipe(StylometricFeatures(), StandardScaler()), ["text"])
        ],
        transformer_weights={
            "char_tfidf": 1.0,
            "stylometry": 0.1
        },
        sparse_threshold=1.0
    )
    """
    def fit(self, X, y=None): return self

    def transform(self, X):
        texts = np.asarray(X).ravel()
        return np.array([self.extract_features(text) for text in texts], dtype=np.float32)

    @staticmethod
    def extract_features(text):
        ## Extract words & sentences
        text      = "" if pd.isna(text) else str(text)
        words     = re.findall(r"\b\w+\b", text)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        ## Extract word & sentence lengths 
        n_words = max(1, len(words))
        n_chars = max(1, len(text))
        word_lengths     = [len(w) for w in words]
        sentence_lengths = [ len(re.findall(r"\b\w+\b", s)) for s in sentences]
        ## Extract relevant features
        return [
            # Document length
            len(words), len(text),

            # Punctuation frequency
            *[text.count(c)/n_words for c in PUNCTUATIONS],
            # Word statistics
            np.mean(word_lengths) if words else 0,
            np.std(word_lengths) if words else 0,
            # Sentence statistics
            np.mean(sentence_lengths) if sentences else 0,
            np.std(sentence_lengths) if sentences else 0,
            # Vocabulary diversity
            len(set(w.lower() for w in words)) / n_words,
            # Formatting
            text.count("\n") / n_words,
            # Capitalization
            sum(c.isupper() for c in text) / n_chars,
        ]

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def make_TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2,
                         max_features=200_000, sublinear_tf=True):
    return TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngram_range,
        min_df=min_df, max_features=max_features,
        sublinear_tf=sublinear_tf, dtype=np.float32,
    )

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from ..utils import log


class VectorTransformer(TransformerMixin):
    @log('Embedding: Tf-Idf')
    def _tfidf(self, text):
        return TfidfVectorizer(max_features=self.features, norm=self.norm).fit_transform(text)

    @log('Embedding: Hashing')
    def _hashing(self, text):
        return HashingVectorizer(n_features=self.features, non_negative=True, norm=self.norm).fit_transform(text)

    @log('Embedding: Count (BoW)')
    def _count(self, text):
        return CountVectorizer(max_features=self.features).fit_transform(text)

    def __init__(self, attrs):
        self.embeddings = {
            'hashing': self._hashing,
            'tfidf': self._tfidf,
            'count': self._count
        }
        self.norm = None if attrs['norm'] == 'none' else attrs['norm']
        self.features = int(attrs['featureNumber'])
        self.embedding = attrs['embedding']

    def fit(self, y=None):
        return self

    def transform(self, text, y=None):
        X = self.embeddings[self.embedding](text)
        return X.toarray()

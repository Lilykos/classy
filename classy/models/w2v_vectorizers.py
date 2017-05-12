from sklearn.base import TransformerMixin


class W2VVectorTransformer(TransformerMixin):
    def __init__(self, attrs):
        pass

    def fit(self, y=None):
        return self

    def transform(self, text, y=None):
        X = self.embeddings[self.embedding](text)
        return X.toarray()

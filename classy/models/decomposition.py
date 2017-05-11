from sklearn.base import TransformerMixin
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import pairwise_distances
from ..utils import log


class DecompositionTransformer(TransformerMixin):
    @log('Decomposition: t-SNE')
    def _tsne(self, X):
        return TSNE(n_components=self.dimensions, learning_rate=100, perplexity=15, metric=self.metric).fit(X)

    @log('Decomposition: MDS')
    def _mds(self, X):
        X = pairwise_distances(X, metric=self.metric)
        return MDS(n_components=self.dimensions, dissimilarity='precomputed', random_state=42, n_jobs=-1).fit(X)

    def __init__(self, attrs):
        self.decomp_models = {
            'tsne': self._tsne,
            'mds': self._mds
        }
        self.decomp = attrs['decomposition']
        self.metric = attrs['decompositionMetric']
        self.dimensions = int(attrs['dimensionsNumber'])

    def fit(self, y=None):
        return self

    def transform(self, X, y=None):
        if self.decomp != 'none':
            model = self.decomp_models[self.decomp](X)
            return model.embedding_
        return X

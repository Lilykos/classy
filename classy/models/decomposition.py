from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import pairwise_distances
from ..utils import log


@log('Decomposition: t-SNE')
def tsne(X, metric, dimensions):
    return TSNE(n_components=dimensions, learning_rate=100, perplexity=15, metric=metric).fit(X)


@log('Decomposition: MDS')
def mds(X, metric, dimensions):
    X = pairwise_distances(X, metric=metric)
    return MDS(n_components=dimensions, dissimilarity='precomputed', random_state=42, n_jobs=-1).fit(X)


reduction_models = {
    'tsne': tsne,
    'mds': mds
}


def dimension_reduction(X, attrs):
    """Dimension reduction."""
    decomp = attrs['decomposition']
    metric = attrs['decompositionMetric']
    dimensions = attrs['dimensionsNumber']

    model = reduction_models[decomp](X, metric, dimensions)
    return model.embedding_

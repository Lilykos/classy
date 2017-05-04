from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from gensim.parsing.preprocessing import strip_punctuation, stem
from ..utils import parallelize, log

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

stop = lambda attrs: stopwords if attrs['stop'] else None
norm = lambda attrs: None if attrs['norm'] == 'none' else attrs['norm']


@log('Creating text vectors.')
def get_vectors(text, attrs):
    text = parallelize(text, _strip_punct)          # remove punctuation
    if attrs['stem']:
        text = parallelize(text, _stem)             # stem if in options

    X = get_vectorizer(attrs).fit_transform(text)   # vectorize
    return X.toarray()


def get_vectorizer(attrs):
    """Returna a vectorizer with the user options."""
    stop_ = stop(attrs)
    norm_ = norm(attrs)
    features = int(attrs['featureNumber'])
    vectorizers = {
        'hashing': HashingVectorizer(n_features=features, non_negative=True, norm=norm_, stop_words=stop_),
        'tfidf': TfidfVectorizer(max_features=features, norm=norm_, stop_words=stop_),
        'count': CountVectorizer(max_features=features, stop_words=stop_)
    }
    return vectorizers[attrs['vectorizer']]


def _stem(row):
    """Process a single item, to be used in parallel."""
    return stem(row)


def _strip_punct(row):
    """Remove punctuation."""
    return strip_punctuation(row)

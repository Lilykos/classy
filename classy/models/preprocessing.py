from sklearn.base import TransformerMixin
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


class PreprocessTransformer(TransformerMixin):
    @log('Preprocessing: Stemming')
    def _stem(self, text):
        """Process a single item, to be used in parallel."""
        return parallelize(text, stem)

    @log('Preprocessing: Strip punctuation')
    def _strip_punct(self, text):
        """Remove punctuation."""
        return parallelize(text, strip_punctuation)

    def _remove_stopwords(self, text):
        return ''.join([word.lower() for word in text.split()
                        if word not in stopwords
                        and len(word) > 1])

    def __init__(self, attrs):
        self.stem = attrs['stem']
        self.stop = attrs['stop']

    def fit(self, y=None):
        return self

    def transform(self, text, y=None):
        text = self._strip_punct(text)
        text = parallelize(text, self._remove_stopwords) if self.stop else text
        return self._stem(text) if self.stem else text

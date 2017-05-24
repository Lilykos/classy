from collections import Counter
import time
import matplotlib.pyplot as plt
from matplotlib_venn_wordcloud import venn2_wordcloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from scikitplot.plotters import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve

from .models import classifier_names, ClassifierSelector, VectorTransformer, \
    DecompositionTransformer, PreprocessTransformer
from .utils import get_data_from_files


def venn_wordcloud(sets_of_words):
    pass


def make_plots(y_test, y_pred, y_prob, algorithm, timestamp):
    def _save_and_close(type):
        plt.savefig('static/img/{}/{}-{}.png'.format(type, algorithm, timestamp), dpi=200)
        plt.close('all')

    size = (20, 20)
    name = classifier_names[algorithm]

    plot_confusion_matrix(y_test, y_pred, normalize=True, figsize=size, title_fontsize=40, text_fontsize=30, title=name)
    _save_and_close('cm')

    if y_prob is not None:
        plot_precision_recall_curve(y_test, y_prob, figsize=size, title_fontsize=40, text_fontsize=25, title=name)
        _save_and_close('precrec')

        plot_roc_curve(y_test, y_prob, figsize=size, title_fontsize=40, text_fontsize=25, title=name)
        _save_and_close('roc')


def update_binary(result, y_test, y_pred, pos):
    prec, rec, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=pos)
    result.update({
        'F1 Score': '{:.4f}'.format(f),
        'Recall': '{:.4f}'.format(rec),
        'Prec': '{:.4f}'.format(prec),
    })
    return result


def update_multiclass(result, y_test, y_pred):
    prec_ma, rec_ma, f_ma, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    prec_mi, rec_mi, f_mi, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
    result.update({
        'F1 Score (mi)': '{:.4f}'.format(f_mi),
        'Recall (mi)': '{:.4f}'.format(rec_mi),
        'Prec (mi)': '{:.4f}'.format(prec_mi),
        'F1 Score (ma)': '{:.4f}'.format(f_ma),
        'Recall (ma)': '{:.4f}'.format(rec_ma),
        'Prec (ma)': '{:.4f}'.format(prec_ma)
    })
    return result


def get_classification_results(attrs):
    timestamp = time.time()
    df = get_data_from_files()
    df['processed'] = PreprocessTransformer(attrs).transform(df['text'])

    pipeline = make_pipeline(VectorTransformer(attrs), DecompositionTransformer(attrs))
    features = pipeline.transform(df['processed'])

    X_train, X_test, y_train, y_test = train_test_split(features, df['label'],
                                                        test_size=0.3, random_state=42, stratify=df['label'])

    results = []
    for algorithm in attrs['algorithms']:
        model, y_pred, y_prob = ClassifierSelector(attrs, algorithm, X_test).transform(X_train, y_train)

        result = {
            'Name': classifier_names[algorithm],
            'Acc. (Train)': '{:.4f}'.format(model.score(X_train, y_train)),
            'Acc. (Test)': '{:.4f}'.format(model.score(X_test, y_test)),
        }

        # update for multiclass/binary specific
        result = update_binary(result, y_test, y_pred, model.classes_[1]) \
            if attrs['labelsType'] == 'binary' \
            else update_multiclass(result, y_test, y_pred)
        results.append(result)

        make_plots(y_test, y_pred, y_prob, algorithm, timestamp)

    # venn word diagram
    sets = []
    for label in df['label'].unique():
        docs = df[df['label'] == label]['processed'].values
        text = ' '.join(docs)
        counter = Counter(text.split())
        sets.append(set(
            [item[0] for item in counter.most_common(30)]
        ))

    venn2_wordcloud(sets)
    plt.savefig('static/img/venn/venn_words.png', dpi=200)
    return results, timestamp

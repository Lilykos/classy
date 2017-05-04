import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from scikitplot.plotters import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve

from .models import classifier_names, classifiers, dimension_reduction, get_vectors
from .utils import log, get_data_from_files


def make_plots(y_test, y_pred, y_prob, algorithm):
    plot_confusion_matrix(y_test, y_pred, normalize=True, figsize=(20, 20), title_fontsize='40',
                          text_fontsize='30', title=classifier_names[algorithm])
    plt.savefig('static/img/cm/' + algorithm + '.png', dpi=200)
    plt.close('all')

    plot_precision_recall_curve(y_test, y_prob, figsize=(20, 20), title_fontsize='40',
                                text_fontsize='25', title=classifier_names[algorithm])
    plt.savefig('static/img/prec_rec/' + algorithm + '.png', dpi=200)
    plt.close('all')

    plot_roc_curve(y_test, y_prob, figsize=(20, 20), title_fontsize='40',
                   text_fontsize='25', title=classifier_names[algorithm])
    plt.savefig('static/img/roc/' + algorithm + '.png', dpi=200)
    plt.close('all')


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


@log('Classification results.')
def get_classification_results(attrs):
    text, labels = get_data_from_files()
    features = get_vectors(text, attrs)

    if attrs['decomposition'] != 'none':
        features = dimension_reduction(features, attrs)

    # train/test split of the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    results = []
    for algorithm in attrs['algorithms']:
        # hack for NB/ Multinomial NB
        if algorithm == 'nb' and attrs['labelsType'] != 'binary':
            algorithm = 'multinomialnb'

        model = classifiers[algorithm](X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        result = {
            'name': classifier_names[algorithm],
            'labels': np.unique(y_test).tolist(),
            'Accuracy': '{:.4f}'.format(accuracy_score(y_test, y_pred)),
        }

        # update for multiclass/binary specific
        result = update_binary(result, y_test, y_pred, model.classes_[1]) \
            if attrs['labelsType'] == 'binary' \
            else update_multiclass(result, y_test, y_pred)
        results.append(result)

        make_plots(y_test, y_pred, y_prob, algorithm)
    return results

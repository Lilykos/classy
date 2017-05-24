import sys
import os
import logging
import json
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

OK_200 = json.dumps({'success': True}), 200, {'ContentType': 'application/json'}

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s\t%(message)s',
                    datefmt='(%Y-%m-%d %H:%M:%S)')
logger = logging


def log(msg):
    """Logger decorator."""
    def decorator(func):
        def wrapper(*args):
            logger.info(msg)
            return func(*args)
        return wrapper
    return decorator


@log('Input: Creating folders.')
def create_folders(local_dir):
    """Create the required folders for the plots."""
    paths = [os.path.join(local_dir, path) for path in ['static/img/cm', 'static/img/roc',
                                                        'static/img/precrec', 'static/img/venn']]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(path)


@log('Input: Saving files.')
def save_files(request):
    """Save the data files (using standard names)."""
    text_file = request.files['textFile']
    labels_file = request.files['labelsFile']

    text_file.save('text_data.txt')
    labels_file.save('labels_data.txt')


@log('Input: Retrieving data.')
def get_data_from_files():
    """Load the data files."""
    data = {
        'text': list(open('text_data.txt', 'r')),
        'label': list(open('labels_data.txt', 'r'))
    }
    return pd.DataFrame(data)


def parallelize(items, func):
    """Parallelize operation."""
    parallel = Parallel(n_jobs=-1, backend='multiprocessing', verbose=1)
    return parallel(delayed(func)(item) for item in items)

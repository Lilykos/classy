import json
import os
from flask import Flask, render_template, request, jsonify
from flask_script import Manager, Server
from classy import logger, OK_200, save_files, get_classification_results, nocache, create_folders

app = Flask(__name__)


@app.route('/')
@nocache
def index():
    return render_template('index.html')


@app.route('/load_data', methods=['POST'])
@nocache
def load():
    save_files(request)
    return OK_200


@app.route('/classify', methods=['POST'])
@nocache
def classify():
    attrs = json.loads(request.form.get('data'))
    results = get_classification_results(attrs)
    return jsonify(**{'results': results,
                      'confusion_matrix': render_template('plots/templ.html', algorithms=attrs['algorithms'], plot='cm'),
                      'roc': render_template('plots/templ.html', algorithms=attrs['algorithms'], plot='roc'),
                      'precrec': render_template('plots/templ.html', algorithms=attrs['algorithms'], plot='precrec')})


manager = Manager(app)
manager.add_command('run', Server(use_debugger=True))

if __name__ == '__main__':
    logger.info(
        r"""
           ____  _
          / ___|| |  __ _  ___  ___  _   _
         | |    | | / _` |/ __|/ __|| | | |
         | |___ | || (_| |\__ \\__ \| |_| |
          \____||_| \__,_||___/|___/ \__, |
                                     |___/
        """)

    local_dir = os.path.dirname(__file__)
    create_folders(local_dir)

    manager.run()

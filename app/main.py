# main.py

from flask import Blueprint, render_template, request, abort, redirect, url_for, Flask, Response
from flask_login import login_required, current_user
from .models import clean_text, get_prediction, new_model, clusterize_data
from .models import Preferences
import json
import os
import logging
import pickle

logging.basicConfig(format='%(asctime)s %(levelname)s * %(message)s', filename='app/app.log', filemode='w', level=logging.INFO)

main = Blueprint('main', __name__)

@main.route('/')
@login_required
def index():
    return render_template('clustering.html')

@main.route('/profile')
@login_required
def profile():

    preferences = Preferences.query.filter_by(id=current_user.id).first()

    return render_template('profile.html', name=current_user.name, email=current_user.email, entreprise=preferences.entreprise, genre=preferences.genre)

@main.route('/clustering', methods=['GET','POST'])
@login_required
def clustering():

    if request.method == 'GET':

        return render_template('clustering.html')

    elif request.method == 'POST':

        logging.info(request.form)

        if 'radios' in request.form.keys() and request.form['radios'] == 'kmeans':

            args={}
            args['k'] = request.form['n_clusters_Kmeans_slider'] if 'n_clusters_Kmeans_slider' in request.form.keys() else 10
            args['check_Kmeans'] = request.form['check_Kmeans'] if 'check_Kmeans' in request.form.keys() else False

            args['k'] = 10 if args['k'] == '' else int(args['k'])

            # KMEANS

            logging.info('KMEANS chosen')
            logging.info(args)

            data = clusterize_data('kmeans', args)

            return render_template('clustering_results.html', data=data)

        elif 'radios' in request.form.keys() and request.form['radios'] == 'dbscan':

            # DBSCAN

            args={}
            args['eps'] = request.form['max_epsilon_DBSCAN_slider'] if 'max_epsilon_DBSCAN_slider' in request.form.keys() else 1
            args['min_samples'] = request.form['min_samples_DBSCAN_slider'] if 'min_samples_DBSCAN_slider' in request.form.keys() else 10
            args['check_DBSCAN'] = request.form['check_DBSCAN'] if 'check_DBSCAN' in request.form.keys() else False

            args['eps'] = 1 if args['eps'] == '' else float(args['eps'])
            args['min_samples'] = 10 if args['min_samples'] == '' else int(args['min_samples'])

            logging.info('DBSCAN chosen')
            logging.info(args)

            data = clusterize_data('dbscan', args)

            return render_template('clustering_results.html', data=data)

        elif 'radios' in request.form.keys() and request.form['radios'] == 'som':

            # DBSCAN

            args={}
            args['x'] = request.form['dim_SOM_1'] if 'dim_SOM_1' in request.form.keys() else 1
            args['y'] = request.form['dim_SOM_2'] if 'dim_SOM_2' in request.form.keys() else 1
            args['sigma'] = request.form['sigma_SOM_number'] if 'sigma_SOM_number' in request.form.keys() else 1
            args['check_SOM'] = request.form['check_SOM'] if 'check_SOM' in request.form.keys() != '' else False

            args['x'] = 1 if args['x']=='' else int(args['x'])
            args['y'] = 1 if args['y']=='' else int(args['y'])
            args['sigma'] = 1 if args['sigma']=='' else float(args['sigma'])

            logging.info('SOM chosen')
            logging.info(args)

            data = clusterize_data('som', args)

            return render_template('clustering_results.html', data=data)

        else:

            logging.error('No clustering method selected')

            return render_template('clustering.html')


@main.route('/avis', methods=['GET'])
@login_required
def avis():

    list_models = os.listdir('app/static/models_nlp')

    return render_template('avis.html', list_models=list_models)

@main.route('/avis', methods=['POST'])
@login_required
def process_text_from_form():

    text = request.form['avis']
    model_name = request.form['model_name']
    cleaned_text = clean_text(text)
    pred = get_prediction(cleaned_text, model_name)

    maru = True if pred==1 else 0

    return render_template('result_avis.html', maru = maru, text=text)

@main.route('/ameliorer', methods=['GET', 'POST'])
@login_required
def ameliorer():

    if request.method == 'POST':

        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            
            myfile = uploaded_file.read()
            string_value = myfile.decode("utf-8")

            new_model(string_value)

            logging.info('Nouveau modèle créé')

        else:
            logging.error('Aucun fichier uploadé')

        return redirect(url_for('main.avis'))

    return render_template('ameliorer.html')


# models.py

from flask_login import UserMixin
from . import db
import numpy as np
import pandas as pd
import time
import re
from nltk.corpus import stopwords
import pickle
import os
import json

import logging

import json
import numpy as np
from sqlalchemy import create_engine, engine
from sqlalchemy_utils import database_exists, create_database
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
import matplotlib
import matplotlib.pyplot as plt
from minisom import MiniSom

logging.basicConfig(format='%(asctime)s %(levelname)s * %(message)s', filename='app/app.log', filemode='w', level=logging.INFO)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

class Preferences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    genre = db.Column(db.String(100))
    entreprise = db.Column(db.String(100))

# Cleaning function

#Remove Urls and HTML links
def remove_urls(text):
    url_remove = re.compile(r'https?://\S+|www\.\S+')
    return url_remove.sub(r'', text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Lower casing
def lower(text):
    low_text= text.lower()
    return low_text

# Number removal
def remove_num(text):
    remove= re.sub(r'\d+', '', text)
    return remove

#Remove stopwords & Punctuations
def remove_stopwords(text,STOPWORDS):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    
def remove_punct(text):
    punct = re.sub(r"[^\w\s\d]","", text)
    return punct

def clean_text(text):

    STOPWORDS = set(stopwords.words('english'))

    cleaned = remove_urls(text) # Remove Urls
    cleaned = remove_html(cleaned) # Remove HTML links
    cleaned = remove_punct(cleaned) # Remove Punctuations
    cleaned = lower(cleaned) # Lower casing
    cleaned = remove_num(cleaned) # Remove numbers
    cleaned = remove_stopwords(cleaned,STOPWORDS) # Remove stopwords

    return cleaned

def process_text(text):

    text = text.replace(' ','_')
    text = text.upper()

    return text

def read_pickle(file_path):
    objects = []
    with (open(file_path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    return objects[0]

def get_prediction(cleaned_text, model_name):

    path_model = 'app/static/models_nlp/'+model_name

    with open('app/static/models_nlp/nlp_model', 'rb') as f:
        count_vec, tfidf_transformer, clf = pickle.load(f)

    text_count = count_vec.transform([cleaned_text])
    text_tfidf = tfidf_transformer.transform(text_count)
    pred = clf.predict(text_tfidf)[0]
    probas = clf.predict_proba(text_tfidf)[0]

    if probas[pred]<0.7:
        logging.warning('Prediction is not reliable')

    return pred

def new_model(json_file):

    df_new_avis = pd.read_json(json_file, orient='index')
    df_new_avis.rename(columns={'text':'Review', 'label':'Recommanded'}, inplace=True)

    df = pd.read_csv('app/static/data/reviews.csv').fillna('') # Reads csv with data
    df.rename({'Recommended IND':'Recommended'}, axis=1, inplace=True) # Rename column 
    df['Review'] = df['Title'] + ' ' + df['Review Text'] # Concatenate title and review text

    df = df[['Recommended', 'Review']]
    print(len(df))

    df = pd.concat([df,df_new_avis])
    print(len(df))

    X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Recommended'], test_size=0.25, random_state=27)

    # Gather X_train and X_test into a dataframe
    df_X = pd.concat([X_train, y_train], axis=1)

    df_majority = df_X[(df_X['Recommended']==1)]
    df_minority = df_X[(df_X['Recommended']==0)]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                    replace=True,    # sample with replacement
                                    n_samples=len(df_majority) , # to match majority class
                                    random_state=42)  # reproducible results
                                    
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_minority_upsampled, df_majority])

    X_train = df_upsampled['Review']
    y_train = df_upsampled['Recommended']

    count_vec = CountVectorizer()
    X_train_counts = count_vec.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = SGDClassifier(loss='log_loss', penalty='l2', max_iter=30)
    clf.fit(X_train_tfidf, y_train)

    nb_models = len(os.listdir('app/static/models'))

    file_name = 'app/static/models_nlp/nlp_model_' + str(nb_models+1)

    with open(file_name, 'wb') as f:
        pickle.dump((count_vec, tfidf_transformer, clf), f)

    return 

# modules
def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# test
def test_db_config(db_config):
    """
    Test if the database config is valid
    """

    s = f"postgresql://{db_config['user']}:{db_config['pass']}@{db_config['host']}:{db_config['port']}/postgres"
    db_engine = create_engine(s)

    assert type(db_engine.connect()) == engine.base.Connection

def clusterize_data(model, args):

    # read config
    db_config = read_json("app/static/keys/database_config.json")

    # test database connection with config
    test_db_config(db_config)
    print("- Connection to db created successfully with current config")

    # Create engine. Make sure the current IP address is withelisted on Azure.
    s = f"postgresql://{db_config['user']}:{db_config['pass']}@{db_config['host']}:{db_config['port']}/an_database"
    an_engine = create_engine(s)

    query = """
        SELECT * FROM profiles
        JOIN profiles_kpis ON profiles.profile_id = profiles_kpis.profile_id
    """

    df_profiles = pd.read_sql(query,an_engine)
    df_profiles.drop('profile_id',axis=1,inplace=True)

    df_profiles_encoded = df_profiles.copy()

    # Encodage des variables catégorielles
    df_profiles_encoded['age'] = df_profiles_encoded['age'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2)
    df_profiles_encoded['revenus'] = df_profiles_encoded['revenus'].replace({'CSP moins':20000,'CSP moyen':40000,'CSP plus':60000})
    df_profiles_encoded[['Femme','Homme']] = pd.get_dummies(df_profiles_encoded['genre'], prefix='')
    df_profiles_encoded[['Rural','Très rural','Très urbain','Urbain']] = pd.get_dummies(df_profiles_encoded['localisation'], prefix='')

    # Sélection des colonnes
    df_profiles_encoded = df_profiles_encoded[[
        'age',
        'revenus',
        'Femme',
        'Homme',
        'Rural',
        'Très rural',
        'Très urbain',
        'Urbain',
        'ca',
        'commandes',
        'panier_moyen',	'prix_moyen_article',
        'clients',
        'nb_articles_moyen',
        'frequence'
    ]]

    df_profiles_scaled = df_profiles_encoded.copy()

    # Standardisation des variables quantitatives
    cols_to_standardize = [
        'panier_moyen',
        'prix_moyen_article',
        'nb_articles_moyen',
        'frequence'
    ]

    scaler = preprocessing.StandardScaler()
    df_profiles_scaled[cols_to_standardize] = scaler.fit_transform(df_profiles_scaled[cols_to_standardize])

    # Normalisation des variables quantitatives
    cols_to_normalize = [
        'age',
        'revenus',
        'ca',
        'commandes',
        'clients'
    ]

    scaler = preprocessing.MinMaxScaler()
    df_profiles_scaled[cols_to_normalize] = scaler.fit_transform(df_profiles_scaled[cols_to_normalize])

    # Clustering

    if model == 'kmeans':

        if args['check_Kmeans']=='on':
        
            kmeans = read_pickle('app/static/models_clustering/kmeans_model')

            kmeans.predict(df_profiles_scaled)

        else:

            k = args['k']
            kmeans = KMeans(
                n_clusters=k,
                random_state=0)

            kmeans.fit_predict(df_profiles_scaled)

        df_profiles['cluster'] = kmeans.labels_

    elif model == 'dbscan':

        if args['check_DBSCAN']=='on':

            dbscan = read_pickle('app/static/models_clustering/dbscan_model')

        else :

            dbscan = DBSCAN(
                eps=args['eps'],
                min_samples=args['min_samples'],
                metric='euclidean',
                metric_params=None,
                algorithm='auto',
                leaf_size=30,
                p=None,
                n_jobs=None)

            dbscan.fit(df_profiles_scaled)

        df_profiles['cluster'] = dbscan.labels_

        logging.info(df_profiles['cluster'].value_counts())

    elif model == 'som':

        data = df_profiles_scaled.values

        if args['check_SOM']=='on':

            som = read_pickle('app/static/models_clustering/som_model')

            winner_coordinates = np.array([som.winner(x) for x in data]).T

            cluster_index = np.ravel_multi_index(winner_coordinates, (2,3))

        else:

            som = MiniSom(
                args['x'],
                args['y'],
                15,
                sigma=args['sigma'], 
                learning_rate=0.5,
                neighborhood_function='gaussian',
                random_seed=10
            )
            som.random_weights_init(data = data)
            som.train_random(data = data, num_iteration = 1000)

            winner_coordinates = np.array([som.winner(x) for x in data]).T

            cluster_index = np.ravel_multi_index(winner_coordinates, (args['x'], args['y']))

        df_profiles['cluster'] = cluster_index

    else:

        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit_predict(df_profiles_scaled)

        df_profiles['cluster'] = kmeans.labels_

    ### Résultats

    dict_res = {}

    matplotlib.use('agg')

    cats = {
        'genre': ['Femme','Homme'],
        'age': ['18-24','25-34','35-44','45-54','55-64','65-74','75-99'],
        'localisation': ['Rural','Très rural','Très urbain','Urbain'],
        'revenus': ['CSP moins','CSP moyen','CSP plus']
    }

    for cluster in df_profiles['cluster'].unique():

        cluster_index = str(cluster+1)
        dict_res[cluster_index] = {}

        df_profiles_cluster = df_profiles[df_profiles['cluster']==cluster]

        dict_res[cluster_index]['total'] = len(df_profiles_cluster)

        dict_res[cluster_index]['types'] = {
            'age': 'string',
            'genre':'string',
            'revenus':'string',
            'localisation':'string',
            'ca':'number',
            'commandes':'number',
            'clients':'number',
            'panier_moyen':'number',
            'prix_moyen_article':'number',
            'nb_articles_moyen':'number',
            'frequence':'number'
        }

        for col in ['age','genre','revenus','localisation']:

            value_counts = df_profiles_cluster[col].value_counts().to_dict()

            for cat in cats[col]:

                if cat not in value_counts.keys():

                    value_counts[cat] = 0            

            dict_res[cluster_index][col] = [[key,val] for key,val in value_counts.items()]

        for col in ['ca', 'commandes', 'panier_moyen', 'prix_moyen_article', 'clients', 'nb_articles_moyen', 'frequence']:

            dict_res[cluster_index][col] = df_profiles_cluster[col].apply(lambda x: [x]).to_list()

    return dict_res
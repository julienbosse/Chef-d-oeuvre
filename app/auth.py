# auth.py

from flask import Blueprint, render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required
from .models import User, Preferences
from . import db
import os
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s * %(message)s', filename='app/app.log', filemode='w', level=logging.INFO)

auth = Blueprint('auth', __name__)

@auth.route('/login')
def login():
    return render_template('login.html')

@auth.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password): 
        flash('Vérifiez vos identifiants s\'il vous plaît.')
        return redirect(url_for('auth.login')) 

    login_user(user, remember=remember)
    logging.info(email + ' Utilisateur connecté ')
    return redirect(url_for('main.profile'))

@auth.route('/signup')
def signup():
    return render_template('signup.html')

@auth.route('/signup', methods=['POST'])
def signup_post():

    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')
    genre = request.form.get('genre') 
    entreprise = request.form.get('entreprise')

    user = User.query.filter_by(email=email).first() 

    if user: 
        flash('L\'email existe déjà.')
        logging.error(email+ ' L\'email existe déjà. ')

        return redirect(url_for('auth.signup'))

    new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))
    db.session.add(new_user)
    db.session.commit()

    user = User.query.filter_by(email=email).first()

    new_preferences = Preferences(genre=genre, entreprise=entreprise, id=user.id)
    db.session.add(new_preferences)
    db.session.commit()
    
    logging.info(email + ' Nouvel utilisateur créé ')

    return redirect(url_for('auth.login'))

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    list_models = os.listdir('app/static/models')
    list_models.remove('nlp_model')
    # delete models
    for model in list_models:
        os.remove(os.path.join('app/static/models', model))
    return redirect(url_for('main.index'))
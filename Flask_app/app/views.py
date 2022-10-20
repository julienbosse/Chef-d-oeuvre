from flask import render_template, request, abort, redirect, url_for, Flask, Response
from app import app
from app import models
from .models import clean_text, get_prediction




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model', methods=['GET', 'POST'])
def model():
    return render_template('model.html')

@app.route('/avis', methods=['GET'])
def avis():
    return render_template('avis.html')

@app.route('/avis', methods=['POST'])
def process_text_from_form():
    text = request.form['avis']
    cleaned_text = clean_text(text)
    pred = get_prediction(cleaned_text)

    maru = True if pred==1 else 0

    return render_template('result_avis.html', maru = maru)

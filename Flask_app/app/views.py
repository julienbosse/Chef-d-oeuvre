from flask import render_template, request, abort, redirect, url_for, Flask, Response
from app import app
from app import models


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model', methods=['GET', 'POST'])
def model():
    return render_template('model.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')

@app.route('/avis', methods=['GET', 'POST'])
def avis():
    return render_template('avis.html')


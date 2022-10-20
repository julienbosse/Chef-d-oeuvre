# init.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
import json
from sqlalchemy import create_engine, engine
import logging

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    Bootstrap(app)

    db_config = read_json("app/static/keys/database_config.json")
    s = f"postgresql://{db_config['user']}:{db_config['pass']}@{db_config['host']}:{db_config['port']}/op_database"

    app.config['SECRET_KEY'] = '9OLWxND4o83j4K4iuopO'
    app.config['SQLALCHEMY_DATABASE_URI'] = s
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', filename='app/app.log', filemode='w', level=logging.INFO)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.login_message= 'Veuillez vous connecter.'
    login_manager.init_app(app)    

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    # blueprint for auth routes in our app
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    # blueprint for non-auth parts of app
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
# from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__) # Holds the Flask instance
app.config.from_object('config.DevelopmentConfig')
db = SQLAlchemy(app)

from api import views, models

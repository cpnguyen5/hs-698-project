from flask import Flask
from flask_sqlalchemy import SQLAlchemy
# from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__) # Holds the Flask instance
app.config.from_object('config')
db = SQLAlchemy(app)

from api import views

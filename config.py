import os

basedir=os.path.abspath(os.path.dirname(__file__))
db_path=os.path.join(basedir, 'api', 'dataset')
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(db_path, 'cms.db') #required by Flask-SQLAlchemy extension -- path to db file
SQLALCHEMY_TRACK_MODIFICATIONS = False
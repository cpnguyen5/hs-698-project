import os

# basedir = os.path.abspath(os.path.dirname(__file__))
# db_path = os.path.join(basedir, 'api', 'dataset', 'cms.db')
# SQLALCHEMY_DATABASE_URI = 'sqlite:///' + db_path  # required by Flask-SQLAlchemy extension -- path to db file
# SQLALCHEMY_TRACK_MODIFICATIONS = False


class BaseConfig(object):

    basedir=os.path.abspath(os.path.dirname(__file__))
    db_path=os.path.join(basedir, 'api', 'dataset', 'cms.db')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + db_path #required by Flask-SQLAlchemy extension -- path to db file
    # SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # print SQLALCHEMY_DATABASE_URI

class DevelopmentConfig(BaseConfig):
    DEBUG=True


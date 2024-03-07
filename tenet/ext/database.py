from sqlalchemy.exc import OperationalError

from arepo.db import DatabaseConnection as Database
from tenet.core.exc import TenetError


def init_database(app):
    try:
        app.log.info(f"Connecting to database")
        uri = app.config.get('database', 'uri')
        database = Database(uri)
        app.extend('db', database)
    except OperationalError as oe:
        raise TenetError(oe)


def load(app):
    app.hook.register('post_setup', init_database)

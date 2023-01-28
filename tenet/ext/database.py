from sqlalchemy.exc import OperationalError

from tenet.data.database import Database
from tenet.core.exc import TenetError


def init_database(app):
    try:
        database = Database(dialect=app.get_config('dialect'), username=app.get_config('username'),
                            password=app.get_config('password'), host=app.get_config('host'),
                            port=app.get_config('port'), database=app.get_config('database'),
                            debug=app.config.get('log.colorlog', 'database'))

        app.extend('db', database)
        app.log.info(f"Connected to {app.get_config('dialect')}://{app.get_config('username')}@{app.get_config('host')}:{app.get_config('port')}/{app.get_config('database')}")
    except OperationalError as oe:
        raise TenetError(oe)


def load(app):
    app.hook.register('post_setup', init_database)

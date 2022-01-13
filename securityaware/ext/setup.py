from pathlib import Path


def init_workdir(app):
    # workdir = Path(app.get_config('workdir'))

    # if not workdir.exists():
    #    workdir.mkdir(parents=True)
    #    app.log.info(f"Created workdir {workdir}")

    # app.extend('workdir', workdir)
    pass


def load(app):
    app.hook.register('post_setup', init_workdir)

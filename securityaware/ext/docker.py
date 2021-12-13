
import docker
from docker.errors import NotFound

from securityaware.core.exc import SecurityAwareError


def bind_docker(app):
    try:
        docker_client = docker.from_env(timeout=10)
        assert docker_client.ping()
        app.extend('docker', docker_client)
    except AssertionError:
        raise SecurityAwareError("Could not connect to the Docker Client")


def init_volume(app):
    volume_name = app.get_config('docker')['volume']
    try:
        volume = app.docker.volumes.get(volume_name)
        app.log.info(f"Found '{volume.id}' volume.")
    except NotFound as nf:
        app.log.warning(str(nf))
        volume = app.docker.volumes.create(volume_name)
        app.log.info(f"Created volume {volume.id}")


def load(app):
    app.hook.register('post_setup', bind_docker)
    #app.hook.register('post_setup', init_volume)

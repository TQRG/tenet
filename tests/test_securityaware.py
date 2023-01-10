
from pytest import raises
from tenet.main import TenetTest

def test_tenet():
    # test tenet without any subcommands or arguments
    with TenetTest() as app:
        app.run()
        assert app.exit_code == 0


def test_tenet_debug():
    # test that debug mode is functional
    argv = ['--debug']
    with TenetTest(argv=argv) as app:
        app.run()
        assert app.debug is True


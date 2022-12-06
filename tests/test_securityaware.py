
from pytest import raises
from securityaware.main import SecurityAwareTest

def test_securityaware():
    # test securityaware without any subcommands or arguments
    with SecurityAwareTest() as app:
        app.run()
        assert app.exit_code == 0


def test_securityaware_debug():
    # test that debug mode is functional
    argv = ['--debug']
    with SecurityAwareTest(argv=argv) as app:
        app.run()
        assert app.debug is True


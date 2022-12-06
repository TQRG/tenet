
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

# TODO: Fix test
# def test_command1():
#     # test command1 without arguments
#     argv = ['command1']
    # with SecurityAwareTest(argv=argv) as app:
    #     app.run()
    #     data,output = app.last_rendered
    #     assert data['foo'] == 'bar'
    #     assert output.find('Foo => bar')


    # # test command1 with arguments
    # argv = ['command1', '--foo', 'not-bar']
    # with SecurityAwareTest(argv=argv) as app:
    #     app.run()
    #     data,output = app.last_rendered
    #     assert data['foo'] == 'not-bar'
    #     assert output.find('Foo => not-bar')

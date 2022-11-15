from cement import App, TestApp
from cement.core.exc import CaughtSignal, InterfaceError

from .core.exc import SecurityAwareError
from .controllers.base import Base
from .controllers.plugin import Plugin
from .controllers.cwe import CWE

from securityaware.core.interfaces import HandlersInterface
from securityaware.handlers.github import GithubHandler
from securityaware.handlers.command import CommandHandler
from securityaware.handlers.runner import MultiTaskHandler
from securityaware.handlers.workflow import WorkflowHandler
from securityaware.handlers.plugin_loader import PluginLoader
from securityaware.handlers.container import ContainerHandler
from securityaware.handlers.code_parser import CodeParserHandler
from securityaware.handlers.cwe_list import CWEListHandler
from securityaware.handlers.file_parser import FileParserHandler


class SecurityAware(App):
    """SecurityAware primary application."""

    class Meta:
        label = 'securityaware'

        # call sys.exit() on close
        exit_on_close = True

        # load additional framework extensions
        extensions = [
            'securityaware.ext.setup',
            'securityaware.ext.docker',
            'yaml',
            'colorlog',
            'jinja2',
        ]

        # configuration files
        config_files = ['~/.securityaware/config/abstractions.yml', '~/.securityaware/config/mappings.yml']

        # configuration handler
        config_handler = 'yaml'

        # configuration file suffix
        config_file_suffix = '.yml'

        # set the log handler
        log_handler = 'colorlog'

        # set the output handler
        output_handler = 'jinja2'

        # set the handler for loading plugins
        plugin_handler = 'plugin_loader'

        interfaces = [
            HandlersInterface
        ]

        # register handlers
        handlers = [
            Base, Plugin, CWE, PluginLoader, ContainerHandler, CommandHandler, WorkflowHandler, CodeParserHandler,
            MultiTaskHandler, GithubHandler, CWEListHandler, FileParserHandler
        ]

    def get_config(self, key: str):
        if self.config.has_section(self.Meta.label):
            if key in self.config.keys(self.Meta.label):
                return self.config.get(self.Meta.label, key)

        return None

    def get_plugin_handler(self, name: str, setup: bool = True):
        """
            Gets the handler associated to the plugin

            :param name: label of the plugin
            :param setup: setup of the plugin
            :return: handler for the plugin
        """

        try:
            if name not in self.plugin.get_enabled_plugins():
                self.log.error(f"Plugin {name} not enabled.")
                exit(1)

            if name not in self.plugin.get_loaded_plugins():
                self.log.error(f"Plugin {name} not loaded.")
                exit(1)

            if name not in [el.Meta.label for el in self.handler.list('handlers')]:
                return self.handler.resolve('handlers', name)

            return self.handler.get('handlers', name, setup=setup)
        except InterfaceError as ie:
            self.log.error(str(ie))
            exit(1)


class SecurityAwareTest(TestApp, SecurityAware):
    """A sub-class of SecurityAware that is better suited for testing."""

    class Meta:
        label = 'securityaware'


def main():
    with SecurityAware() as app:

        if not app.config.has_section('mappings'):
            app.log.error(f"Views mappings not found, make sure ~/.securityaware/config/mappings.yml exists")
            exit(1)

        if not app.config.has_section('abstractions'):
            app.log.error(f"CWE abstractions not found, make sure ~/.securityaware/config/abstractions.yml exists")
            exit(1)

        try:
            app.run()

        except AssertionError as e:
            print('AssertionError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except SecurityAwareError as e:
            app.log.error('SecurityAwareError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                app.log.error(traceback.format_exc())

        except CaughtSignal as e:
            # Default Cement signals are SIGINT and SIGTERM, exit 0 (non-error)
            print('\n%s' % e)
            app.exit_code = 0


if __name__ == '__main__':
    main()

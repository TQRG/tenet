import pkg_resources

from cement import App, TestApp
from cement.core.exc import CaughtSignal, InterfaceError

from .core.exc import TenetError
from .controllers.base import Base
from .controllers.plugin import Plugin
from .controllers.cwe import CWE

from tenet.core.interfaces import HandlersInterface, PluginsInterface
from tenet.handlers.github import GithubHandler
from tenet.handlers.command import CommandHandler
from tenet.handlers.runner import MultiTaskHandler
from tenet.handlers.workflow import WorkflowHandler
from tenet.handlers.pipeline import PipelineHandler
from tenet.handlers.plugin_loader import PluginLoader
from tenet.handlers.container import ContainerHandler
from tenet.handlers.code_parser import CodeParserHandler
from tenet.handlers.cwe_list import CWEListHandler
from tenet.handlers.file_parser import FileParserHandler
from tenet.handlers.sampling import SamplingHandler


class Tenet(App):
    """Tenet primary application."""

    class Meta:
        def get_absolute_path(package, file_path):
            return pkg_resources.resource_filename(package, file_path)
        
        label = 'tenet'

        # call sys.exit() on close
        exit_on_close = True

        # load additional framework extensions
        extensions = [
            'tenet.ext.setup',
            'tenet.ext.docker',
            'yaml',
            'colorlog',
            'jinja2',
        ]

        # configuration files
        config_files = [
            get_absolute_path(label, 'config/abstractions.yml'), 
            get_absolute_path(label, 'config/mappings.yml'),
            get_absolute_path(label, 'config/keywords.yml'),
            get_absolute_path(label, 'config/tenet.yml')
        ]

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
            HandlersInterface, PluginsInterface
        ]

        # register handlers
        handlers = [
            Base, Plugin, CWE, PluginLoader, ContainerHandler, CommandHandler, WorkflowHandler, CodeParserHandler,
            MultiTaskHandler, GithubHandler, CWEListHandler, FileParserHandler, SamplingHandler, PipelineHandler
        ]

    def get_config(self, key: str):
        if self.config.has_section(self.Meta.label):
            if key in self.config.keys(self.Meta.label):
                return self.config.get(self.Meta.label, key)

        return None

    def get_plugin_handler(self, name: str):
        """
            Gets the handler associated to the plugin

            :param name: label of the plugin
            :return: handler for the plugin
        """

        try:

            if name not in self.plugin.get_loaded_plugins():
                self.plugin.load_plugin(name)

            plugin = self.handler.resolve('plugins', name)
            plugin.__init__()
            plugin._setup(self)
            return plugin

        except InterfaceError as ie:
            self.log.error(str(ie))
            exit(1)
        except TypeError as te:
            self.log.error(str(te))
            exit(1)


class TenetTest(TestApp, Tenet):
    """A sub-class of Tenet that is better suited for testing."""

    class Meta:
        label = 'tenet'


def main():
    with Tenet() as app:
        # TODO: These configurations integrate the package; 
        # therefore, the bellow errors are just helpfull
        # for development mode. Users shouldn't get this issue
        # since these config files are included in the package; 
        # so, should we have a flag for development mode?
        if not app.config.has_section('mappings'):
            app.log.error(f"Views mappings not found, make sure /tenet/config/mappings.yml exists")
            exit(1)

        if not app.config.has_section('abstractions'):
            app.log.error(f"CWE abstractions not found, make sure /tenet/config/abstractions.yml exists")
            exit(1)

        if not app.config.has_section('keywords'):
            app.log.error(f"Keywords not found, make sure /tenet/config/keywords.yml exists")
            exit(1)

        try:
            app.run()

        except AssertionError as e:
            print('AssertionError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except TenetError as e:
            app.log.error('TenetError > %s' % e.args[0])
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

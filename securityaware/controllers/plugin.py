import yaml
from pathlib import Path
from cement import Controller, ex

from securityaware.core.exc import SecurityAwareError


class Plugin(Controller):
    """
        Plugin Controller to handle plugin operations
    """

    class Meta:
        label = 'plugin'
        stacked_on = 'base'
        stacked_type = 'nested'

    @ex(
        help="Lists enabled plugins"
    )
    def enabled(self):
        for p in self.app.plugin.get_enabled_plugins():
            self.app.log.info(p)

    @ex(
        help="Installs plugin",
        arguments=[
            (['-p', '--path'], {'help': 'File path of the plugin.', 'required': True, 'type': str}),
            (['-f', '--force'], {'help': 'Overwrites existing plugins.', 'action': 'store_true'}),
            (['-n', '--name'], {'help': 'Name of the plugin (should match its label).', 'required': True, 'type': str})
        ]
    )
    def install(self):
        """
           Sub command for installing the plugin
        """

        plugin_file = Path(self.app.pargs.path)

        if not plugin_file.exists():
            self.app.log.error(f"{plugin_file} not found.")
            exit(1)

        if not plugin_file.is_file():
            self.app.log.error(f"{plugin_file} must be a file.")
            exit(1)

        plugin_key = f"plugin.{self.app.pargs.name}"

        # TODO: find a better way of doing this
        dest_plugin_file = Path(self.app.get_config('plugin_dir')) / (plugin_file.stem + '.py')
        dest_plugin_file = Path(dest_plugin_file.expanduser())

        if dest_plugin_file.exists():
            self.app.log.warning(f"Plugin file {dest_plugin_file} exists.")

            if not self.app.pargs.force:
                response = input("Do you want to overwrite it? (y/n) ")

                if response not in ["Yes", "Y", "y", "yes"]:
                    self.app.log.info(f"Input '{response}'. Exiting.")
                    exit(0)

        enabled_plugins = self.app.plugin.get_enabled_plugins()

        if self.app.pargs.name in enabled_plugins:
            self.app.log.warning(f"Plugin {self.app.pargs.name} is already enabled.")

        loaded_plugins = self.app.plugin.get_loaded_plugins()

        if self.app.pargs.name in loaded_plugins:
            self.app.log.warning(f"Plugin {self.app.pargs.name} is already loaded.")

        app_handlers = [el.Meta.label for el in self.app.handler.list('handlers')]
        reserved_names = set(app_handlers).difference(set(enabled_plugins+loaded_plugins))

        # check the name of existing handlers
        if self.app.pargs.name in reserved_names:
            raise SecurityAwareError(f"The name {self.app.pargs.name} can not be used.")

        with plugin_file.open(mode="r") as pf, dest_plugin_file.open(mode="w") as dpf:
            self.app.log.info(f"Writing plugin {plugin_file} file to {dest_plugin_file}")
            dpf.write(pf.read())

        for file in self.app._meta.config_files:
            path = Path(file)

            if not path.exists():
                continue

            with path.open(mode="r") as stream:
                configs = yaml.safe_load(stream)

                if 'securityaware' not in configs:
                    continue

            if plugin_key in configs:
                configs[plugin_key]['enabled'] = True
            else:
                configs[plugin_key] = {'enabled': True}

            with path.open(mode="w") as stream:
                yaml.safe_dump(configs, stream)
                self.app.log.info(f"Updated config file")
                break

    @ex(
        help="Uninstalls plugin",
        arguments=[
            (['-n', '--name'], {'help': 'Name of the plugin.', 'required': True, 'type': str})
        ]
    )
    def uninstall(self):
        """
            Removes plugin and associated files.
        """
        for file in self.app._meta.config_files:
            path = Path(file)

            if not path.exists():
                continue

            with path.open(mode="r") as stream:
                configs = yaml.safe_load(stream)

                if 'securityaware' not in configs:
                    continue

            plugin = f"plugins.{self.app.pargs.name}"

            if plugin in configs:
                del configs[plugin]

                # TODO: find a better way for doing this

                plugin_file = Path(self.app.get_config('plugin_dir')) / f"{self.app.pargs.name}.py"
                plugin_file = Path(plugin_file.expanduser())

                if plugin_file.exists():
                    plugin_file.unlink()
                    self.app.log.info(f"{plugin_file} deleted")

                with path.open(mode="w") as stream:
                    yaml.safe_dump(configs, stream)

                    self.app.log.info(f"removed {plugin} from configs")

                    break
            else:
                self.app.log.warning(f"{plugin} not found.")

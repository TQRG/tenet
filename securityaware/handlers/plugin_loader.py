from pathlib import Path

from cement.core.exc import FrameworkError
from cement.ext.ext_plugin import CementPluginHandler

from securityaware.core.exc import SecurityAwareError


class PluginLoader(CementPluginHandler):
    class Meta:
        label = 'plugin_loader'

    def __init__(self):
        super().__init__()

    def check(self, name: str, path: str):
        """
            Checks if plugin can be loaded
        """
        return super()._load_plugin_from_dir(name, path)

'''
    def _setup(self, app_obj):
        super()._setup(app_obj)

        for section in self.app.config.get_sections():
            try:
                kind, name = section.split('.')

                if kind != 'plugins':
                    continue

                try:
                    self.load_plugin(name)
                except FrameworkError as fe:
                    raise SecurityAwareError(str(fe))

                # loaded = name in self._loaded_plugins
                enabled = 'enabled' in self.app.config.keys(section) and self.app.config.get(section, 'enabled')
                self.app.log.warning(f'{name}_{enabled}')
                # if loaded and enabled:
                #    break
            except ValueError:
                continue
'''

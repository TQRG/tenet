from pathlib import Path

import schema
import yaml
from cement import Controller, ex
from cement.utils.version import get_version_banner
from ..core.version import get_version
from ..data.schema import parse_pipeline

VERSION_BANNER = """
Fine-grained approach to detect and patch vulnerabilities %s
%s
""" % (get_version(), get_version_banner())


class Base(Controller):
    class Meta:
        label = 'base'

        # text displayed at the top of --help output
        description = 'Fine-grained approach to detect and patch vulnerabilities'

        # text displayed at the bottom of --help output
        epilog = 'Usage: securityaware command1 --foo bar'

        # controller level arguments. ex: 'securityaware --version'
        arguments = [
            ### add a version banner
            (['-v', '--version'], {'action': 'version', 'version': VERSION_BANNER}),
            (['-vb', '--verbose'], {'help': 'Verbose output.', 'action': 'store_true'})
        ]

    def _default(self):
        """Default action if no sub-command is passed."""

        self.app.args.print_help()

    @ex(
        help='Runs a workflow',
        arguments=[
            (['-f', '--file'], {'help': 'Path to the pipeline config file', 'type': str, 'required': True}),
            (['-wd', '--workdir'], {'help': 'Path to the workdir. (overwrites the workdir in the config file',
                                    'type': str, 'required': False})
        ]
    )
    def pipeline(self):
        file = Path(self.app.pargs.file)

        if not file.exists():
            self.app.log.error(f"File {file} not found.")
            exit(1)

        if self.app.pargs.workdir:
            self.app.workdir = Path(self.app.pargs.workdir)

        with file.open(mode="r") as stream:
            try:
                pipeline = parse_pipeline(yaml.safe_load(stream))
            except schema.SchemaError as se:
                self.app.log.error(str(se))
                exit(1)

            for name, node in pipeline.nodes.items():
                handler = self.app.handler.get('handlers', name, setup=True)
                handler(node, self.app.workdir / name)

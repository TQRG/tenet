from pathlib import Path

import schema
import yaml, os
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
            (['-d', '--dataset'], {'help': 'Path to the input csv dataset file', 'type': str, 'required': True}),
            (['-wd', '--workdir'], {'help': 'Path to the workdir.', 'type': str, 'required': True}),
            (['-b', '--bind'], {'help': 'Docker directory path to bind (to workdir as a volume).',
                                'type': str, 'required': True})
        ]
    )
    def run(self):
        file = Path(self.app.pargs.file)
        dataset = Path(self.app.pargs.dataset)

        if not file.exists():
            self.app.log.error(f"File {file} not found.")
            exit(1)

        if not dataset.exists():
            self.app.log.error(f"Dataset {dataset} not found.")
            exit(1)

        self.app.extend('workdir', Path(self.app.pargs.workdir))

        if not self.app.workdir.exists():
            self.app.workdir.mkdir(parents=True)
            self.app.log.info(f"Created workdir {self.app.workdir}")

        self.app.extend('bind', self.app.pargs.bind)

        if not self.app.bind.startswith('/') or self.app.bind.startswith('~/'):
            self.app.log.error(f"Bind path {self.app.bind} is not a valid directory path.")
            exit(1)

        with file.open(mode="r") as stream:
            try:
                self.app.log.info(f"Parsing pipeline file: {file}")
                pipeline = parse_pipeline(yaml.safe_load(stream))
            except schema.SchemaError as se:
                self.app.log.error(str(se))
                exit(1)

            self.app.extend('pipeline', pipeline)
            workflow_handler = self.app.handler.get('handlers', 'workflow', setup=True)
            workflow_handler.load(dataset)
            workflow_handler(dataset)

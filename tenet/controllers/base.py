import os.path

import inquirer
import schema
import yaml

from pathlib import Path

from cement import Controller, ex
from tenet import __version__
from ..core.exc import TenetError
from ..data.schema import parse_pipeline

VERSION_BANNER = """
Fine-grained approach to detect and patch vulnerabilities (v%s)
""" % (__version__)


class Base(Controller):
    class Meta:
        label = 'base'

        # text displayed at the top of --help output
        description = 'Fine-grained approach to detect and patch vulnerabilities'

        # text displayed at the bottom of --help output
        epilog = 'Usage: tenet command1 --foo bar'

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
            (['-f', '--file'], {'help': 'Path to the pipeline config file', 'type': str, 'required': False}),
            (['-d', '--dataset'], {'help': 'Path to the input csv dataset file', 'type': str, 'required': False}),
            (['-sp', '--suppress-plot'], {'help': 'Suppresses plotting', 'action': 'store_true', 'required': False,
                                          'dest': 'suppress_plot'}),
            (['-t', '--threads'], {'help': 'Number of threads (overwrites the threads in config)', 'type': int,
                                   'required': False}),
            (['-wd', '--workdir'], {'help': 'Path to the workdir.', 'type': str, 'required': True}),
            (['-b', '--bind'], {'help': 'Docker directory path to bind (to workdir as a volume).',
                                'type': str, 'required': False})
        ]
    )
    def run(self):
        self.app.extend('workdir', Path(self.app.pargs.workdir))

        if not self.app.workdir.exists():
            self.app.workdir.mkdir(parents=True)
            self.app.log.info(f"Created workdir {self.app.workdir}")

        dataset = self._parse_dataset()
        pipeline = self._parse_pipeline()

        self.app.threads = self.app.pargs.threads if self.app.pargs.threads else self.app.get_config('local_threads')

        if self.app.pargs.bind:
            bind = self.app.pargs.bind
        else:
            bind = str(self.app.workdir.absolute()).replace(str(Path.home()), '')

        self.app.extend('bind', bind)

        #if not os.path.exists(self.app.bind):
        #    self.app.log.error(f"Bind path {self.app.bind} is not a valid directory path.")
        #    exit(1)

        with pipeline.open(mode="r") as stream:
            try:
                self.app.log.info(f"Parsing pipeline file: {pipeline}")
                pipeline = parse_pipeline(yaml.safe_load(stream))
            except schema.SchemaError as se:
                self.app.log.error(str(se))
                exit(1)

            self.app.extend('pipeline', pipeline)
            workflow_handler = self.app.handler.get('handlers', 'workflow', setup=True)
            workflow_handler.load(dataset)
            workflow_handler(dataset)

    def _parse_dataset(self) -> Path:
        if self.app.pargs.dataset:
            dataset = Path(self.app.pargs.dataset)
        else:
            dataset_files = [f for f in Path(self.app.pargs.workdir).iterdir() if f.suffix in ['.csv', '.tsv']]

            if len(dataset_files) == 0:
                raise TenetError('No dataset file found in the specified working directory')
            elif len(dataset_files) == 1:
                self.app.log.info(f"Using {dataset_files[0]} as dataset...")
                dataset = Path(dataset_files[0])
            else:
                option = [
                    inquirer.List('dataset', message="Select the dataset you want to use:", choices=dataset_files,
                                  ),
                ]
                answer = inquirer.prompt(option)
                dataset = Path(answer["dataset"])

        if not dataset.exists():
            raise TenetError(f"Dataset {dataset} not found.")

        return dataset

    def _parse_pipeline(self) -> Path:
        if self.app.pargs.file:
            pipeline = Path(self.app.pargs.file)
        else:
            pipeline_files = [f for f in Path(self.app.pargs.workdir).iterdir() if f.suffix in ['.yml', '.yaml']]
            if len(pipeline_files) == 0:
                raise TenetError('No pipeline file found in the specified working directory')
            elif len(pipeline_files) == 1:
                self.app.log.info(f"Using {pipeline_files[0]} as pipeline...")
                pipeline = Path(pipeline_files[0])
            else:
                option = [inquirer.List('pipeline_files', message="Select the pipeline you want to run:",
                                        choices=pipeline_files)]
                answer = inquirer.prompt(option)
                pipeline = Path(answer["pipeline_files"])

        if not pipeline.exists():
            raise TenetError(f"File {pipeline} not found.")

        return pipeline

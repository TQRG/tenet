import networkx as nx

import inquirer

from pathlib import Path

from cement import Controller, ex
from tenet import __version__
from ..core.exc import TenetError


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
            (['-gt', '--tokens'], {'help': 'Comma-separated list of tokens for the GitHub API.', 'type': str,
                                   'required': True}),
            (['-b', '--bind'], {'help': 'Docker directory path to bind (to workdir as a volume).',
                                'type': str, 'required': False})
        ]
    )
    def run(self):
        self.app.extend('workdir', Path(self.app.pargs.workdir))

        if not self.app.workdir.exists():
            raise TenetError(f"Working directory {self.app.workdir} not found")

        dataset_path = self._parse_dataset_path()
        pipeline_path = self._parse_pipeline_path()
        self._parse_bind()
        self.app.threads = self.app.pargs.threads if self.app.pargs.threads else self.app.get_config('local_threads')
        pipeline_handler = self.app.handler.get('handlers', 'pipeline', setup=True)
        pipeline_handler.load(pipeline_path)

        workflow_handler = self.app.handler.get('handlers', 'workflow', setup=True)
        workflow_name = self._select_workflow(pipeline_handler.workflows)
        workflow_graph = pipeline_handler.workflows[workflow_name].get_graph()

        if not nx.is_directed_acyclic_graph(workflow_graph):
            raise TenetError(f"Workflow must be directed acyclic graph")

        paths = pipeline_handler.workflows[workflow_name].get_all_paths(workflow_graph)
        path = self._select_path(paths)
        workflow_handler(dataset_path, path)

    @staticmethod
    def _select_workflow(workflows: dict):
        option = [inquirer.List('workflow', message="Select the workflow you want to run:",
                                choices=[f"{n}: {w}" for n, w in workflows.items()])]
        answer = inquirer.prompt(option)
        return answer["workflow"].split(':')[0]

    @staticmethod
    def _select_path(paths: list) -> list:
        if len(paths) > 1:
            option = [inquirer.List('path', message="Select the path you want to run:",
                                    choices=[f"{i}: {p}" for i, p in enumerate(paths)])]
            answer = inquirer.prompt(option)
            return paths[int(answer["path"].split(':')[0])]

        return paths[0]

    def _parse_bind(self):
        if self.app.pargs.bind:
            bind = self.app.pargs.bind
        else:
            bind = str(self.app.workdir.absolute()).replace(str(Path.home()), '')

        self.app.extend('bind', bind)

    def _parse_dataset_path(self) -> Path:
        if self.app.pargs.dataset:
            dataset_path = Path(self.app.pargs.dataset)
        else:
            dataset_files = [f for f in Path(self.app.pargs.workdir).iterdir() if f.suffix in ['.csv', '.tsv']]

            if len(dataset_files) == 0:
                raise TenetError('No dataset file found in the specified working directory')
            elif len(dataset_files) == 1:
                self.app.log.info(f"Using {dataset_files[0]} as dataset...")
                dataset_path = Path(dataset_files[0])
            else:
                option = [
                    inquirer.List('dataset', message="Select the dataset you want to use:", choices=dataset_files,
                                  ),
                ]
                answer = inquirer.prompt(option)
                dataset_path = Path(answer["dataset"])

        if not dataset_path.exists():
            raise TenetError(f"Dataset {dataset_path} not found.")

        return dataset_path

    def _parse_pipeline_path(self) -> Path:
        if self.app.pargs.file:
            pipeline_path = Path(self.app.pargs.file)
        else:
            pipeline_files = [f for f in Path(self.app.pargs.workdir).iterdir() if f.suffix in ['.yml', '.yaml']]
            if len(pipeline_files) == 0:
                raise TenetError('No pipeline file found in the specified working directory')
            elif len(pipeline_files) == 1:
                self.app.log.info(f"Using {pipeline_files[0]} as pipeline...")
                pipeline_path = Path(pipeline_files[0])
            else:
                option = [inquirer.List('pipeline_files', message="Select the pipeline you want to run:",
                                        choices=pipeline_files)]
                answer = inquirer.prompt(option)
                pipeline_path = Path(answer["pipeline_files"])

        if not pipeline_path.exists():
            raise TenetError(f"File {pipeline_path} not found.")

        return pipeline_path

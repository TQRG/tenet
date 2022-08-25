import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from docker.errors import APIError, NotFound
from docker.models.containers import Container

from securityaware.data.output import CommandData
from securityaware.core.exc import SecurityAwareError, CommandError
from securityaware.data.schema import ContainerCommand, Placeholder
from securityaware.handlers.node import NodeHandler
from securityaware.utils.misc import str_to_tarfile


class ContainerHandler(NodeHandler):
    class Meta:
        label = 'container'

    def __getitem__(self, name: str):
        try:
            return self.app.docker.containers.get(name)
        except NotFound as nf:
            self.app.log.error(str(nf))

            return None

    def find_output(self):
        """
            Returns the output path.
        """
        match = re.findall("\{(p\d+)\}", str(self.node.output))

        if match:
            output = str(self.node.output).format(**{m: self.edge.placeholders[m].value for m in match})
            self.app.log.info(f"Parsed output entry: {output}")

            if output.startswith(self.app.bind):
                output = output.replace(self.app.bind, str(self.app.workdir))

            self.output = Path(output)
            # Update as well the connectors
            if self.app.connectors.has_source(self.edge.name, attr='output'):
                self.set('output', self.output, skip=False)

        return self.output.exists()

    def parse(self):
        """
            Inserts the placeholders in the commands
        """
        # parse placeholders from the sinks

        if self.edge.name in self.app.connectors.sinks:
            for _, sink in self.app.connectors.sinks[self.edge.name].items():
                for tag, value in sink.map_placeholders().items():
                    if value is None:
                        self.app.log.error(f"Placeholder {tag} is None.")
                        exit(1)

                    self.edge.placeholders.update({tag: Placeholder(tag=tag, value=value, node=sink.sink)})

        # init placeholder sources
        if self.edge.sources:
            for p in self.edge.sources:
                if p in self.edge.placeholders:
                    value = self.edge.placeholders[p].value

                    if isinstance(value, str) and value.startswith(self.app.bind):
                        value = value.replace(self.app.bind, str(self.app.workdir))

                    self.set(p, value)

        for cmd in self.node.cmds:
            if self.edge.placeholders:
                # TODO: allow placeholders to have varied names
                for p in re.findall("\{(p\d+)\}", cmd.org):
                    if p not in self.edge.placeholders:
                        raise ValueError(f"Missing placeholder {p}")

                    value = str(self.edge.placeholders[p].value).replace(str(self.app.workdir), self.app.bind)
                    cmd.placeholders[p] = Placeholder(tag=p, value=value, node=self.edge.name)

                if not cmd.placeholders:
                    cmd.parsed = cmd.org
                else:
                    cmd.parsed = cmd.org.format(**cmd.get_placeholders())
            else:
                cmd.parsed = cmd.org

    def start(self, container_id: str):
        """
            Starts specified container if it is not running
        """
        container = self[container_id]

        # Start containers if these are not running
        if container and container.status != 'running':
            self.app.log.info(f"Starting {container.name} benchmark container.")
            container.start()

    def execute_cmd(self, cmd: ContainerCommand) -> bool:
        """
            Looks locally for files or directories in placeholders to decide whether the command will be re-executed.
        """
        for tag, placeholder in cmd.placeholders.items():
            path = Path(placeholder.value.replace(self.app.bind, str(self.app.workdir)))
            # if placeholder is a path that does not exist, run cmd
            if not path.exists():
                self.app.log.warning(f"path {path} for placeholder {tag} not found")
                return False

            # if placeholder is a directory that is empty, run cmd
            if path.is_dir() and len(list(path.iterdir())) == 0:
                self.app.log.warning(f"path {path} for placeholder {tag} is empty")
                return False

        self.app.log.warning(f"Paths for placeholders {list(cmd.placeholders.keys())} found.\n\tSkipping \"{cmd}\"")
        # TODO: find a better way for doing this
        cmd.skip = True
        return True

    def run_cmds(self, container_id: str, cmds: List[ContainerCommand]) -> Tuple[bool, List[CommandData]]:
        """
            Run commands inside the specified container.
        """
        cmds_data = []
        container_wd = str(self.path).replace(str(self.app.workdir), str(self.app.bind))

        for cmd in cmds:
            if self.output and self.output.exists() and cmd.placeholders and self.execute_cmd(cmd) and cmd.skip:
                continue

            cmd_data = self.__call__(container_id, cmd_str=str(cmd), cmd_cwd=container_wd, raise_err=False)
            cmds_data.append(cmd_data)

            if cmd_data.error or cmd_data.return_code != 0:
                self.app.log.error(cmd_data.error)
                return False, cmds_data

            if cmd_data.output:
                self.app.log.info(cmd_data.output)

        return True, cmds_data

    def run(self) -> Tuple[bool, List[CommandData]]:
        """
            Executes container node.

            :return: whether the command executions passed or failed as a bool.
        """

        container_name = self.app.workdir.name + '_' + self.edge.name
        container = self[container_name]

        if not container:
            self.app.log.warning(f"Container {container_name} not found.")

            # TODO: Fix this, folder set to the layer name (astminer container inside layer gets codeql folder)
            self.create(self.node.image, container_name)
            container = self[container_name]

            if not container:
                self.app.log.warning(f"Container {container_name} created but not found.")
                return False, []

        self.start(container_name)
        exec_status = self.run_cmds(container.id, self.node.cmds)
        self.app.log.info("Done")

        if container and container.status == "running":
            self.app.log.warning(f"Stopping running container {container.name}")
            container.stop()

        return exec_status

    def create(self, image: str, name: str) -> str:
        """
            Creates container from specified image

            :param image: name of the image
            :param name: name of the container

            :return: str with the id of the container
        """
        binds = {str(self.app.workdir): {'bind': self.app.bind, 'mode': 'rw'}}
        host_config = self.app.docker.api.create_host_config(binds=binds)
        output = self.app.docker.api.create_container(image, name=name, volumes=["/data"], host_config=host_config,
                                                      tty=True, detach=True)

        self.app.log.info(f"Created container for {name} with id {output['Id'][:10]}")

        if 'Warnings' in output and output['Warnings']:
            self.app.log.warning(' '.join(output['Warnings']))

        return output['Id']

    def __call__(self, container_id: str, cmd_str: str, args: str = "", call: bool = True, cmd_cwd: str = None,
                 msg: str = None, env: Path = None, timeout: int = None, raise_err: bool = False,
                 exit_err: bool = False, **kwargs) -> CommandData:

        cmd_data = CommandData(f"{cmd_str} {args}" if args else cmd_str)

        if msg and self.app.pargs.verbose:
            self.app.log.info(msg)

        self.app.log.debug(cmd_data.args, self.Meta.label)

        if not call:
            return cmd_data

        cmd = f"'source {env}; &&" if env else ""

        if cmd_cwd:
            cmd = f"{cmd} cd {cmd_cwd};"

        cmd = f"/bin/bash -c '{cmd}"

        if timeout:
            cmd = f"timeout --kill-after=1 --signal=SIGTERM {timeout} {cmd}"

        cmd = f"{cmd} {cmd_data.args}'"

        try:
            response = self.app.docker.api.exec_create(container_id, cmd, tty=False, stdout=True, stderr=True)
        except APIError:
            raise SecurityAwareError(f"failed to create exec object for command: {cmd}")

        exec_id = response['Id']
        self.app.log.debug(f"created exec object with Id {exec_id} for command: {cmd}", self.Meta.label)

        cmd_data.start = datetime.now()
        out = []
        err = []

        for stdout, stderr in self.app.docker.api.exec_start(exec_id, stream=True, demux=True):
            if stdout:
                line = stdout.decode('utf-8').rstrip('\n')

                if self.app.pargs.verbose:
                    print(line)

                out.append(line)

            if stderr:
                err_line = stderr.decode('utf-8').rstrip('\n')

                if self.app.pargs.verbose:
                    print(err_line)

                err.append(err_line)

        cmd_data.end = datetime.now()
        cmd_data.duration = (cmd_data.end - cmd_data.start).total_seconds()

        cmd_data.output = '\n'.join(out)
        cmd_data.error = '\n'.join(err)
        cmd_data.exit_status = self.app.docker.api.exec_inspect(exec_id)['ExitCode']

        if raise_err and cmd_data.error:
            raise CommandError(cmd_data.error)

        if exit_err and cmd_data.error:
            exit(cmd_data.exit_status)

        return cmd_data

    def write_script(self, container: Container, cmd_str: str, path: Path, name: str, mode: oct = 0o777) -> Path:
        bash_file = path / (name + ".sh")
        tar_bash_file = str_to_tarfile(f"#!/bin/bash\n{cmd_str}\n", tar_info_name=bash_file.name)

        with tar_bash_file.open(mode='rb') as fd:
            if not container.put_archive(data=fd, path=str(path)):
                raise SecurityAwareError(f'Writing bash file {bash_file.name} to {path} failed.')

        self(container_id=container.id, cmd_str=f"chmod {mode} {bash_file}", raise_err=True)

        return bash_file

    def mkdir(self, container_id: str, path: Path, **kwargs) -> CommandData:
        """
            mkdir wrapper that creates directories on the shared volume.
        """
        return self(container_id=container_id, cmd_str=f"mkdir -p {path}", raise_err=True, **kwargs)

    def write(self, container: Container, data: str, path: Path, name: str, mode: oct = 0o777) -> Path:
        tar_bash_file = str_to_tarfile(data, tar_info_name=name)
        file_path = path / name

        with tar_bash_file.open(mode='rb') as fd:
            if not container.put_archive(data=fd, path=str(path)):
                raise SecurityAwareError(f'Writing file {name} to {path} failed.')

        self(container_id=container.id, cmd_str=f"chmod {mode} {file_path}", raise_err=True)

        return file_path

    def iterdir(self, container_id: str, path: Path) -> List[Path]:
        cmd_data = self(container_id=container_id, cmd_str=f"ls {path}", raise_err=True)

        return [Path(path, f) for f in cmd_data.output.split('\n')]

from datetime import datetime
from pathlib import Path
from typing import List

from cement import Handler
from docker.errors import APIError, NotFound
from docker.models.containers import Container
from docker.models.volumes import Volume

from securityaware.data.output import CommandData
from securityaware.core.exc import SecurityAwareError, CommandError
from securityaware.core.interfaces import HandlersInterface
from securityaware.utils.misc import str_to_tarfile


class ContainerHandler(HandlersInterface, Handler):
    class Meta:
        label = 'container'

    def get(self, name: str):
        try:
            return self.app.docker.containers.get(name)
        except NotFound as nf:
            self.app.log.error(str(nf))

            return None

    def start(self, container_id: str):
        container = self.get(container_id)

        # Start containers if these are not running
        if container and container.status != 'running':
            self.app.log.info(f"Starting {container.name} benchmark container.")
            container.start()

    def run_cmds(self, container_id: str, cmds: List[str]) -> bool:
        for cmd in cmds:
            cmd_data = self.__call__(container_id, cmd_str=cmd, raise_err=False)

            if cmd_data.error or cmd_data.return_code != 0:
                self.app.log.error(cmd_data.error)
                return False

            self.app.log.info(cmd_data.output)

        return True

    def create(self, image: str, name: str, bind: str) -> str:
        binds = {bind: {'bind': "/data", 'mode': 'rw'}}
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

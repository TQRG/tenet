import subprocess
import psutil as psutil

from os import environ
from threading import Timer
from datetime import datetime

from cement import Handler

from tenet.core.exc import CommandError
from tenet.data.output import CommandData
from tenet.core.interfaces import HandlersInterface


class CommandHandler(HandlersInterface, Handler):
    class Meta:
        label = 'command'

    def __init__(self, **kw):
        super(CommandHandler, self).__init__(**kw)
        self.log = True

    def _exec(self, proc: subprocess.Popen, cmd_data: CommandData):
        out = []
        cmd = cmd_data.args.split()[0]
        for line in proc.stdout:
            decoded = line.decode()
            out.append(decoded)

            if self.app.pargs.verbose and self.log:
                self.app.log.info(decoded, cmd)

        cmd_data.output = ''.join(out)

        proc.wait(timeout=1)

        if proc.returncode and proc.returncode != 0:
            cmd_data.return_code = proc.returncode
            proc.kill()
            cmd_data.error = proc.stderr.read().decode()

            if cmd_data.error:
                self.app.log.error(cmd_data.error)

    def __call__(self, cmd_data: CommandData, cmd_cwd: str = None, msg: str = None, timeout: int = None,
                 raise_err: bool = False, exit_err: bool = False, **kwargs) -> CommandData:

        if msg and self.app.pargs.verbose:
            self.app.log.info(msg)

        self.app.log.debug(cmd_data.args, cmd_cwd)

        with subprocess.Popen(args=cmd_data.args, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=environ.copy(), cwd=cmd_cwd) as proc:

            cmd_data.start = datetime.now()

            if timeout:
                timer = Timer(timeout, _timer_out, args=[proc, cmd_data])
                timer.start()
                self._exec(proc, cmd_data)
                proc.stdout.close()
                timer.cancel()
            else:
                self._exec(proc, cmd_data)

            cmd_data.end = datetime.now()
            cmd_data.duration = (cmd_data.end - cmd_data.start).total_seconds()

            if raise_err and cmd_data.error:
                raise CommandError(cmd_data.error)

            if exit_err and cmd_data.error:
                exit(proc.returncode)

            return cmd_data


# https://stackoverflow.com/a/54775443
def _timer_out(p: subprocess.Popen, cmd_data: CommandData):
    cmd_data.error = "Command timed out"
    cmd_data.timeout = True
    process = psutil.Process(p.pid)
    cmd_data.return_code = p.returncode if p.returncode else 3

    for proc in process.children(recursive=True):
        proc.kill()

    process.kill()

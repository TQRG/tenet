from queue import Queue
from typing import List, Callable
from threading import Thread
from cement.core.log import LogHandler
from tqdm import tqdm

from securityaware.data.runner import Runner, Task


class TaskWorker(Thread):
    def __init__(self, queue: Queue, logger: LogHandler, func: Callable):
        Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.logger = logger
        self.func = func
        self.start()

    def run(self):
        while True:
            (task, callback) = self.queue.get()
            task.start()

            try:
                self.logger.info(f"Running task {task['id']}")
                # self.logger.info((f"Running {self.context.tool.name} on {self.context.benchmark.name}'s "
                #                  f"{task.program.vuln.id}."))
                task['result'] = self.func(task)
            except Exception as e:
                task.error(str(e))
                raise e.with_traceback(e.__traceback__)
            finally:
                if callback is not None:
                    callback(task)
                self.queue.task_done()
                self.logger.info(f"Task {task['id']} duration: {task.duration()}")


class ThreadPoolWorker(Thread):
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, runner_data: Runner, tasks: List[Task], threads: int, func: Callable, logger: LogHandler):
        Thread.__init__(self)
        self.runner_data = runner_data
        self.tasks = tasks
        self.daemon = True
        self.logger = logger
        self.func = func
        self.queue = Queue(threads)
        self.workers = []

        for _ in range(threads):
            self.workers.append(TaskWorker(self.queue, logger, func))

    def run(self):
        for task in tqdm(self.tasks):
            self.runner_data.running += [task]
            task.wait()
            # self.logger.info(f"Adding task for {self.nexus_handler.Meta.label} handler to the queue.")
            self.add_task(task)

        """Wait for completion of all the tasks in the queue"""
        self.queue.join()

    def add_task(self, task: Task):
        """Add a task to the queue"""
        if task.status is not None:
            self.queue.put((task, self.runner_data.done))

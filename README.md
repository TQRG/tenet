# Fine-grained approach to detect and patch vulnerabilities

## Requirements

- Docker (https://docs.docker.com/get-docker/)  

## Installation

```
$ git clone https://github.com/cmusv/SecurityAware
$ cd SecurityAware
$ pip install -r requirements.txt
$ pip install .
$ ./setup.sh
```

## Usage

```shell
usage: securityaware [-h] [-d] [-q] [-v] [-vb] {run} ...

Fine-grained approach to detect and patch vulnerabilities

optional arguments:
  -h, --help           show this help message and exit
  -d, --debug          full application debug mode
  -q, --quiet          suppress all console output
  -v, --version        show program's version number and exit
  -vb, --verbose       Verbose output.

sub-commands:
  {plugin,run}
    plugin        plugin controller
    run           Runs a workflow
```

### Plugin command

```shell
usage: securityaware plugin [-h] {install,uninstall} ...

optional arguments:
  -h, --help           show this help message and exit

sub-commands:
  {enabled, install,uninstall}
    enabled            Lists enabled plugins
    install            Installs plugin
    uninstall          Uninstalls plugin
```

#### Install sub-command
This command copies the target plugin file to the `plugin_dir` location specified in the `securityaware.yml` config file
(default is `~/.securityaware/plugins`) and enables the plugin in the config file.

> **_Note:_**
> - Some names are reserved for core handlers — e.g. _workflow_, _container_, _command_, etc.
> - The **name** of the plugin must match the **label** attributed to the plugin.
 
```shell
usage: securityaware plugin install [-h] -p PATH [-f] -n NAME

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  File path of the plugin.
  -f, --force           Overwrites existing plugins.
  -n NAME, --name NAME  Name of the plugin (should match its label).
```

##### Example
```shell
securityaware plugin install -p ~/workdir/plugins/code2vec.py -n code2vec -f
```

#### Uninstall sub-command

```shell
usage: securityaware plugin uninstall [-h] -n NAME

optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  Name of the plugin.
```

##### Example
```shell
securityaware plugin uninstall -n code2vec
```

### Run command
```shell
usage: securityaware run [-h] -f FILE -d DATASET -wd WORKDIR -b BIND

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Path to the pipeline config file
  -d DATASET, --dataset DATASET
                        Path to the input csv dataset file
  -wd WORKDIR, --workdir WORKDIR
                        Path to the workdir.
  -b BIND, --bind BIND  Docker directory path to bind (to workdir as a volume).
```

#### Example:

```shell
securityaware -vb run -f ~/projects/code2vec/code2vec_pipe.yml -d ~/projects/code2vec/cwe79.tsv -wd ~/projects/code2vec -b /projects/code2vec
```

## Schema 
The schema consists of three components: `nodes`, `layers`, and `workflow`.
A node can be a plugin or a container, and these that are defined in the `nodes`, instantiated in the `layers` and 
executed in the `workflow`.


### Plugin Node
This type of node is a Python plugin that extends the framework with a specific functionality.

#### Attributes: 

- name: str
  - name of the label of the plugin;
- kwargs: dict
  - a variable number keyword arguments (keys and values) that are passed to the plugin;

#### Example
```yaml
  - plugin:
      name: preprocess
      kwargs:
          max_contexts: 800
  - plugin:
      name: code2vec
      kwargs:
        train: true
        max_contexts: 800
```

### Container Node
This type of node executes commands in a specific container. The container shares a volume with its
respective node directory inside the working directory. Given the local working directory `/home/user/projects/code2vec`,
the node directory for a container with the name `jscodeshift` would be `/home/user/projects/code2vec/jscodeshift`. 
The binding with the working directory in the container is specified with the `-b` flag. For instance, 
the working directory in the container `/projects/code2vec` corresponds 
to the local working directory `/home/user/projects/code2vec`.

#### Attributes:

- name: str
  - name of the container 
- image: str
  - name of the container's image (must exist)
- cmds: List[str]
  - list of commands to be executed in the container
  - these are converted to ContainerCommands
  - can contain placeholders
- output: str
  - specifies the file name of the output, relative to the working directory of the container
  - can contain placeholders

#### Example:
The following node instantiates a container with the name `astminer` from the image `astminer:latest`, 
and executes the respective commands of astminer with different arguments provided from other nodes. 
```yaml
  - container:
      image: astminer:latest
      name: astminer
      cmds: 
        - "mkdir -p {p2}"
        - "export NODE_OPTIONS=\"--max-old-space-size=8192\""
        - "java -jar -Xms4g -Xmx4g build/shadow/astminer.jar code2vec {p1} {p2} {p3} 0"
      output: "{p2}/path_contexts.c2s"
```

### Layers
Consists of a sequential list of `nodes` to be executed in the specified order (from top to bottom).

#### Example:
```yaml
layers:
  dataset:
    - find_refs:
        node: find_references
        ...
    - raw:
        node: github_collector
        ...
  diff:
    - labels:
        node: labeler
        ...
    - jscodeshift:
        node: jscodeshift
        ...
    .
    .
    .
```

For instance: `find_refs` -> `raw` -> `labels` -> `jscodeshift` -> ...

### Workflow
Consists of a sequential list of `layers` to be executed in the specified order (from left to right).

#### Example:
```yaml
workflow: [dataset, diff, prepare, evaluate]
```

For instance `dataset` -> `diff` -> `prepare` -> `evaluate`


## Development

This project includes a number of helpers in the `Makefile` to streamline common development tasks.

### Environment Setup

The following demonstrates setting up and working with a development environment:

```
### create a virtualenv for development

$ make virtualenv

$ source env/bin/activate


### run securityaware cli application

$ securityaware --help


### run pytest / coverage

$ make test
```

## Deployments

### Docker

Included is a basic `Dockerfile` for building and distributing `SecurityAware`,
and can be built with the included `make` helper:

```
$ make docker

$ docker run -it securityaware --help
```
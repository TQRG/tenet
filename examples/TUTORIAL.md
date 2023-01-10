# General Steps
1) Create working directory, for instance, `~/work_dir`;
2) Enable all the plugins in the target `.yml` file containing the pipeline (in the config file `~/.tenet/config/tenet.yml`);
   1) Check for enabled plugins ```tenet pluign enabled```; 
   2) Install/Enable missing plugins ```tenet plugin install -p tenet/plugins/^pluign_name^ -n `plugin name` -f```
3) Make sure all the necessary Docker images in the pipeline `.yml` file are built;
4) Run the pipeline with the ```tenet -vb run``` command.

# Generating dataset from NVD Sources

### Configuration
1) Create working directory, for instance, `~/sources`;
2) Copy to the working directory the pipeline file `nvd.yml` under `examples` folder and a dummy dataset file (e.g. `dummy.tsv`);
3) (Optional) Enable plugins.
4) Run the pipeline ```tenet -vb run -f ~/sources/nvd.yml -d ~/sources/dummy.tsv -wd ~/sources -b /sources```

# Training code2vec on CWE-79 vulnerabilities in javascript
> **Note**: you must add your GitHub API token in the configuration files to run the collector plugin. 
### Configuration

1) Create working directory, for instance, `~/js_cwe_79`; 
2) (Optional) Copy to the working directory the pipeline file `code2vec.yml` under `examples` folder and the dataset file `js_cwe_79.tsv` under `datasets`;
3) (Optional) Enable plugins.
4) Necessary Docker images:
   1) jscodeshift - ```docker pull epicosy/securityaware:jscodeshift```
   2) astminer - ```docker pull epicosy/securityaware:astminer```
   3) code2vec - ```docker pull epicosy/securityaware:code2vec```
5) Run the pipeline ```tenet -vb run -f ~/js_cwe_79/code2vec.yml -d ~/js_cwe_79/js_cwe_79.tsv -wd ~/js_cwe_79 -b /js_cwe_79```

> **Note**: you must run twice the command in step 5 as after training code2vec script writes to sterr and the 
> following nodes are not executed.

# Training CodeBERT on CWE-79 vulnerabilities in javascript

### Configuration

1) Create working directory, for instance, `~/js_cwe_79`; 
2) (Optional) Copy to the working directory the pipeline file `codebert.yml` under `examples` folder and the dataset file `js_cwe_79.tsv` under `datasets`;
3) (Optional) Enable plugins.
4) Necessary Docker images:
   1) jscodeshift - ```docker pull epicosy/securityaware:jscodeshift```
   2) astminer - ```docker pull epicosy/securityaware:astminer```
   3) codebert - ```docker pull epicosy/securityaware:codebert```
5) Run the pipeline ```tenet -vb run -f ~/js_cwe_79/codebert.yml -d ~/js_cwe_79/js_cwe_79.tsv -wd ~/js_cwe_79 -b /js_cwe_79```


# Training Traditional ML models on CWE-79 vulnerabilities in javascript
### Configuration

1) Create working directory, for instance, `~/js_cwe_79`; 
2) (Optional) Copy to the working directory the pipeline file `basic.yml` under `examples` folder and the dataset file `js_cwe_79.tsv` under `datasets`;
3) Enable all the plugins in `basic.yml` (in the config file `~/.tenet/config/tenet.yml`);
   1) Check for enabled plugins ```tenet pluign enabled```; 
   2) Install/Enable missing plugins ```tenet plugin install -p securityaware/plugins/^pluign_name^ -n `plugin name` -f```
4) Necessary Docker images:
   1) jscodeshift - ```docker pull epicosy/securityaware:jscodeshift```
   2) astminer - ```docker pull epicosy/securityaware:astminer```
5) Run the pipeline ```securityaware -vb run -f ~/js_cwe_79/basic.yml -d ~/js_cwe_79/js_cwe_79.tsv -wd ~/js_cwe_79 -b /js_cwe_79```

# Training Traditional ML models on CWE-79 vulnerabilities in javascript with CodeQL labelling

### Configuration
1) Create working directory, for instance, `~/js_cwe_79`; 
2) (Optional) Copy to the working directory the pipeline file `codeql_basic.yml` under `examples` folder and the dataset file `js_cwe_79.tsv` under `datasets`;
3) (Optional) Enable plugins.
4) Necessary Docker images:
   1) jscodeshift - ```docker pull epicosy/securityaware:jscodeshift```
   2) astminer - ```docker pull epicosy/securityaware:astminer```
   3) codeql - ```docker pull epicosy/securityaware:codeql```
5) Run the pipeline ```securityaware -vb run -f ~/js_cwe_79/basic.yml -d ~/js_cwe_79/js_cwe_79.tsv -wd ~/js_cwe_79 -b /js_cwe_79```


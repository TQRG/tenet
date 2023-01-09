# General Steps
1) Create working directory, for instance, `~/work_dir`;
2) Enable all the plugins in the target `.yml` file containing the pipeline (in the config file `~/.securityaware/config/securityaware.yml`);
   1) Check for enabled plugins ```securityaware pluign enabled```; 
   2) Install/Enable missing plugins ```securityaware plugin install -p securityaware/plugins/^pluign_name^ -n `plugin name` -f```
3) Make sure all the necessary Docker images in the pipeline `.yml` file are built;
4) Run the pipeline with the ```securityaware -vb run``` command.


# Generating dataset from NVD Sources

### Configuration
1) Create working directory, for instance, `~/sources`;
2) Copy to the working directory the pipeline file `nvd.yml` under `examples` folder and a dummy dataset file (e.g. `dummy.tsv`);
3) (Optional) Enable plugins.
4) Run the pipeline ```securityaware -vb run -f ~/sources/nvd.yml -d ~/sources/dummy.tsv -wd ~/sources -b /sources```


# Generating dataset from OSV Sources

### Configuration
1) Create working directory, for instance, `~/sources`;
2) Copy to the working directory the pipeline file `osv.yml` under `examples` folder and a dummy dataset file (e.g. `dummy.tsv`);
3) (Optional) Enable plugins.
4) Run the pipeline ```securityaware -vb run -f ~/sources/osv.yml -d ~/sources/dummy.tsv -wd ~/sources -b /sources```


# Merge Datasets

### Configuration
1) Create working directory, for instance, `~/sources`;
2) Copy to the working directory the pipeline file `merge_sources.yml` under `examples` folder and a dummy dataset file (e.g. `dummy.tsv`);
3) (Optional) Enable plugins.
4) Run the pipeline ```securityaware -vb run -f ~/sources/merge_sources.yml -d ~/sources/dummy.tsv -wd ~/sources -b /sources```
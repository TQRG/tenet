# Steps
## Preparation
1) Create working directory, for instance, `~/cain23`;
2) Copy to the working directory the dataset file `dummy.tsv` under `datasets`;
3) (Optional) Make sure all the plugins in the target `pipeline.yml` file are located correctly (`~/.tenet/plugins/`);
   - The ```./setup.sh``` script copies all available plugins in the ```tenet/plugins``` repo project to the correct location;
4) (Optional) Make sure all the necessary Docker images in the `pipeline.yml` file are built;

## Execution
6) Run the pipeline with ```tenet -vb run -wd ~/cain23 -gt [GITHUB TOKENS] -t [NUMBER OF THREADS] -x js``` command.
7) Select the workflow you want to run:

```
>  eval_basic_diff: evaluates basic models on diff analysis labels
>  eval_basic_static: evaluates basic models on static analysis labels
>  eval_c2v_diff: evaluates code2vec on diff analysis labels
>  eval_c2v_static: evaluates code2vec on static analysis labels
>  eval_codebert_diff: evaluates CodeBERT on diff analysis labels
>  eval_codebert_static: evaluates CodeBERT on static analysis labels
```

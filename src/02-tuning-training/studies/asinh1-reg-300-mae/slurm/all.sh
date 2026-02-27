#!/bin/bash

for f in epf-with-ml-on-orderbooks/src/hpo/studies/asinh1-reg-300-mae/slurm/submit_*.sh; do sbatch "$f"; done
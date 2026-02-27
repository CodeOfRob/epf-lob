#!/bin/bash
#SBATCH --job-name=TabPFN_10k
#SBATCH --array=1-11            # 20 Worker parallel
#SBATCH --cpus-per-task=8        # 8 CPUs reichen locker für 10k
#SBATCH --mem=32G                # 32GB ist ein sehr sicherer Puffer (16GB ginge wohl auch)
#SBATCH --partition=clara,paula
#SBATCH --gpus=1
#SBATCH --time=01:00:00          # Sollte sehr schnell durchlaufen
#SBATCH --output=$HOME/jobfiles/log/hpo/tabpfn_10k/%x_%A_%a.out
#SBATCH --error=$HOME/jobfiles/log/hpo/tabpfn_10k/%x_%A_%a.error
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

echo "Job started: $(date) on $SLURMD_NODENAME"

# 1. Module laden
module purge
module load Anaconda3
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 # Passend zu deiner PyTorch Version

export HF_TOKEN=""

eval "$(conda shell.bash hook)"
conda activate Masterarbeit

# 2. Threading-Steuerung
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Debug Informationen
echo "Job $SLURM_ARRAY_TASK_ID started on $(hostname)"
echo "Job gestartet auf Node: $(hostname)"
echo "CPU Limit: $SLURM_CPUS_PER_TASK"
echo "Memory Limit: $SLURM_MEM_PER_NODE"
echo "Threading: 4 Prozesse à 16 Threads"
echo ""

# 3. In das Verzeichnis wechseln
cd $HOME/epf-with-ml-on-orderbooks/src/hpo/studies/asinh1-reg-300-mae/tabpfn || exit 1

echo "Starting tabpfn training with 50k Samples context..."

# 4. Ausführen
# -u sorgt für unbuffered output, damit du den tqdm Fortschritt sofort im Log siehst
python -u train_tabpfn_parallel.py

echo "Job finished: $(date)"
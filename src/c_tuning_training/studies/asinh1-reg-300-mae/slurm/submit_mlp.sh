#!/bin/bash
#SBATCH --job-name=HPO_mlp      
#SBATCH --cpus-per-task=32
#SBATCH --partition=paula,paul
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=$HOME/jobfiles/log/hpo/mlp/%x_%j.out   
#SBATCH --error=$HOME/jobfiles/log/hpo/mlp/%x_%j.error  
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80  

# --- KONFIGURATION DER PARAMETER ---
# Hier definieren Sie alle Argumente für Ihr Python-Skript zentral

MODEL_NAME="mlp"
CONFIG_FILE="configs/${MODEL_NAME}_config.py"

STUDY_DIR="/home/sc.uni-leipzig.de/to65jevo/epf-with-ml-on-orderbooks/src/hpo/studies/asinh1-reg-300-mae"

TRAIN_FILE="../../data/parquet/features/asinh1-reg/splits/train.parquet"
OPTUNA_DB="sqlite:///${STUDY_DIR}/asinh1-reg-300-mae.db"
OUT="${STUDY_DIR}/results/${MODEL_NAME}/"

THINNING_FREQ_SEC=300       # in Sekunden
SUBSAMPLE_N=250000    # Anzahl der Samples für HPO
TARGET_MODE="reg"
# -----------------------------------

echo "Job started: $(date) on $SLURMD_NODENAME"
echo "Config: $CONFIG_FILE | Thinning: $THINNING_FREQ_SEC | Subsample: $SUBSAMPLE_N | Target Mode: $TARGET_MODE"

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
conda activate Masterarbeit

# In das Verzeichnis wechseln
cd $HOME/epf-with-ml-on-orderbooks/src/hpo/

echo "Starting mlp HPO..."

# Aufruf mit den definierten Variablen
# Hinweis: Wir bauen den Befehl dynamisch zusammen
CMD="python -u run_hpo_v2-mae.py $CONFIG_FILE \
    --thinning_freq_sec $THINNING_FREQ_SEC \
    --subsample_n $SUBSAMPLE_N \
    --train_file $TRAIN_FILE \
    --optuna_db $OPTUNA_DB \
    --model_out $OUT \
    --target_type $TARGET_MODE"

# Befehl ausführen
echo "Executing: $CMD"
$CMD

echo "Job finished: $(date)"
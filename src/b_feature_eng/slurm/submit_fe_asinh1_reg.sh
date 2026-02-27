#!/bin/bash
#SBATCH --job-name=FE_REG                           # Ein spezifischerer Job-Name
#SBATCH --time=2:00:00                                      # Maximale Laufzeit (4 Stunden)
#SBATCH --partition=paula,paul                             # Partition/Queue
#SBATCH --mem=100G                                          # Arbeitsspeicher
#SBATCH --ntasks=1                                          # Eine Hauptaufgabe
#SBATCH --cpus-per-task=32                                   # CPUs für Datenladen etc.
#SBATCH -o $HOME/jobfiles/log/fe_reg/%x.out-%j               # Standard-Output-Datei %j für Job-ID, %x für Job-Name
#SBATCH -e $HOME/jobfiles/log/fe_reg/%x.error-%j             # Standard-Error-Datei
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80            # Benachrichtigung bei Ende/Fehler

# --- Umgebung einrichten (gemäß Cluster-FAQ) ---
echo "Job gestartet am $(date)"
echo "Job laeuft auf Knoten: $SLURMD_NODENAME"

# 1. Umgebung zurücksetzen
module purge
echo "Module nach 'purge' geladen:"
module list

# 2. Notwendige System-Module laden
module load Anaconda3
echo "Module nach 'load' geladen:"
module list

# 3. Conda-Shell initialisieren
eval "$(conda shell.bash hook)"

# 4. Ihre persönliche Conda-Umgebung aktivieren
#    (Die Sie mit der environment.yml erstellt haben)
conda activate Masterarbeit
echo "Conda-Umgebung 'Masterarbeit' aktiviert."
echo "Verwende Python-Version: $(which python)"

# --- Python-Skript ausführen ---
echo "Starte Python-Skript zur Feature Creation..."

# Der eigentliche Befehl zum Ausführen des Skripts
python -u /home/sc.uni-leipzig.de/to65jevo/epf-with-ml-on-orderbooks/src/features/featureseng_reg.py

echo "Python-Skript beendet mit Exit-Code $?"
echo "Job beendet am $(date)"

#!/bin/bash
#SBATCH --job-name=CaHTGP_5seeds
#SBATCH --output=CaHTGP_5seeds_%j.out
#SBATCH --error=CaHTGP_5seeds_%j.err
#SBATCH --time=120:00:00
#SBATCH --partition=quad_rtx_8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Job started at: $(date)"
export PATH=/home/cril/r.abidi/bin:$PATH
cd /home/cril/r.abidi/CaHTGP/CaHTGP

for SEED in 2024 42 123 456 789
do
    echo "==============================="
    echo "SEED: $SEED --- $(date)"
    echo "==============================="

    echo "--- RetailRocket seed $SEED ---"
    python3 main.py --dataset RetailRocket --seed $SEED

    echo "--- Tmall seed $SEED ---"
    python3 main.py --dataset Tmall --seed $SEED

    echo "--- lastfm seed $SEED ---"
    python3 main.py --dataset lastfm --seed $SEED

done

echo "Job finished at: $(date)"
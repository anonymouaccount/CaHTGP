#!/bin/bash
#SBATCH --job-name=Tmall_Ksweep
#SBATCH --output=Tmall_Ksweep_%j.out
#SBATCH --error=Tmall_Ksweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=quad_rtx_8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Job started at: $(date)"
export PATH=/home/cril/r.abidi/bin:$PATH
cd /home/cril/r.abidi/CaHTGP/CaHTGP

SEED=2024

for K in 1 2 3 4 5 6
do
    echo "==============================="
    echo "Tmall K=$K seed=$SEED --- $(date)"
    echo "==============================="
    python3 main.py \
        --dataset Tmall \
        --seed $SEED \
        --interests $K \
        --dropout_global 0.5 \
        --dropout_gcn 0.5
done

echo "Job finished at: $(date)"
#!/bin/bash
#SBATCH --account=zichen0719
#SBATCH --job-name=experiment_run_june10
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=tianlab-contrib
#SBATCH --gres=gpu:4
#SBATCH --ntasks=3
#SBATCH --time=4:00:00
#SBATCH --output=experiment_%j.out
#SBATCH --error=experiment_%j.err

cd /home/zichen0719/AI-Scientist-v2
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate ~/ai-scientist-env

python launch_scientist_bfts.py \
    --writeup-type normal \
    --skip_review \
    --load_ideas ai_scientist/ideas/game_theoretic_moe_enriched.json \
    --idea_idx 0
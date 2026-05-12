#!/bin/bash
#SBATCH --partition=rtx3090
#SBATCH --gpus=1  
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=molgen_flex_test
#SBATCH --output=flex_test_%j.log
#SBATCH --error=flex_test_%j.log
#SBATCH --mail-user=evronba@post.bgu.ac.il 
#SBATCH --mail-type=BEGIN,END,FAIL 

# 1. הגדרת שם התיקייה שקיימת כרגע אצלך ב-VS Code
BASE_DIR="/home/evronba/MolGenOrg_MultiGoal"
cd $BASE_DIR

# Optional overrides passed from:
# `sbatch run_molgen.sbatch <config_path> <wandb_proj> <wandb_name>`
CONFIG_PATH="${1:-$BASE_DIR/data/configs/dt_config_char_normalized_goal_conditioned.toml}"
WANDB_PROJ="${2:-MolGen-Flexible-Test}"
WANDB_NAME="${3:-}"

# 2. טעינת הסביבה
module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate molgen

# 3. הגדרות סביבה
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$BASE_DIR:$BASE_DIR/molgen:$BASE_DIR/molgen/utils

# 4. הרצה (השתמשתי ב-$BASE_DIR כדי שכל הנתיבים יהיו מדויקים)
PY_CMD=(
    /home/evronba/.conda/envs/molgen/bin/python -m scripts.pretrain
    --data-path "$BASE_DIR/data/datasets/zinc_smiles_toy.text"
    --tokenizer-path "$BASE_DIR/data/tokenizers/charTokenizer"
    --save-path "$BASE_DIR/data/results"
    --model-type "DT"
    --dataset-type "DT_SMILES"
    --config-path "$CONFIG_PATH"
    --wandb-entity "evronba-ben-gurion-university-of-the-negev"
    --wandb-proj "$WANDB_PROJ"
)

if [ -n "$WANDB_NAME" ]; then
    PY_CMD+=(--wandb-name "$WANDB_NAME")
fi

"${PY_CMD[@]}"

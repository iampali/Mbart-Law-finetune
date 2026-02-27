#!/bin/bash

#SBATCH -p compute            
#SBATCH -J NLLB               
#SBATCH --cpus-per-task=16    
#SBATCH --mem=30000           
#SBATCH -t 1-02:30:00         
#SBATCH --gres=gpu:rtxa6000:1 


module load python/3.12.12

python -m venv my_pipeline_env

source my_pipeline_env/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

python main.py --wandb_run_name="NLLB-First-train" --save_steps=17000 --logging_steps=500 --num_train_epochs=2 --learning_rate=2e-4 --batch_size=32 --eval_batch_size=16
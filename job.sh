#!/bin/bash

#SBATCH -p compute            
#SBATCH -J NLLB               
#SBATCH --cpus-per-task=4   
#SBATCH --mem=60000           
#SBATCH -t 1-02:30:00         
#SBATCH --gres=gpu:a100:1

source Practicum/bin/activate

uv run main.py --wandb_run_name="NLLB-First-train" --save_every_epochs=1 --save_total_limit=10 --gradient_accumulation_steps=8 --logging_steps=500 --num_train_epochs=2 --learning_rate=2e-4 --batch_size=4 --eval_batch_size=16
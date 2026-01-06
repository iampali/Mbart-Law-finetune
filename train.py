from initialize_model import init_model, init_tokenizer

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from environment_variables import model_save_path
from create_tokenized_dataset import get_tokenized_dataset
from eval import compute_metrics
import argparse
import torch
from setup_logging import logger
import wandb
from dotenv import load_dotenv

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="Pass some Arguemtns to train the model")

parser.add_argument("--dataset_length",type= int, default=0, help="Select how much data you want to train the model, by default it's 0 which means whole data.")
parser.add_argument("--eval_strategy", default="no", help="Would you want to eval your model while training.")
parser.add_argument("--eval_steps", type=int, default=100, help="Steps on which you want to eval your model")
parser.add_argument("--learning_rate",type=float, default=2e-2, help="Add a learning rate")
parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch Size for training")
parser.add_argument("--loggin_steps", type=int, default=100, help="Provide after how many steps you want to send logs")
parser.add_argument("--save_steps", type=int, default=100, help="Steps afer which you want to save your model")
parser.add_argument("--wandb_run_name", default="Mbart50", help="Provide a run name for wandb")

args = parser.parse_args()

logger.info(f"Setting up model and tokenizer")
model = init_model(get_lora_model=True)
tokenizer = init_tokenizer()

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id
)

# Training Arguments

training_args = Seq2SeqTrainingArguments(
    output_dir=model_save_path,
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    learning_rate=args.learning_rate,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, # by default
    weight_decay=0.01,
    save_total_limit=10,
    num_train_epochs=2,
    predict_with_generate =True,
    fp16=False,
    bf16 = True,  # Use mixed precision
    logging_steps=args.loggin_steps,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_steps=args.save_steps,
    report_to="wandb",
    run_name=args.wandb_run_name
)

logger.info("Loading tokenized data")
tokenized_datasets = get_tokenized_dataset(tokenizer, args.dataset_length)

## seting up wandb
load_dotenv()
wandb.login()

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    processing_class=tokenizer, # tokenizer
    data_collator=data_collator,
    compute_metrics= lambda eval_preds : compute_metrics(eval_preds, tokenizer)  # Uncomment if using metrics
)

logger.info("Started Training")
# Start training
trainer.train()


logger.info(f"Training has been successfully completed and all models has been saved to {model_save_path}")
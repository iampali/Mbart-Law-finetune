import argparse
from train import training_model
from eval import evaluation_model
import torch.multiprocessing as mp
from setup_logging import logger
from environment_variables import model_save_path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Pass some Arguemtns to train the model")

    parser.add_argument("--dataset_length",type= int, default=0, help="Select how much data you want to train the model, by default it's 0 which means whole data.")
    parser.add_argument("--wandb_run_name", default="Mbart50", help="Provide a run name for wandb")
    parser.add_argument("--save_every_epochs", type=int, default=1, help="Epochs afer which you want to save your model")
    parser.add_argument("--save_total_limit", type=int, default=10, help="How many checkpoints you want to save")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Mention the gradients accumulated steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Provide after how many steps you want to send logs")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Provide the number of epochs")
    parser.add_argument("--learning_rate",type=float, default=2e-5, help="Add a learning rate")
    parser.add_argument("--batch_size",type=int, default=32, help="Add the batch_size that will be trained in one go.")
    parser.add_argument("--train_strategy", type=int, default=1, help="Would you want to train your model? 0 for False and 1 for True")
    parser.add_argument("--eval_strategy", type=int, default=1, help="Would you want to start the evaluation of your models? 0 for False and 1 for True")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch Size for training")

    return parser.parse_args()


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Prevents crashing if the method is accidentally set twice

    args = parse_args()
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    wandb_run_name_final = f"{args.wandb_run_name}-{date_time}"
    output_folder = f"Model-{date_time}"
    if args.train_strategy:

        logger.info(f"Starting Training Here {args.train_strategy}")

        training = training_model(
            dataset_length=args.dataset_length,
            wandb_run_name=wandb_run_name_final + '-train',
            save_every_epochs=args.save_every_epochs,
            save_total_limit=args.save_total_limit,
            logging_steps=args.logging_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            output_folder=output_folder
        )

        training.start_training()

        logger.info(f"Training has been successfully completed and all models has been saved to {model_save_path}")


    if args.eval_strategy:

        logger.info("Starting Evaluation")

        evaluation = evaluation_model(
            eval_batch_size=args.eval_batch_size,
            wandb_eval_run_name=wandb_run_name_final + '-eval',
            output_folder=output_folder
        )

        evaluation.start_eval()

        logger.info(f"Done with complete evaluation of all models. Check results on weights and biases.")
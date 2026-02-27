import os
import shutil
import torch
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from setup_logging import logger
from initialize_model import init_model, init_tokenizer
from create_tokenized_dataset import get_tokenized_dataset
from torch.utils.data import DataLoader
from environment_variables import model_save_path
from dotenv import load_dotenv
from transformers import DataCollatorForSeq2Seq




class training_model:

    def __init__(self, dataset_length : int,wandb_run_name : str, save_steps : int, logging_steps : int, num_train_epochs : int, learning_rate : float, batch_size : int):

        self.dataset_length = dataset_length,
        self.wandb_run_name = wandb_run_name
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        weight_decay = 0.01

        load_dotenv(override=True)

        logger.info(f"Setting up the tokenizer")
        tokenizer = init_tokenizer()

        logger.info("Loading tokenized data")
        train_tokenized_dataset, _ = get_tokenized_dataset(tokenizer, dataset_length)


        logger.info(f"Setting up the Model")
        self.model = init_model(get_lora_model=True)


        # Instantiate the Data Collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            model=self.model, 
            padding=True # This tells it to dynamically pad to the longest sequence in the batch
        )

        logger.info("Loading Train DataLoader")
        self.train_dataloader = DataLoader(
            train_tokenized_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=data_collator
        )

        # Assume the following are defined externally:
        # - model: Your NLLB 1.3B model with frozen parameters and trainable MoE layers for PEFT
        # - train_dataloader: DataLoader for training with batch_size=per_device_train_batch_size

        ## Setup Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"The device using to train the model is {self.device}")

        # Move model to CUDA (assume it's already in bfloat16 if needed)
        self.model.to(self.device)

        # Optimizer only on trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Total training steps for cosine scheduler
        total_steps = num_train_epochs * len(self.train_dataloader)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=0  # Decay to 0 as in typical cosine schedule
        )

        # Initialize wandb
        wandb.init(name=wandb_run_name)


    def start_training(self):

        # Training loop
        global_step = 0
        accum_loss = 0.0

        for epoch in tqdm(range(self.num_train_epochs), desc="Training Progress"):
            self.model.train()
            
            for batch in self.train_dataloader:

                for key in batch:
                    batch[key] = batch[key].to(self.device)

                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                accum_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % self.logging_steps == 0 and global_step > 0:
                    avg_loss = accum_loss / self.logging_steps
                    wandb.log({
                        "train_loss": avg_loss,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "global_step": global_step
                    })
                    accum_loss = 0.0
            
            # Save checkpoint every save_every_epochs epochs
            if (epoch + 1) % self.save_every_epochs == 0:
                checkpoint_path = os.path.join(model_save_path, f"checkpoint-{epoch+1}")
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
                
                # Manage save_total_limit: keep only the last save_total_limit checkpoints
                checkpoints = [d for d in os.listdir(model_save_path) if d.startswith("checkpoint-")]
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                while len(checkpoints) > self.save_total_limit:
                    oldest_checkpoint = checkpoints.pop(0)
                    shutil.rmtree(os.path.join(model_save_path, oldest_checkpoint))

        # Finish wandb run
        wandb.finish()
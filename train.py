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
from environment_variables import model_save_path, wandb_project_name
from dotenv import load_dotenv
from transformers import DataCollatorForSeq2Seq

class training_model:

    def __init__(self, dataset_length: int, wandb_run_name: str, save_every_epochs: int, save_total_limit: int, logging_steps: int, num_train_epochs: int, learning_rate: float, batch_size: int, gradient_accumulation_steps: int, output_folder : str):

        self.dataset_length = dataset_length # FIXED: Removed trailing comma that created a tuple
        self.wandb_run_name = wandb_run_name
        self.save_every_epochs = save_every_epochs # FIXED: Added to __init__
        self.save_total_limit = save_total_limit   # FIXED: Added to __init__
        self.logging_steps = logging_steps
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size # This is now your MICRO batch size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_folder = output_folder
        weight_decay = 0.01

        load_dotenv(override=True)

        logger.info(f"Setting up the tokenizer")
        tokenizer = init_tokenizer()

        logger.info("Loading tokenized data")
        train_tokenized_dataset, _ = get_tokenized_dataset(tokenizer, self.dataset_length)

        logger.info(f"Setting up the Model")
        self.model = init_model(get_lora_model=True)
        
        # # ---------------------------------------------------------
        # # OOM FIX 1: Enable Gradient Checkpointing
        # # ---------------------------------------------------------
        # self.model.gradient_checkpointing_enable()    
        # # THE FIX: Force the graph to stay alive for PEFT adapters
        # self.model.enable_input_require_grads() 
        
        logger.info("Gradient checkpointing enabled to save VRAM.")
        logger.info("Gradient checkpointing enabled to save VRAM.")

        # Instantiate the Data Collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            model=self.model, 
            padding=True 
        )

        logger.info("Loading Train DataLoader")
        self.train_dataloader = DataLoader(
            train_tokenized_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=data_collator
        )

        ## Setup Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"The device using to train the model is {self.device}")

        self.model.to(self.device)

        # Optimizer only on trainable parameters (your MoE adapters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Total training steps adjusted for gradient accumulation
        total_steps = (self.num_train_epochs * len(self.train_dataloader)) // self.gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=0  
        )

        # Initialize wandb
        wandb.init(project=wandb_project_name, name=self.wandb_run_name)

    def start_training(self):

        # Training loop
        global_step = 0
        accum_loss = 0.0

        for epoch in tqdm(range(self.num_train_epochs), desc="Training Progress"):
            self.model.train()
            
            for batch_idx, batch in enumerate(self.train_dataloader):

                for key in batch:
                    batch[key] = batch[key].to(self.device)
                    # 2. ONLY cast floating-point tensors (like custom masks or weights).
                    if batch[key].is_floating_point():
                        batch[key] = batch[key].to(torch.bfloat16)

                # ---------------------------------------------------------
                # OOM FIX 2: PyTorch Autocast for bfloat16 Mixed Precision
                # ---------------------------------------------------------
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = self.model(**batch)
                    # OOM FIX 3: Scale the loss for gradient accumulation
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                loss.backward()
                accum_loss += loss.item() * self.gradient_accumulation_steps # Track actual loss
                
                # ---------------------------------------------------------
                # OOM FIX 3: Step the optimizer only after accumulating gradients
                # ---------------------------------------------------------
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
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
                final_path = model_save_path + self.output_folder
                checkpoint_path = os.path.join(final_path, f"checkpoint-{epoch+1}")
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
                
                # Manage save_total_limit
                checkpoints = [d for d in os.listdir(final_path) if d.startswith("checkpoint-")]
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                while len(checkpoints) > self.save_total_limit:
                    oldest_checkpoint = checkpoints.pop(0)
                    shutil.rmtree(os.path.join(final_path, oldest_checkpoint))

        # Finish wandb run
        wandb.finish()
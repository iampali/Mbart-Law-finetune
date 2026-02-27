import os
import json
import torch
import wandb
import numpy as np
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Your custom imports
from sacrebleu.metrics import BLEU
import evaluate
from environment_variables import model_save_path, temp_file_path
from setup_logging import logger
from initialize_model import init_model, init_tokenizer
from torch.utils.data import DataLoader
from create_tokenized_dataset import get_tokenized_dataset
from transformers import DataCollatorForSeq2Seq


def run_inference_process(checkpoint_path, eval_batch_size, tokenizer, dataset):
    """Runs the model generation and saves raw text to a temp JSON file."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Process started: Generating results for {checkpoint_path} on {device}")

    # Initialize the model and load the specific checkpoint weights
    model = init_model(get_lora_model=True) 
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model, 
        padding=True
    )

    test_dataloader = DataLoader(
        dataset, 
        batch_size=eval_batch_size, 
        shuffle=True, 
        collate_fn=data_collator
    )
    
    state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path))
    
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_inputs = []

    with torch.no_grad():
        for batch in test_dataloader:
            generated_tokens = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=100
            )
            
            labels = batch["labels"]
            
            # FIXED: Convert to standard Python lists for JSON serialization
            all_inputs.extend(batch["input_ids"].cpu().numpy().tolist())
            all_preds.extend(generated_tokens.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Save to temp file
    data = {"all_preds": all_preds, "all_labels": all_labels, "all_inputs": all_inputs}
    with open(temp_file_path, "w") as f:
        json.dump(data, f)
        
    logger.info(f"Inference complete for {checkpoint_path}. Exiting process to free VRAM.")


def run_metrics_process(checkpoint_name, tokenizer, return_dict):
    """Reads the temp JSON file, calculates metrics, and updates the shared dict."""
    logger.info(f"Process started: Calculating metrics for {checkpoint_name}")
    
    with open(temp_file_path, "r") as f:
        data = json.load(f)
    
    inputs = data["all_inputs"]
    preds = data["all_preds"]
    labels = data["all_labels"]
    
    # FIXED: Temporarily cast to numpy array for the masking condition
    labels = [np.where(np.array(label) != -100, label, tokenizer.pad_token_id).tolist() for label in labels]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_sources = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    decoded_sources = [src.strip() for src in decoded_sources]

    logger.info("Calculating BLEU")
    bleu_score = BLEU().corpus_score(decoded_preds, [decoded_labels]).score
    
    logger.info("Calculating chrF")
    chrf_metric = evaluate.load("chrf")
    chrf_score = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    
    logger.info("Calculating TER")
    ter_metric = evaluate.load("ter")
    ter_score = ter_metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]

    logger.info("Calculating COMET")
    comet_metric = evaluate.load("comet", "wmt20-comet-da")
    comet_res = comet_metric.compute(
        sources=decoded_sources,
        predictions=decoded_preds,
        references=decoded_labels,
    )
    comet_score = comet_res["mean_score"]
    
    metrics = {
        "bleu": bleu_score, 
        "chrf": chrf_score, 
        "ter": ter_score, 
        "comet": comet_score
    }

    # Save results to the shared dictionary
    return_dict[checkpoint_name] = metrics
    logger.info(f"Metrics complete for {checkpoint_name}. Exiting process.")


class evaluation_model:

    def __init__(self, eval_batch_size: int, wandb_eval_run_name: str):
        self.eval_batch_size = eval_batch_size
        load_dotenv(override=True)

        logger.info("Setting up the tokenizer for evaluation")
        self.tokenizer = init_tokenizer()

        logger.info("Setting up the test tokenized dataset")
        _, self.test_tokenized_dataset = get_tokenized_dataset(self.tokenizer, 0)
        self.test_tokenized_dataset = self.test_tokenized_dataset.select(range(20)) # comment it later

        # Get list of checkpoints
        self.checkpoints = [d for d in os.listdir(model_save_path) if d.startswith("checkpoint-")]
        self.checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        logger.info(f"Found the total saved checkpoints to be {len(self.checkpoints)}")

        self.wandb_eval_run_name = wandb_eval_run_name

        logger.info("Initiating the multi-processing pipeline")
        self.manager = mp.Manager()
        self.return_dict = self.manager.dict()

    def start_eval(self):
        # Initialize wandb here in the main process
        wandb.init(name=self.wandb_eval_run_name)
        
        for checkpoint in tqdm(self.checkpoints, desc="Evaluating Checkpoints"):
            logger.info(f"Initiating the evaluation for {checkpoint}")
            checkpoint_path = os.path.join(model_save_path, checkpoint)

            # STEP 1: Spawn and run Inference (Notice the comma making args a tuple)
            p_infer = mp.Process(
                target=run_inference_process, 
                args=(checkpoint_path, self.eval_batch_size, self.tokenizer, self.test_tokenized_dataset)
            )
            p_infer.start()
            p_infer.join() 
            
            # STEP 2: Spawn and run Metrics on the output (Notice the comma making args a tuple)
            p_metrics = mp.Process(
                target=run_metrics_process, 
                args=(checkpoint, self.tokenizer, self.return_dict)
            )
            p_metrics.start()
            p_metrics.join() 

            # IMPORTANT: Log to wandb from the main process, not the spawned child
            if checkpoint in self.return_dict:
                metrics = self.return_dict[checkpoint]
                wandb.log(metrics)
                logger.info(f"Logged metrics for {checkpoint}: {metrics}")
            
            logger.info(f"All done for {checkpoint}")

        # Finish wandb run
        wandb.finish()
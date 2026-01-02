from sacrebleu.metrics import BLEU
import evaluate
import os
import numpy as np
from environment_variables import model_save_path, tokenized_data_path
from setup_logging import logger
from initialize_model import load_save_model, init_tokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
import argparse

parser = argparse.ArgumentParser(description="Pass the language you want to start the evaluation")
parser.add_argument("--language", default="French", choices=["Portuguese","French","Spanish","German"], help="Select the language to translate to")

args = parser.parse_args()

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu_score = BLEU().corpus_score(decoded_preds, [decoded_labels]).score
    
    # chrF
    chrf_metric = evaluate.load("chrf")
    chrf_res = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)
    chrf_score = chrf_res["score"]
    
    # TER
    ter_metric = evaluate.load("ter")
    ter_res = ter_metric.compute(predictions=decoded_preds, references=decoded_labels)
    ter_score = ter_res["score"]

    # Get sources from the evaluation dataset (assuming order is preserved)
    #sources = final_dataset['val']['eng']


    # # COMET
    # comet_metric = evaluate.load("comet", "wmt20-comet-da")  # defaults to a model checkpoint
    # comet_res = comet_metric.compute(
    #     sources=sources,
    #     predictions=decoded_preds,
    #     references=decoded_labels,
    #     batch_size=8,
    #     #model_name_or_path="Unbabel/wmt22-comet-da"  # example checkpoint
    # )
    # The returned dictionary may include e.g. 'scores' (list) and 'mean_score'
    #comet_score = comet_res["mean_score"] if "mean_score" in comet_res else sum(comet_res["scores"]) / len(comet_res["scores"])
    
    return {"bleu": bleu_score, "chrf" : chrf_score, "ter" : ter_score}

# Define paths
output_dir = os.path.join(model_save_path, args.language)
logger.info(f"Fetching all the checkpoints from {output_dir}")
checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
checkpoint_dirs.sort(key = lambda x : int(x.split('-')[-1]))

dataset = load_from_disk(os.path.join(tokenized_data_path, args.language))

for checks in checkpoint_dirs:
    model = load_save_model(args.language, checks)
    tokenizer, target_lang = init_tokenizer(args.language)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id
    )

    # Define training args for evaluation only (minimal settings)
    eval_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # Can reuse or set a temp dir
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        bf16=True,
        report_to="none",  # No logging
    )

    # Create a new trainer for evaluation
    eval_trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset["val"],  # Or "test" if preferred
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # gen_kwargs={
    # "forced_bos_token_id": target_lang,  
    # "max_length": 128,
    # "num_beams": 4
    # }

    # Run evaluation
    eval_results = eval_trainer.evaluate()
    print(f"Results for {checks}: {eval_results}")
    print("-" * 50)

    # Optional: Clean up to free memory
    del model
    del eval_trainer
    import torch
    torch.cuda.empty_cache()


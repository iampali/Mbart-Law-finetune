# pyright: ignore[reportMissingImports]

from sacrebleu.metrics import BLEU
import evaluate
from tokenize_dataset import initlized_tokenizer
import numpy as np

def compute_metrics(eval_preds):
    tokenizer = initlized_tokenizer()
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
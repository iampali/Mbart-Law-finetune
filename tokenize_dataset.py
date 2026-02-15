import random
from datasets import Dataset, DatasetDict

max_length = 250

def create_dataset_dict(df):
    dataset = Dataset.from_pandas(df)
    return dataset

# Preprocessing function
def preprocess_function(examples, tokenizer):
    inputs = examples["source"]
    targets = examples["target"]

    tokenizer.src_lang = examples['src_lang']
    tokenizer.tgt_lang = examples['tgt_lang']
    
    # Tokenize inputs with source language
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
    
    # labels["input_ids"] = [
    #     [(l if l != tokenizer.pad_token_id else -100) for l in label]
    #     for label in labels["input_ids"]
    # ]

    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def map_dataset(tokenizer, dataset):

    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=False, 
        remove_columns=dataset["train"].column_names )

    return tokenized_datasets

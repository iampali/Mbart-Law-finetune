import random
from datasets import Dataset, DatasetDict
from environment_variables import langs, lang_codes

max_length = 250

def generate_pairs(example):
    pairs = []
    for src_lang in langs:
        for tgt_lang in langs:
            if src_lang != tgt_lang:
                pairs.append({
                        'source': example[src_lang],
                        'target': example[tgt_lang],
                        'src_lang' : lang_codes[src_lang],
                        'tgt_lang' : lang_codes[tgt_lang]
                })
    # Optional: Shuffle and sample to avoid redundancy/excess data
    random.shuffle(pairs)
    return {'pairs': pairs[:10]}  # Limit per example if dataset is large


def create_dataset_dict(df):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(generate_pairs, remove_columns=dataset.column_names)

    def pair_generator():
        for example in dataset:
            for pair in example['pairs']:
                yield pair
    
    dataset = Dataset.from_generator(pair_generator)

    split_dataset = dataset.train_test_split(test_size = 0.2)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    val_test = test_dataset.train_test_split(test_size=0.5)
    val_dataset = val_test['train']
    test_dataset = val_test['test']

    final_dataset = DatasetDict({ "train" : train_dataset, 
                                "val" : val_dataset,
                                "test" : test_dataset})
    
    return final_dataset

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

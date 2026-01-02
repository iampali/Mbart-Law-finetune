from initialize_model import init_tokenizer

from datasets import Dataset, DatasetDict


max_length = 250


def create_dataset_dict(df):
    dataset_dict = Dataset.from_pandas(df)
    split_dataset = dataset_dict.train_test_split(test_size = 0.2)
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
        batched=True, 
        remove_columns=dataset["train"].column_names )

    return tokenized_datasets

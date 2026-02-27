from datasets import Dataset
from environment_variables import tokenized_data_path
from setup_logging import logger

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
        text_target=targets,
        truncation=True
    )

    return model_inputs


def map_dataset(tokenizer, dataset):

    def tokenize_shard(shard):
        return shard.map(
                lambda examples: preprocess_function(examples, tokenizer),
                batched=False, 
                remove_columns=dataset["train"].column_names 
            )
    

    batch_size = 100000
    num_rows = len(dataset['train'])
    for start in range(0, len(dataset['train']), batch_size):
        end = min(start + batch_size, num_rows)
        shard = dataset['train'].select(range(start, end))
        tokenized = tokenize_shard(shard)
        tokenized.save_to_disk(f"{tokenized_data_path}/train_shard_{start // batch_size}")
        logger.info(f"Succesfully tokenized and saved shard no {start // batch_size} ")
    
    tokenized = tokenize_shard(dataset['test'])
    tokenized.save_to_disk(f"{tokenized_data_path}/test_shard")
    logger.info(f"Succesfully tokenized and saved final test shard")


    # tokenized_datasets = dataset.map(
    #     lambda examples: preprocess_function(examples, tokenizer),
    #     batched=False, 
    #     remove_columns=dataset["train"].column_names )

    # return tokenized_datasets

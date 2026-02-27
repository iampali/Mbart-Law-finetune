from tokenize_dataset import create_dataset_dict, map_dataset
from format_dataset import get_final_dataframe
import os
from environment_variables import tokenized_data_path
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from setup_logging import logger
from pathlib import Path

def load_dataset():
    data_dir = Path(tokenized_data_path)
    shard_paths = [f for f in data_dir.iterdir() if not f.is_file() and f.name.split('_')[0] == 'train']
    datasets_list = [load_from_disk(path) for path in shard_paths]
    train_tokenized_dataset = concatenate_datasets(datasets_list)
    test_tokenized_dataset = load_from_disk(data_dir/'test_shard')
    return train_tokenized_dataset, test_tokenized_dataset

def get_tokenized_dataset(tokenizer, dataset_length):

    """
    In this function, we're checking if the tokenized dataset is already avaiable in the path or not. If the tokneized dataset if already present then we're gonna take that from that part, otherwise we'll tokenize the data from scrach.

    Parameters
    tokenizer : tokenizer to tokenize data. Arleady set the src and tgt language.

    language : which language are we choosing to toeknize. The choices are ["French","Portuguese","Spanish","German"]

    dataset_length : default value is 0, a certain amount from dataset like if you want to train the model for first 10000 rows then specify dataset_length = 10000
    """
    
    if os.path.exists(tokenized_data_path):
        logger.info(f"Tokenized dataset is already present at {tokenized_data_path}")
    else:
        # Apply preprocessing
        logger.info(f"Tokenized dataset for not found.. creating and tokenzing data")
        train_df, test_df = get_final_dataframe()
        train_dataset, test_dataset = create_dataset_dict(train_df), create_dataset_dict(test_df)
        dataset = DatasetDict({ "train" : train_dataset,
                                "test" : test_dataset})
        logger.info("Created the dataset.. now starting tokenizing it")
        map_dataset(tokenizer, dataset)        
        logger.info(f"Tokenization successfull and saved the tokenized data on {tokenized_data_path}")
    
    train_tokenized_dataset, test_tokenized_dataset = load_dataset()
    if dataset_length == 0 :
        return train_tokenized_dataset, test_tokenized_dataset

    return filter_dataset(train_tokenized_dataset, dataset_length), test_tokenized_dataset


def filter_dataset(tokenized_datasets, dataset_length):
    logger.info(f"Extracting first {dataset_length} from the train dataset.")

    return tokenized_datasets.select(range(dataset_length))


from tokenize_dataset import create_dataset_dict, map_dataset
from format_dataset import get_final_dataframe
import os
from environment_variables import tokenized_data_path
from datasets import load_from_disk, DatasetDict
from setup_logging import logger


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
        tokenized_datasets = load_from_disk(tokenized_data_path)
    else:
        # Apply preprocessing
        logger.info(f"Tokenized dataset for not found.. creating and tokenzing data")
        train_df, test_df = get_final_dataframe()
        train_dataset, test_dataset = create_dataset_dict(train_df), create_dataset_dict(test_df)
        dataset = DatasetDict({ "train" : train_dataset,
                                "test" : test_dataset})
        logger.info("Created the dataset.. now starting tokenizing it")
        tokenized_datasets = map_dataset(tokenizer, dataset)
        tokenized_datasets.save_to_disk(tokenized_data_path)
        logger.info(f"Tokenization successfull and saved the tokenized data on {tokenized_data_path}")

    if dataset_length == 0 :
        return tokenized_datasets

    return filter_dataset(tokenized_datasets, dataset_length)


def filter_dataset(tokenized_datasets, dataset_length):
    logger.info(f"Extracting first {dataset_length} from the dataset.")
    train_data_length = len(tokenized_datasets['train'])
    test_data_length = len(tokenized_datasets['test'])
    per_train = round(dataset_length / train_data_length * 100, 2)
    test_length = round(per_train * test_data_length / 100)

    return DatasetDict({
    "train": tokenized_datasets["train"].select(range(dataset_length)),
    #"val": tokenized_datasets["val"].select(range(val_length)),
    "test": tokenized_datasets["test"].select(range(test_length))
    })


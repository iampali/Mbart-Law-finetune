from environment_variables import checkpoint, model_save_path
import torch
import os
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from setup_logging import logger

def init_tokenizer() :

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return tokenizer

def init_model(get_lora_model : bool = True):

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
    )


    model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
    )

    if get_lora_model:

        model = prepare_model_for_kbit_training(model)

        # Define QLoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "all-linear"
                # "q_proj",
                # "k_proj",
                # "v_proj",
                # "out_proj",
                # "gate_proj",
                # "up_proj",
                # "down_proj"
            ],  # Typical targets for encoder-decoder models like mBART
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )

        # Apply QLoRA to the model
        model = get_peft_model(model, lora_config)
        logger.info("Got the LORA config Model")
        # logger.info(f"Total trainable parameter in model are {model.print_trainable_parameters()}")

    return model


def load_save_model(language, model_checkpoint_name):
    model_path = os.path.join(model_save_path, language, model_checkpoint_name)
    logger.info(f"Successfully load the model from path {model_path}")
    new_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = PeftModel.from_pretrained(new_model, model_path)
    return model
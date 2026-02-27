import torch
from initialize_model import init_tokenizer, load_save_model, init_model
import argparse
from setup_logging import logger
from environment_variables import lang_codes, checkpoint
from vllm import LLM


parser = argparse.ArgumentParser(description="Pass input_text, language and model_checkpoint_name")
parser.add_argument("--input_text", default="something", type=str, required=False, help="Input text to translate")
parser.add_argument("--language", default="French", choices=["Portuguese","French","Spanish","German"], help="Select the language to translate to")
parser.add_argument("--model_checkpoint_name", type=str, required=False, help="Model name to use")
args = parser.parse_args()



def translate(input_text, language, model_checkpoint_name):
    
    if args.model_checkpoint_name :
        model = load_save_model(language, model_checkpoint_name)
        logger.info(f"Loading model {model_checkpoint_name} for {language}")
    else:
        model = init_model(get_lora_model=False)
        logger.info(f"Loading base model from {checkpoint} for {language}")
    logger.info(f"Model successfully loaded now Loading tokenizer for {language}")
    llm = LLM(model=model)
    tokenizer = init_tokenizer()
    target_lang = lang_codes[language]
    logger.info(f"The lang code for {language} is {target_lang}")
    logger.info(f"Tokenizer successfully loaded now enjoy translating :-D ")
    while input_text != 'exit':
        input_text = input("Enter text to translate: ")
        #target_lang = input("Enter the target_lang to translate: ")
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
        # outputs = model.generate(**inputs, max_new_tokens=100)
        # Generate Bemba translation
        with torch.no_grad():
            outputs = llm.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),  # Force start with Bemba lang
        )
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        for output in outputs:
            print({output.outputs[0].text})


translate(args.input_text, args.language, args.model_checkpoint_name)
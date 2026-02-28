checkpoint = "facebook/nllb-200-1.3B"
tokenized_data_path = "./tokenized_dataset/"
model_save_path = "./output/"
raw_data_path = "./data/"
temp_file_path = './temp/data.json' # used to save the model output during evaluation before killing the process
wandb_project_name = "NLLB"

langs = ['English', 'French', 'Portuguese', 'Finnish', 'Maltese']

lang_codes = {
    "English": "eng_Latn",
    "Portuguese": "por_Latn",
    "German": "deu_Latn",
    "Finnish" : "fin_Latn",
    "Maltese" : "mlt_Latn"
} # Adjust codes as needed
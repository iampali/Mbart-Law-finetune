checkpoint = "facebook/nllb-200-3.3B"
tokenized_data_path = "./tokenized_dataset/"
model_save_path = "./output/"
raw_data_path = "./data/"


langs = ['English', 'French', 'Portuguese', 'Finnish', 'Maltese']

lang_codes = {
    "English": "eng_Latn",
    "Portuguese": "por_Latn",
    "German": "deu_Latn",
    "Finnish" : "fin_Latn",
    "Maltese" : "mlt_Latn"
} # Adjust codes as needed
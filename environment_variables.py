checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
tokenized_data_path = "./tokenized_dataset/"
model_save_path = "./output/"
raw_data_path = "./data/"


langs = ['english', 'French', 'Portuguese', 'Spanish', 'German']

lang_codes = {
    "english": "en_XX",
    "Portuguese": "pt_XX",
    "Spanish": "es_XX",
    "French": "fr_XX",
    "German": "de_DE"
} # Adjust codes as needed
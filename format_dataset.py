from lxml import etree
import pandas as pd
from pathlib import Path
import re
from environment_variables import raw_data_path
from setup_logging import logger


def get_data(lang_name):
    # Path to TMX file
    tmx_path = f"data/English-{lang_name}.tmx"

    # Parse the TMX XML
    tree = etree.parse(tmx_path)
    root = tree.getroot()

    rows = []
    # TMX content is inside <body>
    body = root.find("body")

    for tu in body.findall("tu"):
        en_text = None
        second_text = None

        for tuv in tu.findall("tuv"):
            # Language attribute (xml:lang or lang)
            lang = (
                tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                or tuv.attrib.get("lang")
            )
            seg = tuv.find("seg")
            if seg is None:
                continue

            if lang == "en":
                en_text = seg.text
            else:
                second_text = seg.text

        # Only keep pairs where both languages exist
        if en_text and second_text:
            rows.append({
                "english": en_text,
                lang_name : second_text
            })

    # Create DataFrame
    return pd.DataFrame(rows)

def get_final_dataframe():
    
    dir_path = Path(raw_data_path)
    files = [f for f in dir_path.iterdir() if f.is_file()]
    files = sorted(files, key=lambda x : x.stat().st_size)
    languages = [f.name.split('.')[0].split('-')[1] for f in files]
    # data = {}
    merged_df = pd.DataFrame()
    for language in languages:
        
        logger.info(f"Creating dataframe for {language}")
    
        # data[language] = get_data(language)
        # data[language] = data[language].map(clean_text)
        # data[language] = data[language].sort_values("english")

        df = get_data(language)
        df = df.map(clean_text).sort_values("english")
        
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on="english", how="inner")

    # base_df = min(data.values(), key=len)
    # merged_df = base_df.copy()

    # for df in data.values():
    #     if df is not base_df:
    #         merged_df = merged_df.merge(df, on="english", how="inner")

    
    
    merged_df = merged_df.drop_duplicates(subset=["english"])

    # for language in list(merged_df.columns):
    
    #     if language == "Portuguese" :
    #         target_lang = '<pt_XX>'

    #     elif language == "French" :
    #         target_lang = '<fr_XX>'

    #     elif language == "Spanish" :
    #         target_lang = '<es_XX>'
        
    #     elif language == "english":
    #         target_lang = '<en_XX>'
        
    #     else :
    #         target_lang = '<de_DE>'
        
    #     merged_df[language] = merged_df[language].apply(lambda x : f"{target_lang} {x}")

    # final_df = pd.concat([df for df in data.values()], ignore_index=True)

    return merged_df


def clean_text(text):
    if not isinstance(text, str):
        return text

    # Replace non-breaking space
    text = text.replace('\xa0', ' ')

    # Normalize dashes
    text = re.sub(r'[‐-–—]', '-', text)

    # Normalize quotes
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")

    # Remove replacement character
    text = text.replace('�', '')

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


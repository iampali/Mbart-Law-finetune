from lxml import etree
import pandas as pd
from pathlib import Path
import re
from environment_variables import raw_data_path, lang_codes
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
                "source": en_text,
                "target" : second_text,
                "src_lang" : lang_codes['English'] ,
                "tgt_lang" : lang_codes[lang_name]
            })

    # Create DataFrame
    df = pd.DataFrame(rows)
    test_split = round(len(df) * 0.9)
    return df[:test_split], df[test_split:]

def get_final_dataframe():
    
    dir_path = Path(raw_data_path)
    files = [f for f in dir_path.iterdir() if f.is_file()]
    files = sorted(files, key=lambda x : x.stat().st_size)
    languages = [f.name.split('.')[0].split('-')[1] for f in files]
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for language in languages:
        
        logger.info(f"Creating dataframe for {language}")
    
        # data[language] = get_data(language)
        # data[language] = data[language].map(clean_text)
        # data[language] = data[language].sort_values("english")

        dfs = get_data(language)
        for df in dfs:
            df = df.map(clean_text).sort_values("source")

        train_df = pd.concat([train_df, dfs[0]], ignore_index=True)
        test_df = pd.concat([test_df, dfs[1]], ignore_index=True)

    return train_df, test_df


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


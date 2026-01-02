from lxml import etree
import pandas as pd
import os
from pathlib import Path
import re


def get_data(file_name):
    # Path to TMX file
    tmx_path = f"data/{file_name}.tmx"

    # Parse the TMX XML
    tree = etree.parse(tmx_path)
    root = tree.getroot()

    rows = []

    #print(f"Creating the dataframe for {lang_name} language")
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
                "target" : second_text
            })

    # Create DataFrame
    return pd.DataFrame(rows)

def get_final_dataframe(language):
    filename = f'English-{language}'
    
    print(f"Creating dataframe for {language}", flush=True)
    
    df = get_data(filename)
    df =  df.drop_duplicates(subset=["source"])
    df = df.map(clean_text)
    #final_df = pd.concat([df for df in data.values()], ignore_index=True)

    return df


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


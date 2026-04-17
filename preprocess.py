import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

def load_data():
    df = pd.read_csv("data.csv")
    df['text'] = df['text'].apply(clean_text)
    return df
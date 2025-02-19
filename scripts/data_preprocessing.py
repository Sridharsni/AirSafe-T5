import pandas as pd
import re

def preprocess_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = df.dropna()
    df['clean_text'] = df['text'].str.lower().str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
    df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")

if __name__ == "__main__":
    preprocess_data("data/raw_atc_text.csv", "data/cleaned_atc_text.csv")

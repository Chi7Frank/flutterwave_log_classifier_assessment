import pandas as pd
import re

def clean_text(text):
    text = text.lower()# Change case to lowercase and remove hex codes or IPs
    text = re.sub(r'\[.*?\]', '', text) # Removes text inside []
    text = re.sub(r'\d+', '', text)    # Removes numbers
    return text

df = pd.read_csv('data/Flutterwave AI Engineer Assessment Dataset.csv')
df['clean_message'] = df['log_message'].apply(clean_text)
df.to_csv('data/cleaned_logs.csv', index=False)
print("Data cleaned and saved to data/cleaned_logs.csv")
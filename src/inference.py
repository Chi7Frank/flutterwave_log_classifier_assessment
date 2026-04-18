import pandas as pd
import joblib
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Load saved model and vectorizer
model = joblib.load('models/classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def run_inference():
    # Load the data
    df = pd.read_csv('data/cleaned_logs.csv')
    
    # Predict all logs
    vectors = vectorizer.transform(df['clean_message'])
    df['prediction'] = model.predict(vectors)
    
    # Add structured summary
    df['summary'] = df['prediction'].apply(lambda x: f"Log entry triggered a {x} error. Root cause identified via automated classification.")
    
    # Save to CSV
    df[['log_id', 'prediction', 'summary']].to_csv('outputs/predictions.csv', index=False)
    print("Inference complete. Results saved to outputs/predictions.csv")

if __name__ == "__main__":
    run_inference()
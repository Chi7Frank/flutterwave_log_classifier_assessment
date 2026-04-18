import pandas as pd
from sklearn.metrics import classification_report
import joblib

# Load data and model
df = pd.read_csv('data/cleaned_logs.csv')
model = joblib.load('models/classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Transform and Predict
X = vectorizer.transform(df['clean_message'])
predictions = model.predict(X)

# Generate Metrics
report = classification_report(df['root_cause_label'], predictions)
with open('outputs/metrics.json', 'w') as f:
    f.write(report)

print("Evaluation complete. Results saved to outputs/metrics.json")
print("\n--- Metrics Report ---\n")
print(report)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Loading data
df = pd.read_csv('data/cleaned_logs.csv')

# Converting text to mathematical form using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
y = df['root_cause_label']

# Training  the Model
model = LogisticRegression()
model.fit(X, y)

# Save the work
joblib.dump(model, 'models/classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("Training complete. Model and vectorizer saved to models/")
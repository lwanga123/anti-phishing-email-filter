import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

print("🔄 Starting model retraining with approved feedback...")

# Load original data
try:
    original = pd.read_csv('cleaned_data.csv')
except:
    original = pd.DataFrame(columns=['text', 'label'])

# Load approved feedback and force integer labels
try:
    feedback = pd.read_csv('feedback_pending.csv', header=None, names=['text', 'label'])
    feedback['label'] = feedback['label'].astype(int)   # <-- This fixes the error
    if len(feedback) > 0:
        df = pd.concat([original, feedback], ignore_index=True)
        print(f"✅ Added {len(feedback)} new feedback examples")
    else:
        df = original
        print("No new feedback — using existing data")
except:
    df = original
    print("No feedback file yet — using existing data")

# Ensure labels are integer
df['label'] = df['label'].astype(int)

# Train the model
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tfidf, y)

# Save the new model
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("🎉 Model successfully retrained and saved!")
print(f"Total training examples now: {len(df)}")
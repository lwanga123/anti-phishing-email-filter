import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🧠 Starting Model Training...")

# 1. Load Cleaned Data
df = pd.read_csv('cleaned_data.csv')
X = df['text']
y = df['label']

# 2. Vectorization (Turning text into numbers)
print("Converting text to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest
print("Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"\n🎉 Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. SAVE THE BRAIN
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\n✅ Success! 'phishing_model.pkl' and 'tfidf_vectorizer.pkl' are updated.")
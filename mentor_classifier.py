import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load combined dataset
df = pd.read_csv("full_chatbot_dataset.csv")

# Vectorize text
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "mentor_classifier.pkl")
joblib.dump(vectorizer, "mentor_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")

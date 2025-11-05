from src.data.dataset import load_dataset
from src.data.preprocess import preprocess_texts
from src.model.spam_model import SpamClassifier

# Load data
texts, labels = load_dataset("data/train.csv")

# Preprocess
X, vectorizer = preprocess_texts(texts)

# Train model
model = SpamClassifier()
model.train(X, labels, vectorizer)

# Save model
model.save()
print("Training complete. Model saved as spam_model.pkl")

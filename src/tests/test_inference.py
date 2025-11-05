from src.model.spam_model import SpamClassifier

model = SpamClassifier()
model.load()

emails = ["You won a gift card!", "Please review the report"]
preds = model.predict(emails)
labels = ["spam" if p==1 else "safe" for p in preds]

print(labels)

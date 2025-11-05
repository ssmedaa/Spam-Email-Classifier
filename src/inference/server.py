from flask import Flask, request, jsonify
from src.model.spam_model import SpamClassifier
from src.inference.batcher import batcher

app = Flask(__name__)

# Load trained model
model = SpamClassifier()
model.load()  

@app.route("/", methods=["GET"])
def home():
    return "Spam Email Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    emails = data.get("emails", [])

    if not emails:
        return jsonify({"error": "No emails provided"}), 400

    all_labels = []
    for batch in batcher(emails, batch_size=16):
        preds = model.predict(batch)
        labels = ["spam" if p == 1 else "safe" for p in preds]
        all_labels.extend(labels)

    return jsonify({"predictions": all_labels})

if __name__ == "__main__":
    
# localhost:5000
    app.run(debug=True)

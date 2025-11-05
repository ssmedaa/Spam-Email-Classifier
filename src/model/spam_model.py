from sklearn.linear_model import LogisticRegression
import pickle

class SpamClassifier:
    def __init__(self):
        self.model = LogisticRegression()
        self.vectorizer = None

    def train(self, X, y, vectorizer):
        self.vectorizer = vectorizer
        self.model.fit(X, y)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def save(self, path="spam_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load(self, path="spam_model.pkl"):
        with open(path, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)

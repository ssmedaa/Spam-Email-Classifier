from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_texts(texts):
  
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

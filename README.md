# Spam Email Classifier using LLMs

**Purpose:** Detect whether an email is **spam** or **safe** using a transformer-based model.  
**Goal:** Provide a fast, scalable, and easy-to-use solution to classify emails in real-time.

##





## **Workflow**

1. **Collect Data** – Emails labeled as spam (1) or safe (0) in `data/train.csv`.  
2. **Preprocess** – Clean and tokenize text for the model.  
3. **Load Dataset** – Read CSV and split into training/validation sets.  
4. **Define Model** – Transformer-based `SpamClassifier` (e.g., DistilBERT).  
5. **Train** – Train model with batching and save trained model.  
6. **Evaluate** – Test on validation set; track accuracy/F1-score.  
7. **Deploy API** – Run Flask server with `/predict` endpoint.  
8. **Test & Predict** – Send sample emails to get spam/safe predictions.  
9. **Update Data** – Add new emails and retrain as needed.

##
**Dependencies**
  - Python 3.11+
  - Flask
  - PyTorch (or torch-cpu)
  - scikit-learn
  - pandas
  - transformers


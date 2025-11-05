import pandas as pd

def load_dataset(file_path="data/train.csv"):
    """
    Load spam dataset. CSV should have 'text' and 'label' columns
    """
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

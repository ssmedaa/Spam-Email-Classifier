import os
import pytest
import pandas as pd
from src.data.dataset import load_dataset

# Create a temporary CSV for testing
TEST_CSV = "src/tests/temp_test.csv"

def setup_module(module):
    
    df = pd.DataFrame({
        "text": ["Win a prize!", "Please review the report"],
        "label": [1, 0]
    })
    df.to_csv(TEST_CSV, index=False)

def teardown_module(module):
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)

def test_load_dataset():
    texts, labels = load_dataset(TEST_CSV)
    assert len(texts) == 2
    assert len(labels) == 2
    assert texts[0] == "Win a prize!"
    assert labels[0] == 1
    assert labels[1] == 0

def test_empty_file():
    # Create empty CSV
    empty_csv = "src/tests/empty.csv"
    pd.DataFrame(columns=["text","label"]).to_csv(empty_csv, index=False)
    
    texts, labels = load_dataset(empty_csv)
    assert texts == []
    assert labels == []

    os.remove(empty_csv)

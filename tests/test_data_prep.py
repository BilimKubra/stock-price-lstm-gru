import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_prep import load_and_prepare

def test_shapes():
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(lookback=10)
    assert X_train.shape[1] == 10
    assert X_train.shape[2] == 1
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

def test_scaler_range():
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(lookback=10)
    assert y_train.min() >= -1.01
    assert y_train.max() <= 1.01

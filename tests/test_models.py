import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from models import LSTMModel, GRUModel

def test_lstm_output_shape():
    model = LSTMModel()
    x = torch.randn(4, 10, 1)
    out = model(x)
    assert out.shape == (4, 1)

def test_gru_output_shape():
    model = GRUModel()
    x = torch.randn(4, 10, 1)
    out = model(x)
    assert out.shape == (4, 1)

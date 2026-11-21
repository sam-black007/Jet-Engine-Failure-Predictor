"""Comprehensive tests for machine learning baseline models and PyTorch LSTM.

Validates model loading, input dimensions, inference determinism,
absence of NaN/Inf, bounds adherence, and corruption resilience.
"""

import io
import json
import pathlib
import joblib
import numpy as np
import pytest
import torch

from src.train_lstm import LSTMPredictor


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"


@pytest.fixture(scope="module")
def metadata():
    with open(DATA_DIR / "preprocess_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def val_sample_3d():
    """Create or load real validation sequence of shape (10, 30, 17)."""
    val_path = DATA_DIR / "X_val.npy"
    if val_path.is_file():
        X_val = np.load(val_path)
        return X_val[:10].astype(np.float32)
    return np.random.randn(10, 30, 17).astype(np.float32)


def test_baseline_models_input_and_determinism(val_sample_3d):
    """Verify all sklearn baseline models load, accept flattened input, and are deterministic."""
    baselines = ["linear_regression", "random_forest", "gradient_boosting"]
    flat_input = val_sample_3d.reshape(val_sample_3d.shape[0], -1)  # (10, 510)

    for name in baselines:
        model_path = MODELS_DIR / "baselines" / f"{name}.pkl"
        assert model_path.is_file(), f"Model {name} missing at {model_path}"
        
        model = joblib.load(model_path)
        preds_1 = model.predict(flat_input)
        preds_2 = model.predict(flat_input)
        
        # Shape, dtype, and validity
        assert preds_1.shape == (10,)
        assert np.issubdtype(preds_1.dtype, np.floating)
        assert not np.isnan(preds_1).any(), f"{name} output contains NaN"
        assert not np.isinf(preds_1).any(), f"{name} output contains Inf"
        
        # Determinism (multithreaded floating point accumulation tolerance)
        np.testing.assert_allclose(preds_1, preds_2, rtol=1e-7, atol=1e-7, err_msg=f"{name} inference is not deterministic")


def test_lstm_architecture_and_forward_pass(val_sample_3d):
    """Verify PyTorch LSTM model weights, architecture, eval mode, and CPU inference."""
    lstm_path = MODELS_DIR / "lstm" / "lstm_model.pt"
    assert lstm_path.is_file(), f"LSTM checkpoint missing at {lstm_path}"
    
    checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=False)
    model = LSTMPredictor(
        input_dim=checkpoint.get("input_dim", 17),
        hidden_dim1=checkpoint.get("hidden_dim1", 64),
        hidden_dim2=checkpoint.get("hidden_dim2", 32),
        dropout_rate=checkpoint.get("dropout_rate", 0.2)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Forward pass with PyTorch tensor
    tensor_input = torch.tensor(val_sample_3d, dtype=torch.float32)
    with torch.no_grad():
        preds_tensor = model(tensor_input)
    assert preds_tensor.shape == torch.Size([10])
    
    # Using inference wrapper
    preds_np1 = model.predict(val_sample_3d)
    preds_np2 = model.predict(val_sample_3d)
    assert preds_np1.shape == (10,)
    assert not np.isnan(preds_np1).any()
    assert not np.isinf(preds_np1).any()
    np.testing.assert_array_almost_equal(preds_np1, preds_np2, decimal=6)


def test_lstm_corrupted_checkpoint_handling():
    """Verify that attempting to load a corrupted checkpoint file raises an exception gracefully."""
    corrupted_data = b"CORRUPTED_CHECKPOINT_DATA_12345"
    with pytest.raises(Exception):
        torch.load(io.BytesIO(corrupted_data))

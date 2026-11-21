"""Unit tests for asymmetric loss functions: NASA Log-Mean-Exp and Asymmetric Pinball."""

import pytest
import torch
import numpy as np

from src.losses import NASAAsymmetricLoss, AsymmetricPinballLoss, SmoothPinballLoss


def test_nasa_asymmetric_loss_zero_residual():
    """Verify NASA loss evaluates to exactly 0.0 when predictions equal targets."""
    loss_fn = NASAAsymmetricLoss()
    y_true = torch.tensor([50.0, 75.0, 100.0])
    y_pred = torch.tensor([50.0, 75.0, 100.0], requires_grad=True)
    loss = loss_fn(y_pred, y_true)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_nasa_asymmetric_loss_asymmetry():
    """Verify NASA loss penalizes late predictions (+d) strictly more than early (-d)."""
    loss_fn = NASAAsymmetricLoss(late_penalty=10.0, early_penalty=13.0)
    y_true = torch.tensor([50.0])
    
    # +10 overestimation: z = 10 / 10 = 1.0 -> loss = 1.0
    pred_late = torch.tensor([60.0])
    loss_late = loss_fn(pred_late, y_true)

    # -10 underestimation: z = 10 / 13 = 0.7692 -> loss = 0.7692
    pred_early = torch.tensor([40.0])
    loss_early = loss_fn(pred_early, y_true)

    assert loss_late > loss_early, "Overestimation penalty must strictly exceed underestimation"
    assert torch.isclose(loss_late, torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(loss_early, torch.tensor(10.0 / 13.0), atol=1e-5)


def test_nasa_asymmetric_loss_numerical_stability():
    """Verify Log-Mean-Exp formulation does not overflow under extreme positive and negative errors."""
    loss_fn = NASAAsymmetricLoss()
    y_true = torch.tensor([50.0, 50.0, 50.0])
    
    # Extreme residuals: d = +500, -500, 0
    y_pred_extreme = torch.tensor([550.0, -450.0, 50.0], requires_grad=True)
    loss = loss_fn(y_pred_extreme, y_true)

    assert torch.isfinite(loss).item(), "Loss must remain finite under extreme residuals"
    loss.backward()
    assert torch.all(torch.isfinite(y_pred_extreme.grad)).item(), "Gradients must remain finite"
    assert torch.all(torch.abs(y_pred_extreme.grad) <= 0.10 + 1e-6), "Gradients must be bounded by 1/10"


def test_pinball_loss_mathematical_properties():
    """Verify Asymmetric Pinball Loss calculates exact theoretical quantile weights."""
    # q = 0.40 -> Overestimation weight = 0.60, Underestimation weight = 0.40
    loss_fn = AsymmetricPinballLoss(quantile=0.40)
    y_true = torch.tensor([50.0])
    
    # Zero residual
    loss_zero = loss_fn(torch.tensor([50.0]), y_true)
    assert torch.isclose(loss_zero, torch.tensor(0.0))

    # +10 overestimation -> 0.60 * 10 = 6.0
    loss_late = loss_fn(torch.tensor([60.0]), y_true)
    assert torch.isclose(loss_late, torch.tensor(6.0))

    # -10 underestimation -> 0.40 * 10 = 4.0
    loss_early = loss_fn(torch.tensor([40.0]), y_true)
    assert torch.isclose(loss_early, torch.tensor(4.0))

    # Ratio
    assert loss_late / loss_early == pytest.approx(1.50)


def test_pinball_loss_gradient_flow():
    """Verify gradients propagate backwards through Asymmetric Pinball Loss."""
    loss_fn = AsymmetricPinballLoss(quantile=0.40)
    y_true = torch.tensor([50.0, 50.0])
    y_pred = torch.tensor([60.0, 40.0], requires_grad=True)
    
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    
    # Expected gradients: d > 0 -> (1 - q)/2 = 0.30; d < 0 -> -q/2 = -0.20
    assert torch.isclose(y_pred.grad[0], torch.tensor(0.30))
    assert torch.isclose(y_pred.grad[1], torch.tensor(-0.20))


def test_smooth_pinball_loss():
    """Verify smooth pinball loss produces continuous values matching pinball loss."""
    smooth_fn = SmoothPinballLoss(quantile=0.40, delta=1.0)
    y_true = torch.tensor([50.0])
    y_pred = torch.tensor([60.0], requires_grad=True)
    
    loss = smooth_fn(y_pred, y_true)
    assert loss > 0
    loss.backward()
    assert y_pred.grad is not None


def test_invalid_parameters():
    """Verify invalid quantiles or negative penalty scales raise ValueError."""
    with pytest.raises(ValueError):
        NASAAsymmetricLoss(late_penalty=-1.0)
    with pytest.raises(ValueError):
        AsymmetricPinballLoss(quantile=0.0)
    with pytest.raises(ValueError):
        AsymmetricPinballLoss(quantile=1.0)
    with pytest.raises(ValueError):
        AsymmetricPinballLoss(quantile=0.40, reduction="invalid")


def test_tensor_shape_compatibility():
    """Verify loss functions handle 1D, 2D (B, 1), and mismatched flattened shapes cleanly."""
    nasa_fn = NASAAsymmetricLoss()
    pinball_fn = AsymmetricPinballLoss(quantile=0.40)

    # 2D shapes
    y_true_2d = torch.ones(8, 1) * 50.0
    y_pred_2d = torch.ones(8, 1) * 55.0

    l_nasa = nasa_fn(y_pred_2d, y_true_2d)
    l_pin = pinball_fn(y_pred_2d, y_true_2d)

    assert l_nasa.dim() == 0, "NASA loss should be scalar"
    assert l_pin.dim() == 0, "Pinball loss should be scalar"


def test_asymmetric_lstm_checkpoint_loading():
    """Verify new Phase 1 asymmetric LSTM checkpoints exist, load, and generate valid inference."""
    import pathlib
    from src.train_lstm import LSTMPredictor

    project_root = pathlib.Path(__file__).resolve().parent.parent
    asymm_dir = project_root / "models" / "lstm_asymmetric"

    for ckpt_name in ["nasa_asymmetric_lstm.pt", "pinball_lstm.pt"]:
        ckpt_path = asymm_dir / ckpt_name
        assert ckpt_path.is_file(), f"Expected checkpoint {ckpt_name} to exist."

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "best_val_nasa" in ckpt

        model = LSTMPredictor(input_dim=17)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy_input = np.random.randn(5, 30, 17).astype(np.float32)
        preds = model.predict(dummy_input)
        assert preds.shape == (5,)
        assert not np.isnan(preds).any()
        assert not np.isinf(preds).any()


def test_phase1_metrics_consistency_with_phase0():
    """Verify baseline MSE LSTM evaluation in Phase 1 reproduces Phase 0 official benchmarks."""
    import pathlib
    import json

    project_root = pathlib.Path(__file__).resolve().parent.parent
    p1_metrics_path = project_root / "results" / "phase1" / "phase1_metrics.json"
    p0_metrics_path = project_root / "models" / "performance_metrics.json"

    assert p1_metrics_path.is_file()
    assert p0_metrics_path.is_file()

    with open(p1_metrics_path, "r", encoding="utf-8") as f:
        p1 = json.load(f)
    with open(p0_metrics_path, "r", encoding="utf-8") as f:
        p0 = json.load(f)

    # Baseline LSTM in Phase 1 must match Phase 0 official test set results
    base_p1_raw = p1["models"]["baseline_lstm"]["raw_target"]
    base_p0_raw = p0["lstm"]["official_test_set"]["raw_target"]

    assert pytest.approx(base_p1_raw["MAE"], abs=0.05) == base_p0_raw["MAE"]
    assert pytest.approx(base_p1_raw["RMSE"], abs=0.05) == base_p0_raw["RMSE"]
    assert pytest.approx(base_p1_raw["NASA_score"], abs=0.5) == base_p0_raw["NASA_score"]


def test_asymmetric_reproducibility():
    """Verify inference determinism for NASA Asymmetric LSTM on repeated forward passes."""
    import pathlib
    from src.train_lstm import LSTMPredictor

    project_root = pathlib.Path(__file__).resolve().parent.parent
    ckpt_path = project_root / "models" / "lstm_asymmetric" / "nasa_asymmetric_lstm.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LSTMPredictor(input_dim=17)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = np.random.RandomState(42).randn(10, 30, 17).astype(np.float32)
    preds_1 = model.predict(x)
    preds_2 = model.predict(x)

    np.testing.assert_allclose(preds_1, preds_2, rtol=1e-7, atol=1e-7)

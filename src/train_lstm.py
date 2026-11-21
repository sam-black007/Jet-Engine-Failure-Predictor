"""Train LSTM neural network for Remaining Useful Life (RUL) prediction.

Architecture:
- Input: (batch_size, sequence_length=30, num_features=17)
- LSTM Layer 1: hidden_dim=64, return sequences=True
- Dropout (0.2)
- LSTM Layer 2: hidden_dim=32, return final state
- Dropout (0.2)
- Dense Layer: Linear(32 -> 1)
- Output: Predicted RUL (continuous value)

Uses PyTorch for robust CPU/GPU execution without long-path issues.
Saves model weights and full state to models/lstm/lstm_model.pt.
Updates models/performance_metrics.json with both sequence-level
and per-engine evaluation metrics.
"""

import os
import json
import pathlib
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_loader import load_dataset
from src.evaluation import compute_metrics, evaluate_per_engine


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMPredictor(nn.Module):
    """Deep LSTM Network for Turbofan Engine Degradation Forecasting."""

    def __init__(self, input_dim: int = 17, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout_rate: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, (hn, _) = self.lstm2(out1)
        # Use last hidden state or last time step
        last_step = out2[:, -1, :]
        last_step = self.dropout2(last_step)
        pred = self.fc(last_step)
        return pred.squeeze(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inference helper accepting 2D or 3D numpy arrays, returning 1D numpy predictions."""
        self.eval()
        if X.ndim == 2:
            # Flattened (N, 510) -> reshape to (N, 30, 17)
            X = X.reshape(X.shape[0], 30, -1)
        elif X.ndim == 3:
            pass
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")

        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32)
            preds = self.forward(tensor_x).cpu().numpy()
        return np.atleast_1d(preds)


def train_lstm_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    project_root = pathlib.Path(__file__).resolve().parent.parent
    processed_dir = project_root / "dataset" / "processed"
    models_dir = project_root / "models"
    lstm_dir = models_dir / "lstm"
    lstm_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(processed_dir / "preprocess_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    sensor_cols = meta["selected_sensors"]
    sensor_means = np.array([meta["sensor_means"][s] for s in sensor_cols], dtype=np.float32)
    sensor_stds = np.array([meta["sensor_stds"][s] for s in sensor_cols], dtype=np.float32)
    val_engine_ids = meta["val_engine_ids"]
    window = meta["window_length"]
    rul_clip = meta["rul_clip"]
    raw_df = load_dataset(meta.get("dataset_subset", "FD001"), kind="train")

    # Load preprocessed arrays
    print("Loading 3D sequences for LSTM training...")
    X_train = np.load(processed_dir / "X_train.npy").astype(np.float32)  # (14489, 30, 17)
    y_train = np.load(processed_dir / "y_train.npy").astype(np.float32)
    X_val = np.load(processed_dir / "X_val.npy").astype(np.float32)      # (3242, 30, 17)
    y_val = np.load(processed_dir / "y_val.npy").astype(np.float32)

    # PyTorch DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = LSTMPredictor(input_dim=X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    best_weights_path = lstm_dir / "lstm_model.pt"
    patience = 8
    patience_counter = 0
    max_epochs = 35

    print(f"\n--- Training LSTM ({max_epochs} max epochs) ---")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train MSE: {train_loss:.2f} | Val MSE: {val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': X_train.shape[2],
                'hidden_dim1': 64,
                'hidden_dim2': 32,
                'dropout_rate': 0.2,
                'best_val_loss': best_val_loss,
                'epoch': epoch
            }, best_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Load best checkpoint for final evaluation
    checkpoint = torch.load(best_weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"\nBest model saved to {best_weights_path} (Val MSE: {checkpoint['best_val_loss']:.2f})")

    # 1. Sequence-level validation
    y_pred_val = model.predict(X_val)
    seq_metrics = compute_metrics(y_val, y_pred_val)
    print("\nSequence-level Validation Metrics:")
    print(f"  MAE: {seq_metrics['MAE']}")
    print(f"  RMSE: {seq_metrics['RMSE']}")
    print(f"  NASA Score: {seq_metrics['NASA_score']}")

    # 2. Per-engine truncated-trajectory evaluation
    eng_eval = evaluate_per_engine(
        model=model,
        raw_df=raw_df,
        val_engine_ids=val_engine_ids,
        sensor_cols=sensor_cols,
        sensor_means=sensor_means,
        sensor_stds=sensor_stds,
        window_length=window,
        rul_clip=rul_clip,
        is_3d=True
    )
    eng_metrics = eng_eval["metrics"]
    print(f"Per-engine Truncated-trajectory Metrics ({eng_eval['num_engines_evaluated']} engines):")
    print(f"  MAE: {eng_metrics['MAE']}")
    print(f"  RMSE: {eng_metrics['RMSE']}")
    print(f"  NASA Score: {eng_metrics['NASA_score']}")

    # Update performance_metrics.json
    metrics_file = models_dir / "performance_metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)

    all_metrics["lstm"] = {
        "sequence_level": seq_metrics,
        "per_engine": eng_metrics,
        "training_details": {
            "best_epoch": checkpoint["epoch"],
            "best_val_mse": round(float(checkpoint["best_val_loss"]), 4)
        }
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Updated {metrics_file} with LSTM results.")


if __name__ == "__main__":
    train_lstm_model()

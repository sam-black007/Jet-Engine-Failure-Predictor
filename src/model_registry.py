"""Model Registry for Jet Engine Failure Predictor (THRUSTLAB Phase 2).

Manages model versions, deployment tags (e.g. 'production', 'challenger', 'baseline'),
checkpoint metadata, input signatures, and verified official test performance.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import numpy as np

from src.model_factory import create_model
from src.config_manager import MasterConfig


class ModelRegistry:
    """Persistent catalog and loader for trained prognostic models."""

    def __init__(self, registry_file: Optional[Union[str, Path]] = None):
        self.registry_file = Path(registry_file or "models/model_registry.json")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"active_tag": "pinball_q35", "models": {}}

    def _save(self) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)

    def register_model(
        self,
        tag: str,
        name: str,
        architecture: str,
        checkpoint_path: str,
        config: Optional[Union[MasterConfig, Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        input_dim: int = 14,
        seq_len: int = 30,
        is_active: bool = False
    ) -> Dict[str, Any]:
        """Register a trained model checkpoint in the registry."""
        entry = {
            "tag": tag,
            "name": name,
            "architecture": architecture,
            "checkpoint_path": str(checkpoint_path),
            "config": config.model_dump() if isinstance(config, MasterConfig) else (config or {}),
            "metrics": metrics or {},
            "input_dim": input_dim,
            "seq_len": seq_len,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "ready"
        }
        self.data["models"][tag] = entry
        if is_active or len(self.data["models"]) == 1:
            self.data["active_tag"] = tag
        self._save()
        return entry

    def list_models(self) -> List[Dict[str, Any]]:
        """Return list of all registered models."""
        return list(self.data.get("models", {}).values())

    def get_model_entry(self, tag: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata by tag."""
        return self.data.get("models", {}).get(tag)

    def set_active_model(self, tag: str) -> None:
        """Set the active/production model tag."""
        if tag not in self.data.get("models", {}):
            raise KeyError(f"Model with tag '{tag}' not found in registry.")
        self.data["active_tag"] = tag
        self._save()

    def get_active_tag(self) -> str:
        """Get the current active model tag."""
        return self.data.get("active_tag", "pinball_q35")

    def load_model(self, tag: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Instantiate and load weights for the specified model tag (or active model)."""
        target_tag = tag or self.get_active_tag()
        entry = self.get_model_entry(target_tag)
        if not entry:
            raise KeyError(f"Model '{target_tag}' not found in registry.")

        ckpt_path = Path(entry["checkpoint_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        arch = entry["architecture"]
        cfg_dict = entry.get("config", {})
        input_dim = entry.get("input_dim", 17)
        seq_len = entry.get("seq_len", 30)

        # Inspect checkpoint weights to dynamically infer input_dim if available
        state = None
        if ckpt_path.suffix in (".pt", ".pth"):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
            if isinstance(sd, dict):
                if "lstm1.weight_ih_l0" in sd:
                    input_dim = sd["lstm1.weight_ih_l0"].shape[1]
                elif "network.0.conv1.weight" in sd:
                    input_dim = sd["network.0.conv1.weight"].shape[1]
                elif "network.0.conv1.parametrizations.weight.original" in sd:
                    input_dim = sd["network.0.conv1.parametrizations.weight.original"].shape[1]
                elif "input_proj.weight" in sd:
                    input_dim = sd["input_proj.weight"].shape[1]

        master_cfg = MasterConfig.model_validate(cfg_dict) if cfg_dict else MasterConfig()
        master_cfg.model.architecture = arch
        model = create_model(master_cfg.model, input_dim=input_dim, seq_len=seq_len)

        # Load weights
        if isinstance(model, torch.nn.Module):
            if state is None:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            model.eval()

        return model, entry

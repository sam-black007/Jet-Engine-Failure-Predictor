"""THRUSTLAB — Model Registry (Streamlit Dashboard).

Manage production deployments, active checkpoints, and model tags.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from src.model_registry import ModelRegistry

st.set_page_config(
    page_title="THRUSTLAB — Model Registry",
    page_icon="📦",
    layout="wide"
)

st.title("📦 THRUSTLAB — Model Registry")
st.markdown("Audited catalog of production and candidate prognostic models.")

registry = ModelRegistry()
models = registry.list_models()

if not models:
    st.info("No models registered in registry.")
else:
    active_tag = registry.get_active_tag()
    st.success(f"⭐ **Current Active Production Model Tag**: `{active_tag}`")

    records = []
    for m in models:
        tag = m["tag"]
        is_active = (tag == active_tag)
        metrics = m.get("metrics", {})
        ckpt = Path(m["checkpoint_path"])
        size_mb = f"{ckpt.stat().st_size / (1024*1024):.2f} MB" if ckpt.exists() else "Missing"

        records.append({
            "Tag": tag,
            "Name": m["name"],
            "Architecture": m["architecture"],
            "Production Status": "⭐ ACTIVE" if is_active else "Candidate",
            "Raw NASA Score": metrics.get("NASA_score_raw", "N/A"),
            "Clipped NASA Score": metrics.get("NASA_score_clipped", "N/A"),
            "MAE": metrics.get("MAE", "N/A"),
            "Checkpoint Size": size_mb,
            "Registered": m.get("registered_at", "")[:19]
        })

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("Switch Active Production Model")
    tags = [m["tag"] for m in models]
    new_active = st.selectbox("Select Model to Promote to Active Production", tags, index=tags.index(active_tag))
    if st.button("Promote to Production"):
        registry.set_active_model(new_active)
        st.success(f"Successfully promoted `{new_active}` to Active Production!")
        st.rerun()

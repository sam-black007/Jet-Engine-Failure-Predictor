"""JETWISE AI — Contextual Help, Glossary, and Guided Setup Wizard.

Provides plain-language translations, aerospace engineering glossaries,
and step-by-step onboarding walkthroughs for beginner and advanced users.
"""

from typing import Dict, Any, List, Optional
import streamlit as st


# Plain-Language Dictionary for Simple Mode vs Advanced Engineering Mode
PLAIN_LANGUAGE_MAP: Dict[str, Dict[str, str]] = {
    "rul_prediction": {
        "simple": "Estimated Remaining Engine Life",
        "advanced": "RUL Quantile Regression Prediction (Pinball q10-q90)",
        "explanation": "Estimates how many operating cycles the engine can safely perform before requiring shop overhaul."
    },
    "ood_detection": {
        "simple": "Operating Condition Safety Check",
        "advanced": "Out-of-Distribution (OOD) Mahalanobis Detection",
        "explanation": "Determines whether the engine is operating outside the physical flight conditions the AI model was trained on."
    },
    "sensor_fdir": {
        "simple": "Sensor Fault & Health Check",
        "advanced": "Fault Detection, Isolation & Recovery (FDIR)",
        "explanation": "Identifies if sensor readings are genuine engine measurements or faulty sensors (drift, stuck values, or electrical noise)."
    },
    "epistemic_uncertainty": {
        "simple": "AI Prediction Confidence",
        "advanced": "Epistemic & Aleatoric Uncertainty Bound (90% Conformal)",
        "explanation": "Measures how confident the AI model is in its estimate. A smaller number means higher reliability."
    },
    "engine_health_score": {
        "simple": "Engine Overall Health",
        "advanced": "Thermodynamic & Mechanical Health Index (0-100)",
        "explanation": "A composite score where 100 means factory new condition and below 40 indicates degraded components needing attention."
    },
    "engine_risk_score": {
        "simple": "Operational Risk Level",
        "advanced": "Critical Excursion & Hazard Probability Score (0-100)",
        "explanation": "Higher values indicate increased risk of uncontained failure or severe operational degradation."
    },
    "simulation_validity": {
        "simple": "Physics Simulation Quality",
        "advanced": "7-Point Aerothermal Conservation & Monotonicity Verification",
        "explanation": "Verifies that the simulation obeys the laws of physics, including mass conservation, energy balance, and temperature limits."
    },
    "brayton_cycle": {
        "simple": "Jet Engine Thermodynamic Cycle",
        "advanced": "Continuous-Flow Open Brayton Gas-Turbine Cycle",
        "explanation": "The thermodynamic sequence of compression, combustion, expansion, and exhaust powering the jet engine."
    }
}


AEROSPACE_GLOSSARY: List[Dict[str, str]] = [
    {
        "term": "Remaining Useful Life (RUL)",
        "category": "Prognostics",
        "definition": "The estimated number of flight cycles remaining before an engine or component reaches its operational wear threshold.",
        "units": "Operating Cycles"
    },
    {
        "term": "Exhaust Gas Temperature (EGT / T50)",
        "category": "Aerothermodynamics",
        "definition": "The temperature of combustion gases exiting the low-pressure turbine. Monitored closely as an indicator of turbine blade stress.",
        "units": "°R / °C"
    },
    {
        "term": "High-Pressure Compressor Discharge (P30 / T30)",
        "category": "Aerothermodynamics",
        "definition": "The total pressure and temperature exiting the high-pressure compressor right before fuel combustion.",
        "units": "psia / °R"
    },
    {
        "term": "N1 & N2 Spool Speeds",
        "category": "Mechanical",
        "definition": "N1 is the rotational velocity of the low-pressure fan/booster shaft; N2 is the rotational velocity of the high-pressure core shaft.",
        "units": "RPM"
    },
    {
        "term": "Conformal Prediction",
        "category": "Machine Learning",
        "definition": "A mathematically rigorous statistical framework that guarantees prediction intervals cover the true value at a specified probability (e.g. 90%).",
        "units": "Percentage Coverage"
    },
    {
        "term": "Sensor Drift & Bias",
        "category": "Instrumentation",
        "definition": "A gradual systematic deviation (drift) or constant offset (bias) in a sensor's measurement from the true physical value.",
        "units": "Δ %"
    },
    {
        "term": "Bypass Ratio (BPR)",
        "category": "Propulsion",
        "definition": "The ratio between the mass flow rate of air that passes through the fan duct (bypass) and the mass flow rate that enters the engine core.",
        "units": "Dimensionless"
    },
    {
        "term": "Turbofan Digital Twin",
        "category": "Simulation",
        "definition": "A synchronized computational representation of the physical engine updating in real-time with telemetry, physics, and AI predictions.",
        "units": "3D WebGL / Three.js"
    }
]


def translate_term(term_key: str, mode: str = "simple") -> str:
    """Translate an aerospace/ML term to simple or advanced view."""
    item = PLAIN_LANGUAGE_MAP.get(term_key)
    if not item:
        for k, v in PLAIN_LANGUAGE_MAP.items():
            if term_key.lower() in (k.lower(), v.get("simple", "").lower(), v.get("advanced", "").lower()):
                return v.get(mode, v.get("simple", term_key))
        return term_key
    return item.get(mode, item.get("simple", term_key))


def get_term_definition(term_kw: str) -> Optional[Dict[str, str]]:
    """Retrieve full glossary definition for a keyword."""
    kw = term_kw.lower()
    for entry in AEROSPACE_GLOSSARY:
        if kw in entry["term"].lower() or kw in entry.get("category", "").lower():
            return entry
    return None


def render_help_tooltip(term_key: str, default_label: Optional[str] = None) -> str:
    """Return formatted label with tooltip based on user mode."""
    mode = st.session_state.get("app_presentation_mode", "simple")
    item = PLAIN_LANGUAGE_MAP.get(term_key)
    if not item:
        return default_label or term_key

    label = item["simple"] if mode == "simple" else item["advanced"]
    return label


def render_contextual_help_modal(term_key: str):
    """Render an expandable 'What does this mean?' card inline."""
    item = PLAIN_LANGUAGE_MAP.get(term_key)
    if not item:
        return

    with st.expander(f"ℹ️ What does '{item['simple']}' mean?"):
        st.write(item["explanation"])
        st.caption(f"**Engineering Definition:** {item['advanced']}")


def render_glossary_view():
    """Render interactive searchable glossary."""
    st.markdown("### 📚 Aerospace & AI Engineering Glossary")
    st.caption("Standardized definitions for jet engine propulsion, prognostics, sensor instrumentation, and machine learning.")

    search_kw = st.text_input("🔍 Search glossary terms...", "", placeholder="e.g. RUL, EGT, Spool, Conformal...")
    
    filtered = AEROSPACE_GLOSSARY
    if search_kw:
        kw = search_kw.lower()
        filtered = [g for g in AEROSPACE_GLOSSARY if kw in g["term"].lower() or kw in g["definition"].lower() or kw in g["category"].lower()]

    for entry in filtered:
        with st.container():
            st.markdown(f"#### {entry['term']} `[{entry['category']}]`")
            st.markdown(f"**Definition:** {entry['definition']}")
            st.caption(f"**Standard Aerospace Units:** `{entry['units']}`")
            st.markdown("---")

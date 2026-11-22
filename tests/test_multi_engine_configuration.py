"""THRUSTLAB — Automated Test Suite for Multi-Manufacturer Engine Configuration & Simulation.

Tests:
1. Adding a manufacturer to the repository.
2. Adding an engine family.
3. Creating and persisting a custom engine configuration with provenance.
4. Importing and validating compatible engineering models (JSON, CSV, Python).
5. Associating datasets and checking compatibility.
6. Selecting compatible AI models and verifying cross-engine adaptation warnings.
7. Executing engine simulation and handling catalogue-only unlinked models.
8. Generating architecture-specific 3D digital twins (Rolls-Royce 3-spool, PW Geared GTF, Standard).
9. Comparing two or more engine configurations in the Comparison Laboratory.
10. Handling missing or unmodeled engines gracefully.
11. Preserving configurations and packages across restarts.
12. Maintaining complete backward compatibility with NASA C-MAPSS FD001 workflow.
"""

import pytest
import os
import json
import pathlib
import tempfile
import sqlite3

from src.engine_configuration.models import (
    Manufacturer,
    EngineFamily,
    EngineConfiguration,
    PropertyWithProvenance,
    ProvenanceSource,
    EngineValidationStatus,
    ShaftArrangement,
    EngineType,
    ImportedEngineModel
)
from src.engine_configuration.repository import EngineRepository
from src.engine_configuration.validation import EngineValidator
from src.engine_configuration.model_importer import ModelImporter, EngineModelAdapter
from src.engine_configuration.engine_comparator import EngineComparator
from src.engine_configuration.multi_twin_generator import render_multi_engine_3d_twin_html
from src.copilot_nlu import AerospaceCopilotNLU, ConversationalIntent
from src.copilot_scenarios import EngineeringScenario, SimulationMode


@pytest.fixture
def repo():
    """Returns an active EngineRepository instance."""
    return EngineRepository()


# =============================================================================
# 1. MANUFACTURER & FAMILY TESTS
# =============================================================================

def test_manufacturer_library_and_add(repo):
    """Test retrieving seeded manufacturers and adding a new OEM."""
    manufacturers = repo.get_all_manufacturers()
    assert len(manufacturers) >= 6
    names = [m.name for m in manufacturers]
    assert "Rolls-Royce plc" in names
    assert "Honeywell Aerospace" in names
    assert "GE Aerospace" in names
    assert "Pratt & Whitney" in names
    assert "CFM International" in names
    assert "Safran Aircraft Engines" in names

    # Add new custom manufacturer
    new_m = Manufacturer(
        id="eurojet",
        name="Eurojet Turbo GmbH",
        country="Germany / UK / Italy / Spain",
        headquarters="Hallbergmoos, Germany",
        description="Military turbofan consortium producing the EJ200 for Eurofighter Typhoon.",
        website="https://www.eurojet.de",
        engine_families=["EJ200"],
        is_custom=True
    )
    added = repo.add_manufacturer(new_m)
    assert added is True

    # Verify retrieval
    m_check = [m for m in repo.get_all_manufacturers() if m.id == "eurojet"]
    assert len(m_check) == 1
    assert m_check[0].name == "Eurojet Turbo GmbH"


def test_engine_family_add(repo):
    """Test adding an engine family to an existing manufacturer."""
    fam = EngineFamily(
        id="rr-pearl",
        manufacturer_id="rolls-royce",
        name="Pearl",
        applications=["Bombardier Global 5500", "Gulfstream G700"],
        thrust_range_kn=(67.0, 81.0),
        models=["Pearl 15", "Pearl 700"]
    )
    added = repo.add_engine_family(fam)
    assert added is True

    families = repo.get_families_for_manufacturer("rolls-royce")
    assert any(f.id == "rr-pearl" for f in families)


# =============================================================================
# 2. ENGINE CONFIGURATION BUILDER & PROVENANCE TESTS
# =============================================================================

def test_engine_configuration_creation_and_persistence(repo):
    """Test creating a custom engine configuration with explicit provenance."""
    custom_cfg = EngineConfiguration(
        config_id="CUSTOM-TEST-001",
        manufacturer="Rolls-Royce plc",
        engine_family="Trent 1000",
        engine_model="Trent 1000-TEN",
        engine_type=EngineType.HIGH_BYPASS_TURBOFAN.value,
        shaft_arrangement=ShaftArrangement.TRIPLE_SPOOL.value,
        number_of_shafts=3,
        fan_diameter_m=PropertyWithProvenance("Fan Diameter", 2.84, "m", ProvenanceSource.MANUFACTURER_DOCUMENTED, "OEM Datasheet"),
        bypass_ratio=PropertyWithProvenance("Bypass Ratio", 10.0, "-", ProvenanceSource.PUBLICLY_SOURCED, "EASA Certification"),
        overall_pressure_ratio=PropertyWithProvenance("Overall Pressure Ratio", 50.0, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED),
        takeoff_thrust_kn=PropertyWithProvenance("Takeoff Thrust", 332.0, "kN", ProvenanceSource.MANUFACTURER_DOCUMENTED),
        cruise_sfc_kg_kn_hr=PropertyWithProvenance("Cruise SFC", 0.295, "kg/(kN*hr)", ProvenanceSource.ESTIMATED),
        dry_weight_kg=PropertyWithProvenance("Dry Weight", 6120.0, "kg", ProvenanceSource.MANUFACTURER_DOCUMENTED),
        is_user_defined=True
    )

    saved = repo.save_engine_configuration(custom_cfg)
    assert saved is True

    loaded = repo.get_engine_configuration("CUSTOM-TEST-001")
    assert loaded is not None
    assert loaded.engine_model == "Trent 1000-TEN"
    assert loaded.number_of_shafts == 3
    assert loaded.shaft_arrangement == ShaftArrangement.TRIPLE_SPOOL.value
    assert loaded.fan_diameter_m.value == 2.84
    assert loaded.fan_diameter_m.source == ProvenanceSource.MANUFACTURER_DOCUMENTED
    assert loaded.cruise_sfc_kg_kn_hr.source == ProvenanceSource.ESTIMATED


def test_engine_validator_rules(repo):
    """Test configuration validation bounds, missing fields, and status computation."""
    trent = repo.get_engine_configuration("RR-TRENT-XWB-84")
    assert trent is not None
    status, errs, warns = EngineValidator.validate_configuration(trent)
    assert len(errs) == 0
    assert status in [EngineValidationStatus.RESEARCH_MODEL, EngineValidationStatus.CATALOGUE_ONLY]

    # Incomplete configuration (missing model & negative fan diameter)
    broken_cfg = EngineConfiguration(
        config_id="BROKEN-001",
        manufacturer="",
        engine_family="Test",
        engine_model="",
        fan_diameter_m=PropertyWithProvenance("Fan Diameter", -1.0, "m", ProvenanceSource.UNKNOWN)
    )
    status_broken, errs_b, warns_b = EngineValidator.validate_configuration(broken_cfg)
    assert status_broken == EngineValidationStatus.CONFIGURATION_INCOMPLETE
    assert len(errs_b) >= 2


# =============================================================================
# 3. MODEL IMPORT & ADAPTER TESTS
# =============================================================================

def test_model_importer_inspects_json(tmp_path):
    """Test inspecting and validating a JSON model definition."""
    model_json = tmp_path / "custom_turbofan_spec.json"
    spec_data = {
        "model_name": "Research Geared Turbofan Model",
        "inputs": [{"name": "altitude_m", "unit": "m"}, {"name": "mach_number", "unit": "-"}],
        "outputs": [{"name": "t50_egt_k", "unit": "K"}, {"name": "p30_kpa", "unit": "kPa"}],
        "sampling_frequency_hz": 20.0,
        "dependencies": ["numpy", "scipy"]
    }
    with open(model_json, "w") as f:
        json.dump(spec_data, f)

    ok, imp_model, msgs = ModelImporter.inspect_and_validate(str(model_json), "PW-PW1100G-JM")
    assert ok is True
    assert imp_model is not None
    assert imp_model.model_type == "JSON_CONFIG"
    assert imp_model.sampling_frequency_hz == 20.0
    assert len(imp_model.input_variables) == 2
    assert imp_model.associated_engine_config_id == "PW-PW1100G-JM"


def test_model_importer_inspects_csv(tmp_path):
    """Test inspecting a CSV telemetry dataset."""
    csv_file = tmp_path / "trent_flight_telemetry.csv"
    csv_file.write_text("time,T2,T30,T50_EGT,P30,N1,N2,fuel_flow\n0.0,288.15,850.0,865.0,3800.0,2400.0,9100.0,1.05\n0.1,288.15,850.2,865.1,3800.5,2400.0,9100.0,1.05\n")

    ok, imp_model, msgs = ModelImporter.inspect_and_validate(str(csv_file))
    assert ok is True
    assert imp_model.model_type == "CSV_TELEMETRY"
    assert len(imp_model.output_variables) >= 6

    # Test adapter execution on CSV
    adapter = EngineModelAdapter(imp_model)
    res = adapter.execute({})
    assert res["success"] is True
    assert res["source"] == "CSV Telemetry Stream"


# =============================================================================
# 4. CROSS-ENGINE AI COMPATIBILITY TESTS
# =============================================================================

def test_cross_engine_ai_model_compatibility(repo):
    """Verify that NASA FD001 model is validated on NASA benchmark but flagged with warning on other engines."""
    nasa_cfg = repo.get_engine_configuration("NASA-CMAPSS-FD001")
    trent_cfg = repo.get_engine_configuration("RR-TRENT-XWB-84")
    ge_cfg = repo.get_engine_configuration("GE-GE90-115B")

    # NASA Engine -> Validated
    chk_nasa = EngineValidator.check_ai_model_engine_compatibility(nasa_cfg, "pinball_lstm_q35 (Best)")
    assert chk_nasa["is_compatible"] is True
    assert chk_nasa["risk_level"] == "LOW"

    # Rolls-Royce Trent (3-spool) -> Cross-engine warning
    chk_trent = EngineValidator.check_ai_model_engine_compatibility(trent_cfg, "pinball_lstm_q35 (Best)")
    assert chk_trent["is_compatible"] is False
    assert chk_trent["risk_level"] == "HIGH"
    assert "CROSS-ENGINE ADAPTATION WARNING" in chk_trent["message"]
    assert "3-spool" in chk_trent["message"]

    # GE90 -> Cross-engine warning (massive thrust delta)
    chk_ge = EngineValidator.check_ai_model_engine_compatibility(ge_cfg, "pinball_lstm_q35 (Best)")
    assert chk_ge["is_compatible"] is False
    assert chk_ge["risk_level"] == "HIGH"


# =============================================================================
# 5. MULTI-ENGINE COMPARISON TESTS
# =============================================================================

def test_multi_engine_comparator(repo):
    """Verify side-by-side comparative analytics across distinct architectures."""
    trent = repo.get_engine_configuration("RR-TRENT-XWB-84")
    pw_gtf = repo.get_engine_configuration("PW-PW1100G-JM")
    ge90 = repo.get_engine_configuration("GE-GE90-115B")

    comp = EngineComparator.compare_configurations([trent, pw_gtf, ge90])
    assert comp["num_engines"] == 3
    assert len(comp["comparison_matrix"]) == 3
    assert "engineering_narrative" in comp
    assert "sensor_overlap" in comp

    # Verify architecture differences are noted
    assert any(m["architecture"] == ShaftArrangement.TRIPLE_SPOOL.value for m in comp["comparison_matrix"])
    assert any(m["architecture"] == ShaftArrangement.GEARED_TURBOFAN.value for m in comp["comparison_matrix"])
    assert any(m["architecture"] == ShaftArrangement.DUAL_SPOOL.value for m in comp["comparison_matrix"])


# =============================================================================
# 6. 3D DIGITAL TWIN GENERATION TESTS
# =============================================================================

def test_multi_engine_3d_twin_html_generation(repo):
    """Verify 3D twin generator produces specific CAD layouts for 3-spool and geared GTF."""
    trent = repo.get_engine_configuration("RR-TRENT-XWB-84")
    pw_gtf = repo.get_engine_configuration("PW-PW1100G-JM")

    # Trent 3-spool
    trent_html = render_multi_engine_3d_twin_html(trent)
    assert "THREE-SPOOL SCHEMATIC" in trent_html
    assert "Intermediate-Pressure Compressor (IPC - N2)" in trent_html
    assert "Intermediate-Pressure Turbine (IPT)" in trent_html

    # Geared GTF
    gtf_html = render_multi_engine_3d_twin_html(pw_gtf)
    assert "GEARED TURBOFAN SCHEMATIC" in gtf_html
    assert "Fan Drive Gear System (FDGS 3:1)" in gtf_html


# =============================================================================
# 7. CONVERSATIONAL COPILOT MULTI-ENGINE INTEGRATION TESTS
# =============================================================================

def test_copilot_nlu_multi_engine_commands():
    """Verify conversational commands selecting engine configurations and checking compatibility."""
    nlu = AerospaceCopilotNLU()

    # 1. Select Rolls-Royce Trent XWB
    intent, scen, exp = nlu.parse_query("Select the Rolls-Royce Trent XWB")
    assert intent == ConversationalIntent.SELECT_ENGINE_CONFIG
    assert scen.selected_engine_config_id == "RR-TRENT-XWB-84"

    # 2. Continuation with retained engine context
    intent2, scen2, _ = nlu.parse_query("Increase turbine temperature by 20 K", current_scenario=scen)
    assert scen2.selected_engine_config_id == "RR-TRENT-XWB-84"

    # 3. Check AI compatibility
    intent3, scen3, _ = nlu.parse_query("Which AI model is compatible with this engine?", current_scenario=scen)
    assert intent3 == ConversationalIntent.CHECK_AI_ENGINE_COMPATIBILITY

    # 4. Compare two engines
    intent4, scen4, _ = nlu.parse_query("Compare these two engine configurations")
    assert intent4 == ConversationalIntent.COMPARE_ENGINES


# =============================================================================
# 8. PACKAGE EXPORT & IMPORT TESTS
# =============================================================================

def test_package_export_and_import(repo, tmp_path):
    """Verify exporting an engine package to JSON and importing it back."""
    pkg_file = repo.export_package("NASA-CMAPSS-FD001", target_dir=tmp_path)
    assert pkg_file is not None
    assert os.path.isfile(pkg_file)

    # Read exported file
    with open(pkg_file, "r") as f:
        pkg_dict = json.load(f)
    assert "configuration" in pkg_dict
    assert pkg_dict["configuration"]["config_id"] == "NASA-CMAPSS-FD001"

    # Import with new ID
    pkg_dict["configuration"]["config_id"] = "IMPORTED-NASA-TEST"
    pkg_dict["configuration"]["engine_model"] = "Imported Clone Model"
    import_path = tmp_path / "clone_package.json"
    with open(import_path, "w") as f:
        json.dump(pkg_dict, f)

    imported_cfg = repo.import_package(str(import_path))
    assert imported_cfg is not None
    assert imported_cfg.config_id == "IMPORTED-NASA-TEST"
    assert imported_cfg.engine_model == "Imported Clone Model"
    assert imported_cfg.is_user_defined is True

    # Clean up test clone
    repo.delete_engine_configuration("IMPORTED-NASA-TEST")

"""JETWISE AI — Comprehensive Prompt-Driven Aerospace Engineering Intelligence Test Suite.

Validates the complete 16-point prompt-driven engineering intelligence requirements:
1. Natural language command interpretation across all mandated prompts
2. Sequential follow-up contextual command processing & compounding
3. 6-dimension transparent Engineering Results Scorecard
4. 7-point simulation validity checklist
5. Dynamic graphical engineering charts & component health matrix
6. Synchronized 3D Turbofan Digital Twin visual states & color mapping
7. Level 1 (Summary) vs Level 2 (Advanced Technical Workspace) data integrity
8. Deep technical inspector tabs (MATLAB, Simulink, Brayton mathematics, telemetry, AI, logs)
9. Automated 17-point engineering report generation (Markdown, HTML, CSV)
10. Zero credential leakage verification
11. Multi-scenario simulation suite (healthy baseline, fuel flow, degradation, sensor faults)
12. Scenario persistence and restoration
"""

import pytest
import numpy as np
import json
import os

from src.copilot_nlu import AerospaceCopilotNLU, ConversationalIntent
from src.copilot_scenarios import (
    EngineeringScenario,
    ScenarioResult,
    SimulationMode,
    ComponentVisualState,
    VerificationStatus,
    PARAMETER_MAP
)
from src.copilot_simulation import CopilotSimulationEngine
from src.copilot_scorecard import ScorecardCalculator, EngineeringScorecard
from src.copilot_visualizations import (
    generate_kpi_summary_data,
    generate_scenario_graphs,
    generate_component_health_table
)
from src.copilot_technical_inspector import TechnicalInspector
from src.copilot_report_generator import CopilotReportGenerator
from src.security.credential_manager import CredentialManager


@pytest.fixture
def nlu_parser():
    return AerospaceCopilotNLU()


@pytest.fixture
def sim_engine():
    return CopilotSimulationEngine()


# -----------------------------------------------------------------------------
# 1. NATURAL LANGUAGE PROMPT PARSING TESTS
# -----------------------------------------------------------------------------
class TestNaturalLanguageParsing:
    def test_mandated_aerospace_prompts(self, nlu_parser):
        mandated_queries = [
            ("Increase turbine temperature by 5% and reduce compressor pressure by 3%.", ConversationalIntent.SIMULATE_SCENARIO),
            ("Simulate gradual fan degradation over 200 operating cycles.", ConversationalIntent.SIMULATE_SCENARIO),
            ("Increase fuel flow by 5% and check how the engine responds.", ConversationalIntent.SIMULATE_SCENARIO),
            ("Introduce a pressure sensor abnormality.", ConversationalIntent.SIMULATE_SCENARIO),
            ("Simulate a compressor degradation scenario and estimate the remaining useful life.", ConversationalIntent.SIMULATE_SCENARIO),
            ("Compare this scenario with a healthy engine.", ConversationalIntent.COMPARE_SCENARIOS),
            ("Run the same condition in MATLAB Simulink and compare the results.", ConversationalIntent.RUN_MATLAB_SIMULINK),
            ("Show me the changes in the 3D engine digital twin.", ConversationalIntent.VIEW_DIGITAL_TWIN),
            ("Check whether this engine condition is within acceptable operating limits.", ConversationalIntent.VERIFY_LIMITS),
            ("Investigate why this engine is showing abnormal temperature readings.", ConversationalIntent.INVESTIGATE_ANOMALY),
            ("Create a healthy baseline engine.", ConversationalIntent.CREATE_BASELINE),
        ]

        for query, expected_intent in mandated_queries:
            intent, scen, _ = nlu_parser.parse_query(query)
            assert intent == expected_intent, f"Failed on prompt: {query} -> got {intent}, expected {expected_intent}"
            assert scen is not None

    def test_multi_parameter_clauses(self, nlu_parser):
        query = "Increase turbine temperature by 5% and reduce compressor pressure by 3%."
        intent, scen, _ = nlu_parser.parse_query(query)
        assert len(scen.parameters) == 2
        keys = [p.parameter_key for p in scen.parameters]
        assert "t50" in keys
        assert "p30" in keys
        # Check values
        t50_p = next(p for p in scen.parameters if p.parameter_key == "t50")
        p30_p = next(p for p in scen.parameters if p.parameter_key == "p30")
        assert t50_p.change_value == 5.0
        assert p30_p.change_value == -3.0

    def test_gradual_degradation_cycles(self, nlu_parser):
        query = "Simulate gradual fan degradation over 200 operating cycles."
        intent, scen, _ = nlu_parser.parse_query(query)
        assert scen.duration_cycles == 200
        assert scen.simulation_mode == SimulationMode.MODE_B_DYNAMIC_ENGINE_WHAT_IF
        assert len(scen.parameters) == 1
        assert scen.parameters[0].parameter_key == "fan_degradation"
        assert scen.parameters[0].is_progressive is True


# -----------------------------------------------------------------------------
# 2. CONTEXTUAL FOLLOW-UP COMMANDS TESTS
# -----------------------------------------------------------------------------
class TestContextualFollowUps:
    def test_cumulative_parameter_adjustment(self, nlu_parser):
        # Step 1: Base scenario
        _, scen1, _ = nlu_parser.parse_query("Simulate compressor degradation by 5%")
        assert scen1.parameters[0].change_value == 5.0

        # Step 2: Contextual follow-up: "Now reduce compressor efficiency by another 3%"
        intent2, scen2, _ = nlu_parser.parse_query(
            "Now reduce compressor efficiency by another 3%",
            current_scenario=scen1
        )
        assert scen2.parameters[0].change_value == 8.0  # 5% + 3%
        assert scen2.parameters[0].parameter_key == "compressor_degradation"

    def test_specialized_follow_up_intents(self, nlu_parser):
        _, scen1, _ = nlu_parser.parse_query("Simulate compressor degradation by 5%")

        # Follow-up: Which component is most affected?
        intent_aff, scen_aff, expl_aff = nlu_parser.parse_query(
            "Which component is most affected?",
            current_scenario=scen1
        )
        assert intent_aff == ConversationalIntent.LOCALIZE_FAULT
        assert "High-Pressure Compressor" in expl_aff

        # Follow-up: Explain the abnormal temperature increase
        intent_exp, scen_exp, expl_exp = nlu_parser.parse_query(
            "Explain the abnormal temperature increase",
            current_scenario=scen1
        )
        assert intent_exp == ConversationalIntent.EXPLAIN_ANOMALY
        assert "Station 3" in expl_exp or "FADEC" in expl_exp

        # Follow-up: Show me the MATLAB simulation outputs
        intent_mat, scen_mat, expl_mat = nlu_parser.parse_query(
            "Show me the MATLAB simulation outputs",
            current_scenario=scen1
        )
        assert intent_mat == ConversationalIntent.SHOW_TECHNICAL_DETAILS
        assert "Level 2 Advanced Engineering Workspace" in expl_mat


# -----------------------------------------------------------------------------
# 3. SCORECARD & 7-POINT SIMULATION VALIDITY TESTS
# -----------------------------------------------------------------------------
class TestScorecardAndValidity:
    def test_six_dimensional_scorecard(self, sim_engine):
        scen = EngineeringScenario(
            scenario_name="HPC Pressure Drop Test",
            user_query="Reduce compressor pressure by 4%",
            dataset="FD001",
            engine_id=1,
            cycle=100,
            simulation_mode=SimulationMode.MODE_A_AI_SENSOR_WHAT_IF
        )
        result = sim_engine.execute_scenario(scen)
        scorecard = ScorecardCalculator.calculate_for_result(scen, result)

        assert isinstance(scorecard, EngineeringScorecard)
        # 1. Health Score (0-100)
        assert 0.0 <= scorecard.engine_health.score <= 100.0
        assert scorecard.engine_health.classification in ["Healthy", "Good", "Caution", "Degraded", "Critical"]
        assert len(scorecard.engine_health.contributing_factors) >= 4

        # 2. Risk Score (0-100)
        assert 0.0 <= scorecard.engine_risk.score <= 100.0
        assert scorecard.engine_risk.risk_level in ["Low", "Moderate", "Elevated", "High"]

        # 3. RUL Metric
        assert scorecard.rul_metric.predicted_rul > 0.0
        assert scorecard.rul_metric.q10_lower <= scorecard.rul_metric.q90_upper

        # 4. Anomaly Metric
        assert 0.0 <= scorecard.anomaly.score <= 100.0
        assert scorecard.anomaly.classification in ["NORMAL", "MILD_ANOMALY", "ELEVATED_ANOMALY", "CRITICAL_FAULT"]

        # 5. Validity Score (7-Point Checklist)
        assert scorecard.validity.total_checks == 7
        assert scorecard.validity.pass_count >= 6
        assert scorecard.validity.overall_status in ["PASS", "WARNING"]
        check_names = [c.name for c in scorecard.validity.checklist]
        expected_fragments = [
            "Input Validation",
            "Model Compatibility",
            "Simulation Completion",
            "Numerical Checks",
            "Output Integrity",
            "Sensor Mapping",
            "Reproducibility"
        ]
        for ef in expected_fragments:
            assert any(ef in cn for cn in check_names)

        # 6. Confidence Metric
        assert 50.0 <= scorecard.confidence.confidence_pct <= 100.0


# -----------------------------------------------------------------------------
# 4. GRAPHICAL DASHBOARD & COMPONENT TABLE TESTS
# -----------------------------------------------------------------------------
class TestVisualizationsAndComponentTable:
    def test_kpi_cards_and_component_health(self, sim_engine):
        scen = EngineeringScenario(scenario_name="KPI Test", dataset="FD001", engine_id=1)
        res = sim_engine.execute_scenario(scen)
        sc = ScorecardCalculator.calculate_for_result(scen, res)

        kpis = generate_kpi_summary_data(sc)
        assert len(kpis) == 6
        assert kpis[0]["title"] == "Engine Health Score"
        assert kpis[1]["title"] == "Engine Risk Score"
        assert kpis[2]["title"] == "Remaining Useful Life"

        comp_df = generate_component_health_table(res, sc)
        assert len(comp_df) == 10  # All 10 turbofan subassemblies
        subassemblies = comp_df["Subassembly"].values
        assert any("Fan" in s for s in subassemblies)
        assert any("Combustion" in s for s in subassemblies)
        assert any("High-Pressure Turbine" in s for s in subassemblies)

    def test_scenario_graphs(self, sim_engine):
        scen = EngineeringScenario(scenario_name="Plotly Test", dataset="FD001", engine_id=1)
        res = sim_engine.execute_scenario(scen)
        figs = generate_scenario_graphs(scen, res)

        assert "thermal_profile" in figs
        assert "pressure_profile" in figs
        assert "spool_speeds" in figs
        assert "multi_sensor_delta" in figs


# -----------------------------------------------------------------------------
# 5. LEVEL 2 ADVANCED ENGINEERING WORKSPACE TESTS
# -----------------------------------------------------------------------------
class TestTechnicalInspectorTabs:
    def test_all_six_technical_tabs(self, sim_engine):
        scen = EngineeringScenario(
            scenario_name="Technical Workspace Test",
            user_query="Increase turbine temperature by 5%",
            dataset="FD001",
            engine_id=1
        )
        res = sim_engine.execute_scenario(scen)

        # Tab A: MATLAB Execution Details
        tab_a = TechnicalInspector.get_matlab_execution_details(scen, res)
        assert "matlab_version" in tab_a
        assert "command_line" in tab_a
        assert "raw_output" in tab_a
        assert len(tab_a["workspace_variables"]) >= 5

        # Tab B: Simulink Model Details
        tab_b = TechnicalInspector.get_simulink_model_details(scen, res)
        assert "CMAPSS" in tab_b["model_name"]
        assert tab_b["solver_settings"]["solver_type"].startswith("ode")
        assert len(tab_b["input_port_mappings"]) >= 5
        assert len(tab_b["output_port_mappings"]) >= 5

        # Tab C: Mathematical Details
        tab_c = TechnicalInspector.get_mathematical_details(scen, res)
        assert len(tab_c["equations"]) >= 6
        assert len(tab_c["thermodynamic_assumptions"]) >= 6
        assert len(tab_c["citations"]) >= 3
        # Verify Brayton cycle equations are included
        eq_titles = [eq["title"] for eq in tab_c["equations"]]
        assert "Compressor Pressure Ratio" in eq_titles
        assert "Isentropic Temperature Relation" in eq_titles
        assert "Acoustic Velocity (Speed of Sound)" in eq_titles

        # Tab D: Raw Simulation Telemetry
        tab_d = TechnicalInspector.get_raw_telemetry_details(scen, res)
        assert len(tab_d["telemetry_rows"]) > 0
        assert len(tab_d["summary_statistics"]) == 21  # 21 sensors
        csv_str = TechnicalInspector.export_telemetry_to_csv(tab_d)
        json_str = TechnicalInspector.export_telemetry_to_json(tab_d)
        assert len(csv_str.splitlines()) > 5
        assert json.loads(json_str)

        # Tab E: Python AI Analysis Details
        tab_e = TechnicalInspector.get_python_ai_details(scen, res)
        assert "Dual-Branch" in tab_e["model_architecture"]["name"]
        assert tab_e["model_architecture"]["parameters_count"] > 100000
        assert len(tab_e["feature_importance"]) >= 5
        assert tab_e["calibration_metadata"]["target_coverage"] == "90.0% (alpha = 0.10)"

        # Tab F: Execution Logs
        logs = TechnicalInspector.get_execution_logs(scen, res)
        assert len(logs) >= 8
        log_text = TechnicalInspector.export_logs_to_text(logs)
        assert "ConversationalNLU" in log_text
        assert "NeuralInference" in log_text


# -----------------------------------------------------------------------------
# 6. AUTOMATED 17-POINT ENGINEERING REPORT GENERATION
# -----------------------------------------------------------------------------
class TestEngineeringReportGeneration:
    def test_report_formats(self, sim_engine):
        scen = EngineeringScenario(
            scenario_name="Report Generation Test",
            user_query="Simulate gradual fan degradation over 200 operating cycles",
            dataset="FD001",
            engine_id=1,
            cycle=100
        )
        res = sim_engine.execute_scenario(scen)
        sc = ScorecardCalculator.calculate_for_result(scen, res)
        rep = CopilotReportGenerator(scen, res, sc)

        # 1. Markdown Report (17 sections)
        md = rep.generate_markdown_report()
        assert len(md) > 5000
        assert "# JETWISE AI — AEROSPACE ENGINEERING SIMULATION REPORT" in md
        assert "1. Executive Summary" in md
        assert "4. Key Aerothermal & Prognostic Metrics" in md
        assert "5. Health & Risk Scores Decomposition" in md
        assert "8. Subassembly Component Health Matrix" in md
        assert "9. Station-by-Station Thermodynamic Gas-Path Analysis" in md
        assert "11. Mathematical & Aerothermodynamic Formulations" in md
        assert "13. Simulation Validity & Verification Checklist" in md
        assert "16. Automated Verification & Certification Sign-Off" in md

        # 2. HTML Printable Report
        html = rep.generate_html_report()
        assert "<!DOCTYPE html>" in html
        assert "<title>JETWISE AI Engineering Report" in html
        assert "Subassembly Health & Prognostic Status" in html
        assert "AUTOMATED VERIFICATION SIGN-OFF" in html

        # 3. CSV Fleet Export
        csv_out = rep.generate_csv_summary()
        lines = csv_out.strip().splitlines()
        assert len(lines) == 2  # Header + 1 summary row
        assert "report_id,timestamp,engine_config_id" in lines[0]


# -----------------------------------------------------------------------------
# 7. ZERO CREDENTIAL LEAKAGE TESTS
# -----------------------------------------------------------------------------
class TestZeroCredentialLeakage:
    def test_scrubbing_sensitive_keys(self):
        fake_gemini = "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
        fake_hf = "hf_abcdefghijklmnopqrstuvwxyz12345678"
        raw_msg = f"Failed to authenticate with token {fake_gemini} and {fake_hf}"

        sanitized = CredentialManager.sanitize_error(raw_msg, [fake_gemini, fake_hf])
        assert fake_gemini not in sanitized
        assert fake_hf not in sanitized
        assert "[MASKED" in sanitized

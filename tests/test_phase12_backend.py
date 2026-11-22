"""Unit and Integration Tests for Phase 12 Next-Generation Aerospace Research Modules."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import pathlib

from src.mission_prognostics import MissionAwarePrognostics
from src.component_health import ComponentHealthEngine
from src.multimodal_fusion import MultiModalDataFusionEngine
from src.reliability_analysis import ReliabilityAnalysisEngine
from src.active_learning import ActiveLearningEngine
from src.knowledge_graph import EngineeringKnowledgeGraph
from src.edge_ai import EdgeOptimizationEngine
from src.decision_intelligence import DecisionIntelligenceEngine
from src.database_migrations import apply_phase11_12_migrations, seed_phase11_12_demo_records


@pytest.fixture(scope="module")
def setup_test_db(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "test_phase12.db"
    apply_phase11_12_migrations(db_file)
    seed_phase11_12_demo_records(db_file)
    return db_file


def test_mission_prognostics(setup_test_db):
    """Verify mission profile wear modulation and degradation trajectory synthesis."""
    engine = MissionAwarePrognostics(db_path=setup_test_db)

    res = engine.evaluate_mission_impact(
        engine_id=1,
        dataset_subset="FD001",
        current_cycle=100,
        baseline_predicted_rul=50.0,
        selected_profile="short_haul"
    )

    assert res["wear_acceleration_factor"] > 1.0
    assert res["projected_rul"] < res["baseline_rul"]
    assert res["rul_delta_cycles"] < 0.0
    assert len(res["mission_trajectory"]) > 0


def test_component_health(setup_test_db):
    """Verify hierarchical subsystem health index calculations."""
    comp = ComponentHealthEngine(db_path=setup_test_db)

    z_scores = {
        "sensor_2": 0.5,
        "sensor_3": 3.8,  # High HPC outlet temperature
        "sensor_7": 2.9,  # High HPC outlet pressure
        "sensor_9": 3.2   # Elevated EGT
    }

    eval_res = comp.evaluate_subsystem_health(
        engine_id=1,
        dataset_subset="FD001",
        current_cycle=150,
        sensor_zscores=z_scores,
        predicted_rul=15.0
    )

    assert len(eval_res["subsystems"]) == 7
    hpc_sub = next(s for s in eval_res["subsystems"] if s["subsystem_key"] == "HPC")
    assert hpc_sub["health_index"] < 0.60
    assert hpc_sub["max_sensor_zscore"] > 3.0


def test_multimodal_fusion(setup_test_db):
    """Verify multi-modal data fusion with complete and missing modalities."""
    fusion = MultiModalDataFusionEngine(db_path=setup_test_db)

    vib_data = {"overall_rms_ips": 0.45, "harmonic_1x_ips": 0.15, "blade_pass_freq_ips": 0.09}
    ac_data = {"energy_counts": 280.0}
    maint_data = {"findings": "Minor thermal barrier coating spallation", "severity_score": 0.6}

    res = fusion.fuse_multimodal_observations(
        engine_id=1,
        dataset_subset="FD001",
        current_cycle=120,
        telemetry_anomaly_score=0.75,
        vibration_data=vib_data,
        acoustic_data=ac_data,
        maintenance_log_entry=maint_data
    )

    assert res["active_modalities_count"] == 4
    assert res["composite_multimodal_anomaly_score"] > 0.4
    assert res["overall_data_quality_index"] > 0.85


def test_reliability_analysis(setup_test_db):
    """Verify 2-parameter Weibull reliability estimation and hazard rates."""
    rel = ReliabilityAnalysisEngine(db_path=setup_test_db)

    np.random.seed(42)
    durations = np.random.weibull(2.5, 50) * 200.0
    events = np.ones(50)

    fit_res = rel.fit_weibull_model(durations, events, dataset_name="FD001")
    assert fit_res["shape_beta"] > 1.0
    assert fit_res["scale_eta"] > 50.0
    assert fit_res["mtbf_cycles"] > 0.0

    curve_res = rel.compute_survival_curve(beta=fit_res["shape_beta"], eta=fit_res["scale_eta"])
    assert len(curve_res["curve"]) > 10
    assert curve_res["curve"][0]["survival_probability"] == 1.0

    fail_prob = rel.compute_failure_within_horizon(
        current_cycle=150, delta_cycles=30,
        beta=fit_res["shape_beta"], eta=fit_res["scale_eta"]
    )
    assert 0.0 <= fail_prob <= 1.0


def test_active_learning(setup_test_db):
    """Verify active learning acquisition scoring and priority ranking."""
    al = ActiveLearningEngine(db_path=setup_test_db)

    mock_fleet = [
        {"engine_id": 1, "uncertainty_std": 14.5, "ood_score": 0.8, "anomaly_score": 0.9, "predicted_rul": 15.0},
        {"engine_id": 2, "uncertainty_std": 2.1, "ood_score": 0.1, "anomaly_score": 0.1, "predicted_rul": 90.0},
        {"engine_id": 3, "uncertainty_std": 8.0, "ood_score": 0.4, "anomaly_score": 0.5, "predicted_rul": 40.0}
    ]

    ranked = al.rank_inspection_candidates(mock_fleet, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["engine_id"] == 1
    assert ranked[0]["priority_rank"] == 1


def test_knowledge_graph(setup_test_db):
    """Verify ontology queries and network graph layout coordinates."""
    kg = EngineeringKnowledgeGraph(db_path=setup_test_db)

    triples = kg.get_all_triples()
    assert len(triples) >= 10

    subgraph = kg.query_subgraph("HPC")
    assert len(subgraph) >= 1

    layout = kg.build_network_layout(triples)
    assert len(layout["node_list"]) > 0
    assert len(layout["node_x"]) == len(layout["node_list"])
    assert len(layout["edge_x"]) > 0


def test_edge_ai_optimization(setup_test_db):
    """Verify FP32 vs dynamic INT8 quantization benchmarking on CPU."""
    edge = EdgeOptimizationEngine(db_path=setup_test_db)

    # Simple 2-layer LSTM module for benchmarking
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(17, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = ToyModel()
    benchmarks = edge.benchmark_model_architectures(model, input_dim=17, seq_len=30, num_runs=5)

    assert len(benchmarks) >= 2
    fp32_bench = next(b for b in benchmarks if b["format"] == "PyTorch FP32")
    assert fp32_bench["single_sample_latency_ms"] > 0.0
    assert fp32_bench["meets_avionics_realtime_sla"] is True


def test_decision_intelligence(setup_test_db):
    """Verify action-conditioned counterfactual decision comparison."""
    decision_engine = DecisionIntelligenceEngine(db_path=setup_test_db)

    matrix = decision_engine.evaluate_decision_matrix(
        engine_id=1,
        dataset_subset="FD001",
        current_cycle=160,
        current_predicted_rul=14.0,
        current_mpi=85.0
    )

    assert len(matrix["action_evaluations"]) == 4
    assert matrix["ai_recommended_action"] != ""

    dec_id = decision_engine.record_human_decision(
        engine_id=1,
        dataset_subset="FD001",
        cycle=160,
        action_chosen="minor_servicing",
        projected_rul=42.0,
        rationale="Water wash and borescope selected to clear compressor fouling.",
        engineer_id="Chief Engineer T. Wilson"
    )
    assert dec_id > 0

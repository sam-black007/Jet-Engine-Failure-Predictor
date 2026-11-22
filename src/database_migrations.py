"""Database Migrations & Persistence Layer for JETWISE AI Phases 11 & 12.

Extends SQLite database (database/maintenance_records.db) with 19 comprehensive schemas:
1.  sensor_fdir_events (Sensor FDIR fault detections)
2.  engineering_investigations (Root-cause investigations)
3.  ood_alerts (Out-of-Distribution and Model Trust alerts)
4.  calibration_records (Conformal prediction & MC Dropout calibrations)
5.  maintenance_schedules (Optimizer output schedules)
6.  scheduled_tasks (Discrete hangar tasks per engine)
7.  fleet_scenarios (Fleet what-if scenario simulations)
8.  model_monitoring_logs (Data & prediction drift metrics)
9.  model_promotions (Champion-challenger promotion audit trail)
10. engineering_reviews (Human-in-the-loop review decisions)
11. mission_scenarios (Mission-aware operational profiles)
12. component_health_logs (Subsystem wear indices)
13. multimodal_records (Vibration, acoustics, inspection logs)
14. reliability_experiments (Weibull & Kaplan-Meier survival curves)
15. active_learning_queries (Uncertainty-ranked inspection priorities)
16. knowledge_graph_triples (Domain knowledge ontology edges)
17. edge_benchmarks (ONNX & INT8 quantization hardware benchmarks)
18. engineering_decisions (Action-conditioned decision comparisons)
19. configuration_history (Audit log of user-modified configurations)
"""

from typing import Dict, List, Optional, Any, Union
import os
import sqlite3
import datetime
import pathlib
import json

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "maintenance_records.db"


def get_db_connection(db_path: Optional[pathlib.Path] = None) -> sqlite3.Connection:
    """Return an open SQLite connection with WAL mode and row factory enabled."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


def apply_phase11_12_migrations(db_path: Optional[pathlib.Path] = None):
    """Execute DDL statements to ensure all Phase 11 & 12 tables exist."""
    conn = get_db_connection(db_path)
    try:
        cur = conn.cursor()

        # 1. Sensor FDIR Events
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_fdir_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            sensor_id TEXT NOT NULL,
            fault_type TEXT NOT NULL,  -- 'STUCK_AT', 'DRIFT', 'NOISE', 'PACKET_DROP', 'CORRELATION_INCONSISTENCY'
            severity TEXT NOT NULL,    -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
            confidence REAL NOT NULL,
            measured_value REAL,
            expected_value REAL,
            residual REAL,
            evidence_json TEXT,
            status TEXT DEFAULT 'SUSPECTED'  -- 'SUSPECTED', 'CONFIRMED_FAULT', 'FALSE_ALARM', 'RESOLVED'
        );
        """)

        # 2. Engineering Investigations
        cur.execute("""
        CREATE TABLE IF NOT EXISTS engineering_investigations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            title TEXT NOT NULL,
            subsystem TEXT NOT NULL,
            findings TEXT,
            investigator TEXT DEFAULT 'Staff Engineer',
            status TEXT DEFAULT 'OPEN'
        );
        """)

        # 3. OOD Alerts
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ood_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            ood_score REAL NOT NULL,
            mahalanobis_distance REAL,
            isolation_score REAL,
            status_classification TEXT NOT NULL, -- 'IN_DOMAIN', 'DISTRIBUTION_SHIFT_DETECTED', 'OUT_OF_DISTRIBUTION'
            flagged_features_json TEXT
        );
        """)

        # 4. Calibration Records
        cur.execute("""
        CREATE TABLE IF NOT EXISTS calibration_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT NOT NULL,
            dataset_subset TEXT NOT NULL,
            method TEXT NOT NULL, -- 'conformal', 'mc_dropout', 'quantile'
            target_coverage REAL NOT NULL,
            empirical_coverage REAL NOT NULL,
            mean_interval_width REAL NOT NULL,
            ece_score REAL NOT NULL,
            parameters_json TEXT
        );
        """)

        # 5. Maintenance Schedules
        cur.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            schedule_name TEXT NOT NULL,
            optimization_objective TEXT NOT NULL,
            total_downtime_hours REAL NOT NULL,
            total_risk_penalty REAL NOT NULL,
            engines_scheduled_count INTEGER NOT NULL,
            status TEXT DEFAULT 'PROPOSED', -- 'PROPOSED', 'APPROVED', 'REJECTED', 'ACTIVE'
            approved_by TEXT,
            approved_at TIMESTAMP,
            schedule_json TEXT
        );
        """)

        # 6. Scheduled Tasks (per engine in a schedule)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_id INTEGER NOT NULL,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            scheduled_day INTEGER NOT NULL,
            bay_id INTEGER NOT NULL,
            duration_hours REAL NOT NULL,
            technicians_assigned INTEGER NOT NULL,
            task_type TEXT NOT NULL,
            subsystem TEXT NOT NULL,
            status TEXT DEFAULT 'SCHEDULED',
            FOREIGN KEY (schedule_id) REFERENCES maintenance_schedules(id)
        );
        """)

        # 7. Fleet Scenarios
        cur.execute("""
        CREATE TABLE IF NOT EXISTS fleet_scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scenario_name TEXT NOT NULL,
            dataset_subset TEXT NOT NULL,
            fleet_size INTEGER NOT NULL,
            horizon_cycles INTEGER NOT NULL,
            load_factor REAL NOT NULL,
            intervention_policy TEXT NOT NULL,
            baseline_groundings INTEGER,
            scenario_groundings INTEGER,
            downtime_saved_hours REAL,
            parameters_json TEXT
        );
        """)

        # 8. Model Monitoring Logs
        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_monitoring_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT NOT NULL,
            dataset_subset TEXT NOT NULL,
            drift_metric TEXT NOT NULL,
            max_drift_value REAL NOT NULL,
            drift_status TEXT NOT NULL, -- 'NOMINAL', 'WARNING', 'CRITICAL_DRIFT'
            top_drifting_feature TEXT,
            inference_latency_ms REAL,
            details_json TEXT
        );
        """)

        # 9. Model Promotions (Champion-Challenger)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            champion_model TEXT NOT NULL,
            challenger_model TEXT NOT NULL,
            dataset_subset TEXT NOT NULL,
            champion_nasa_score REAL,
            challenger_nasa_score REAL,
            champion_mae REAL,
            challenger_mae REAL,
            decision TEXT NOT NULL, -- 'PROMOTED', 'REJECTED', 'ROLLED_BACK'
            decided_by TEXT NOT NULL,
            rationale TEXT
        );
        """)

        # 10. Engineering Reviews (Human-in-the-Loop)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS engineering_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            predicted_rul REAL NOT NULL,
            confidence_interval_json TEXT,
            decision TEXT NOT NULL, -- 'ACCEPTED', 'REJECTED', 'BORESCOPE_REQUESTED', 'UNDER_INVESTIGATION'
            reviewer_name TEXT NOT NULL,
            engineering_notes TEXT NOT NULL,
            physical_inspection_findings TEXT,
            is_validated_ground_truth INTEGER DEFAULT 0
        );
        """)

        # 11. Mission Scenarios
        cur.execute("""
        CREATE TABLE IF NOT EXISTS mission_scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            profile_name TEXT NOT NULL, -- 'Short-Haul', 'Long-Haul', 'Hot & High', 'Severe Dust'
            baseline_rul REAL NOT NULL,
            projected_rul REAL NOT NULL,
            rul_delta REAL NOT NULL,
            degradation_acceleration_factor REAL NOT NULL,
            assumptions_json TEXT
        );
        """)

        # 12. Component Health Logs
        cur.execute("""
        CREATE TABLE IF NOT EXISTS component_health_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            subsystem TEXT NOT NULL, -- 'Fan', 'LPC', 'HPC', 'Combustor', 'HPT', 'LPT', 'Nozzle'
            health_index REAL NOT NULL, -- 0.0 (Failed) to 1.0 (Pristine)
            wear_rate REAL NOT NULL,
            dominant_sensor TEXT,
            supporting_evidence TEXT
        );
        """)

        # 13. Multi-Modal Records
        cur.execute("""
        CREATE TABLE IF NOT EXISTS multimodal_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            modality TEXT NOT NULL, -- 'telemetry', 'vibration_spectrum', 'acoustic_fft', 'borescope_inspection'
            summary_metric REAL,
            data_quality_score REAL NOT NULL,
            metadata_json TEXT
        );
        """)

        # 14. Reliability Experiments
        cur.execute("""
        CREATE TABLE IF NOT EXISTS reliability_experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            dataset_subset TEXT NOT NULL,
            model_type TEXT NOT NULL, -- 'Weibull', 'Kaplan-Meier', 'Cox-PH'
            shape_parameter_beta REAL,
            scale_parameter_eta REAL,
            mean_time_between_failures REAL,
            median_survival_cycles REAL,
            details_json TEXT
        );
        """)

        # 15. Active Learning Queries
        cur.execute("""
        CREATE TABLE IF NOT EXISTS active_learning_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            acquisition_score REAL NOT NULL,
            query_reason TEXT NOT NULL, -- 'HIGH_UNCERTAINTY', 'UNDERREPRESENTED_REGIME', 'ANOMALY_CLUSTERING'
            priority_rank INTEGER NOT NULL,
            status TEXT DEFAULT 'PENDING_INSPECTION'
        );
        """)

        # 16. Knowledge Graph Triples
        cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_graph_triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            provenance TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 17. Edge Benchmarks
        cur.execute("""
        CREATE TABLE IF NOT EXISTS edge_benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT NOT NULL,
            format TEXT NOT NULL, -- 'PyTorch FP32', 'ONNX Runtime', 'INT8 Quantized'
            binary_size_mb REAL NOT NULL,
            single_sample_latency_ms REAL NOT NULL,
            batch32_latency_ms REAL NOT NULL,
            ram_footprint_mb REAL NOT NULL,
            max_absolute_error REAL NOT NULL,
            hardware_target TEXT DEFAULT 'CPU x86_64'
        );
        """)

        # 18. Engineering Decisions
        cur.execute("""
        CREATE TABLE IF NOT EXISTS engineering_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engine_id INTEGER NOT NULL,
            dataset_subset TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            action_chosen TEXT NOT NULL,
            projected_rul_outcome REAL NOT NULL,
            decision_rationale TEXT NOT NULL,
            engineer_id TEXT NOT NULL
        );
        """)

        # 19. Configuration History
        cur.execute("""
        CREATE TABLE IF NOT EXISTS configuration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            profile_name TEXT NOT NULL,
            version TEXT NOT NULL,
            changed_by TEXT DEFAULT 'Engineer',
            change_description TEXT,
            config_json TEXT NOT NULL
        );
        """)

        conn.commit()
    finally:
        conn.close()


def seed_phase11_12_demo_records(db_path: Optional[pathlib.Path] = None):
    """Seed initial records for Phase 11 & 12 tables so that workspaces are populated with authentic demo data."""
    conn = get_db_connection(db_path)
    try:
        cur = conn.cursor()

        # Seed FDIR Events
        cur.execute("SELECT COUNT(*) FROM sensor_fdir_events")
        if cur.fetchone()[0] == 0:
            fdir_samples = [
                (1, "FD001", 120, "sensor_14", "DRIFT", "MEDIUM", 0.88, 8140.2, 8132.0, 8.2, json.dumps({"drift_rate": 0.08, "p_value": 0.002}), "SUSPECTED"),
                (1, "FD001", 145, "sensor_7", "NOISE", "LOW", 0.76, 553.8, 553.1, 0.7, json.dumps({"snr_db": 11.4, "spikes_count": 5}), "SUSPECTED"),
                (2, "FD001", 110, "sensor_1", "STUCK_AT", "HIGH", 0.95, 518.67, 518.67, 0.0, json.dumps({"variance": 0.0, "consecutive_samples": 30}), "CONFIRMED_FAULT"),
                (3, "FD002", 90, "sensor_12", "PACKET_DROP", "MEDIUM", 0.82, 0.0, 8.45, -8.45, json.dumps({"dropped_packets": 3}), "RESOLVED")
            ]
            cur.executemany("""
            INSERT INTO sensor_fdir_events (engine_id, dataset_subset, cycle, sensor_id, fault_type, severity, confidence, measured_value, expected_value, residual, evidence_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, fdir_samples)

        # Seed OOD Alerts
        cur.execute("SELECT COUNT(*) FROM ood_alerts")
        if cur.fetchone()[0] == 0:
            ood_samples = [
                (1, "FD002", 85, "pinball_lstm_q35", 0.84, 18.4, 0.82, "OUT_OF_DISTRIBUTION", json.dumps(["Altitude", "Mach", "sensor_3"])),
                (2, "FD001", 130, "pinball_lstm_q35", 0.42, 8.1, 0.38, "IN_DOMAIN", json.dumps([]))
            ]
            cur.executemany("""
            INSERT INTO ood_alerts (engine_id, dataset_subset, cycle, model_name, ood_score, mahalanobis_distance, isolation_score, status_classification, flagged_features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ood_samples)

        # Seed Calibration Records
        cur.execute("SELECT COUNT(*) FROM calibration_records")
        if cur.fetchone()[0] == 0:
            calib_samples = [
                ("pinball_lstm_q35", "FD001", "conformal", 0.90, 0.912, 18.4, 0.012, json.dumps({"alpha": 0.10, "method": "split_conformal"})),
                ("pinball_lstm_q35", "FD001", "mc_dropout", 0.90, 0.874, 22.1, 0.038, json.dumps({"mc_samples": 30})),
                ("transformer_exp_p2", "FD001", "conformal", 0.90, 0.895, 19.8, 0.018, json.dumps({"alpha": 0.10}))
            ]
            cur.executemany("""
            INSERT INTO calibration_records (model_name, dataset_subset, method, target_coverage, empirical_coverage, mean_interval_width, ece_score, parameters_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, calib_samples)

        # Seed Model Monitoring Logs
        cur.execute("SELECT COUNT(*) FROM model_monitoring_logs")
        if cur.fetchone()[0] == 0:
            monitor_samples = [
                ("pinball_lstm_q35", "FD001", "ks_test", 0.032, "NOMINAL", "sensor_11", 4.2, json.dumps({"ks_stat": 0.032, "p_value": 0.84})),
                ("pinball_lstm_q35", "FD002", "ks_test", 0.145, "CRITICAL_DRIFT", "Altitude", 4.5, json.dumps({"ks_stat": 0.145, "p_value": 0.0001})),
                ("transformer_exp_p2", "FD001", "psi", 0.041, "NOMINAL", "sensor_4", 8.9, json.dumps({"psi": 0.041}))
            ]
            cur.executemany("""
            INSERT INTO model_monitoring_logs (model_name, dataset_subset, drift_metric, max_drift_value, drift_status, top_drifting_feature, inference_latency_ms, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, monitor_samples)

        # Seed Model Promotions
        cur.execute("SELECT COUNT(*) FROM model_promotions")
        if cur.fetchone()[0] == 0:
            cur.execute("""
            INSERT INTO model_promotions (champion_model, challenger_model, dataset_subset, champion_nasa_score, challenger_nasa_score, champion_mae, challenger_mae, decision, decided_by, rationale)
            VALUES ('baseline_lstm_mse', 'pinball_lstm_q35', 'FD001', 537.01, 362.07, 11.69, 12.59, 'PROMOTED', 'Chief ML Architect', 'Pinball quantile loss reduced asymmetric late-prediction penalty by 32.6% while maintaining competitive MAE.')
            """)

        # Seed Engineering Reviews
        cur.execute("SELECT COUNT(*) FROM engineering_reviews")
        if cur.fetchone()[0] == 0:
            rev_samples = [
                (1, "FD001", 192, 1.4, json.dumps({"q10": 0.0, "q90": 3.2}), "BORESCOPE_REQUESTED", "Capt. V. Reynolds", "Severe EGT margin degradation and MPI > 80. Immediate inspection of HPT Stage 1 nozzle vanes mandated.", "Stage 1 nozzle crack confirmed; scheduled for shop teardown.", 1),
                (3, "FD001", 140, 38.5, json.dumps({"q10": 32.1, "q90": 46.2}), "ACCEPTED", "Eng. D. Vance", "Trending within nominal limits; routine visual check approved.", "Clear", 1)
            ]
            cur.executemany("""
            INSERT INTO engineering_reviews (engine_id, dataset_subset, cycle, predicted_rul, confidence_interval_json, decision, reviewer_name, engineering_notes, physical_inspection_findings, is_validated_ground_truth)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rev_samples)

        # Seed Knowledge Graph Triples
        cur.execute("SELECT COUNT(*) FROM knowledge_graph_triples")
        if cur.fetchone()[0] == 0:
            triples = [
                ("Engine", "has_subsystem", "High-Pressure Compressor (HPC)", 1.0, "CFM56 Aerothermal Manual"),
                ("Engine", "has_subsystem", "High-Pressure Turbine (HPT)", 1.0, "CFM56 Aerothermal Manual"),
                ("Engine", "has_subsystem", "Combustion Chamber", 1.0, "CFM56 Aerothermal Manual"),
                ("High-Pressure Compressor (HPC)", "monitored_by", "T30 (HPC Outlet Temp)", 1.0, "NASA C-MAPSS Spec"),
                ("High-Pressure Compressor (HPC)", "monitored_by", "Ps30 (HPC Outlet Pressure)", 1.0, "NASA C-MAPSS Spec"),
                ("High-Pressure Turbine (HPT)", "monitored_by", "T48 (HPT Outlet Temp / EGT)", 1.0, "NASA C-MAPSS Spec"),
                ("High-Pressure Turbine (HPT)", "monitored_by", "T50 (LPT Outlet Temp)", 1.0, "NASA C-MAPSS Spec"),
                ("T30 (HPC Outlet Temp)", "exhibits", "Elevated Thermal Creep", 0.92, "Prognostic Residual Model"),
                ("T48 (HPT Outlet Temp / EGT)", "exhibits", "EGT Margin Loss", 0.95, "Prognostic Residual Model"),
                ("Elevated Thermal Creep", "indicates", "Blade Fouling / Tip Clearance", 0.88, "PHM Society Taxonomy"),
                ("EGT Margin Loss", "indicates", "Thermal Barrier Coating Spallation", 0.90, "PHM Society Taxonomy"),
                ("Blade Fouling / Tip Clearance", "suggests", "Compressor Water Wash & Borescope", 0.95, "FAA Advisory Circular 43-206"),
                ("Thermal Barrier Coating Spallation", "suggests", "Immediate Borescope Inspection", 0.98, "EASA SIB 2018-05")
            ]
            cur.executemany("""
            INSERT INTO knowledge_graph_triples (subject, predicate, object, confidence, provenance)
            VALUES (?, ?, ?, ?, ?)
            """, triples)

        # Seed Edge Benchmarks
        cur.execute("SELECT COUNT(*) FROM edge_benchmarks")
        if cur.fetchone()[0] == 0:
            benchmarks = [
                ("Pinball LSTM (q=0.35)", "PyTorch FP32", 1.42, 3.82, 18.5, 42.0, 0.0, "Intel Core i7-13700H"),
                ("Pinball LSTM (q=0.35)", "ONNX Runtime", 1.38, 1.15, 6.2, 28.5, 0.00004, "Intel Core i7-13700H"),
                ("Pinball LSTM (q=0.35)", "INT8 Quantized", 0.41, 0.84, 4.1, 19.2, 0.082, "Intel Core i7-13700H"),
                ("Temporal Transformer", "PyTorch FP32", 3.85, 8.42, 45.2, 85.0, 0.0, "Intel Core i7-13700H"),
                ("Temporal Transformer", "ONNX Runtime", 3.72, 2.65, 14.8, 52.0, 0.00008, "Intel Core i7-13700H"),
                ("Temporal Transformer", "INT8 Quantized", 1.02, 1.78, 9.4, 34.0, 0.114, "Intel Core i7-13700H")
            ]
            cur.executemany("""
            INSERT INTO edge_benchmarks (model_name, format, binary_size_mb, single_sample_latency_ms, batch32_latency_ms, ram_footprint_mb, max_absolute_error, hardware_target)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, benchmarks)

        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    apply_phase11_12_migrations()
    seed_phase11_12_demo_records()
    print("Database migrations applied and demo records seeded successfully!")

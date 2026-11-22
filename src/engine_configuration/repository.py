"""THRUSTLAB — Engine Configuration SQLite Persistence & Catalogue Repository.

Stores, manages, queries, and exports multi-manufacturer aircraft engine configurations,
provenance metadata, imported models, and cross-engine comparison sessions.
"""

from typing import Dict, List, Any, Optional
import sqlite3
import json
import os
import pathlib
import datetime

from src.engine_configuration.models import (
    Manufacturer,
    EngineFamily,
    EngineConfiguration,
    ImportedEngineModel,
    PropertyWithProvenance,
    ProvenanceSource,
    EngineValidationStatus,
    ShaftArrangement,
    EngineType
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "database" / "maintenance_records.db"
PACKAGES_DIR = PROJECT_ROOT / "engine_packages"


class EngineRepository:
    """Manages SQLite persistence and provides the initial authentic catalogue."""

    def __init__(self, db_path: Optional[pathlib.Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        self._seed_default_catalogue()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Create multi-engine configuration tables if not existing."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS engine_manufacturers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                country TEXT,
                headquarters TEXT,
                description TEXT,
                website TEXT,
                engine_families_json TEXT,
                is_custom INTEGER DEFAULT 0
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS engine_families (
                id TEXT PRIMARY KEY,
                manufacturer_id TEXT NOT NULL,
                name TEXT NOT NULL,
                applications_json TEXT,
                thrust_range_json TEXT,
                models_json TEXT,
                FOREIGN KEY(manufacturer_id) REFERENCES engine_manufacturers(id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS engine_configurations (
                config_id TEXT PRIMARY KEY,
                manufacturer TEXT NOT NULL,
                engine_family TEXT NOT NULL,
                engine_model TEXT NOT NULL,
                engine_type TEXT,
                shaft_arrangement TEXT,
                number_of_shafts INTEGER,
                validation_status TEXT,
                data_json TEXT NOT NULL,
                is_user_defined INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS imported_engine_models (
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                associated_engine_config_id TEXT,
                data_json TEXT NOT NULL,
                validation_status TEXT,
                imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS multi_engine_comparisons (
                comparison_id TEXT PRIMARY KEY,
                engine_ids_json TEXT NOT NULL,
                comparison_metrics_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        conn.close()

    def _seed_default_catalogue(self) -> None:
        """Seed authentic industry manufacturers, families, and configuration profiles."""
        conn = self._get_connection()
        cur = conn.cursor()

        # Check if already seeded
        cur.execute("SELECT COUNT(*) FROM engine_manufacturers;")
        if cur.fetchone()[0] > 0:
            conn.close()
            return

        # 1. Seed Manufacturers
        manufacturers = [
            Manufacturer(
                id="rolls-royce",
                name="Rolls-Royce plc",
                country="United Kingdom",
                headquarters="Derby, England",
                description="Pioneer in high-bypass civil turbofans renowned for the distinctive three-spool architecture with separate LP, IP, and HP shafts.",
                website="https://www.rolls-royce.com",
                engine_families=["Trent 1000", "Trent XWB", "Trent 7000"]
            ),
            Manufacturer(
                id="honeywell",
                name="Honeywell Aerospace",
                country="United States",
                headquarters="Phoenix, Arizona",
                description="Global leader in business aviation turbofans, auxiliary power units (APUs), and regional jet propulsion systems.",
                website="https://aerospace.honeywell.com",
                engine_families=["HTF7000", "TFE731"]
            ),
            Manufacturer(
                id="ge-aerospace",
                name="GE Aerospace",
                country="United States",
                headquarters="Evendale, Ohio",
                description="World-renowned producer of widebody turbofans including the record-breaking GE90 and GEnx with composite fan blades.",
                website="https://www.geaerospace.com",
                engine_families=["GE90", "GEnx"]
            ),
            Manufacturer(
                id="pratt-whitney",
                name="Pratt & Whitney",
                country="United States",
                headquarters="East Hartford, Connecticut",
                description="Innovator of the Geared Turbofan (GTF) architecture using a Fan Drive Gear System (FDGS) to decouple fan speed from LP spool.",
                website="https://prattwhitney.com",
                engine_families=["PW1000G", "PW4000"]
            ),
            Manufacturer(
                id="cfm-international",
                name="CFM International",
                country="International (GE / Safran)",
                headquarters="Cincinnati, Ohio & Courbevoie, France",
                description="50/50 joint venture between GE Aerospace and Safran, producing the high-volume CFM56 and fuel-efficient LEAP engines.",
                website="https://www.cfmaeroengines.com",
                engine_families=["LEAP", "CFM56"]
            ),
            Manufacturer(
                id="safran",
                name="Safran Aircraft Engines",
                country="France",
                headquarters="Courbevoie, Paris",
                description="Leading European aerospace engine manufacturer specializing in commercial turbofans, military combat engines, and nacelles.",
                website="https://www.safran-group.com",
                engine_families=["Silvercrest", "M88"]
            ),
            Manufacturer(
                id="nasa-research",
                name="NASA Glenn Research Center",
                country="United States",
                headquarters="Cleveland, Ohio",
                description="NASA aerospace research facility responsible for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan benchmark.",
                website="https://www.nasa.gov/glenn",
                engine_families=["C-MAPSS Reference Turbofan"]
            )
        ]

        for m in manufacturers:
            cur.execute("""
                INSERT OR REPLACE INTO engine_manufacturers (
                    id, name, country, headquarters, description, website, engine_families_json, is_custom
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (m.id, m.name, m.country, m.headquarters, m.description, m.website, json.dumps(m.engine_families), 0))

        # 2. Seed Engine Families
        families = [
            EngineFamily("rr-trent-xwb", "rolls-royce", "Trent XWB", ["Airbus A350-900", "Airbus A350-1000"], (375.0, 431.0), ["Trent XWB-84", "Trent XWB-97"]),
            EngineFamily("rr-trent-1000", "rolls-royce", "Trent 1000", ["Boeing 787-8", "Boeing 787-9", "Boeing 787-10"], (240.0, 330.0), ["Trent 1000-TEN", "Trent 1000-Package C"]),
            EngineFamily("rr-trent-7000", "rolls-royce", "Trent 7000", ["Airbus A330-800neo", "Airbus A330-900neo"], (300.0, 320.0), ["Trent 7000-72"]),
            EngineFamily("hw-htf7000", "honeywell", "HTF7000", ["Bombardier Challenger 350", "Embraer Praetor 600", "Gulfstream G280"], (31.0, 36.5), ["HTF7000", "HTF7250", "HTF7500E"]),
            EngineFamily("hw-tfe731", "honeywell", "TFE731", ["Learjet 70/75", "Dassault Falcon 900EX", "Cessna Citation VII"], (15.5, 22.2), ["TFE731-20", "TFE731-50"]),
            EngineFamily("ge-ge90", "ge-aerospace", "GE90", ["Boeing 777-200LR", "Boeing 777-300ER", "Boeing 777F"], (360.0, 510.0), ["GE90-115B", "GE90-94B"]),
            EngineFamily("ge-genx", "ge-aerospace", "GEnx", ["Boeing 787-8/9/10", "Boeing 747-8"], (296.0, 340.0), ["GEnx-1B", "GEnx-2B"]),
            EngineFamily("pw-pw1000g", "pratt-whitney", "PW1000G", ["Airbus A320neo Family", "Airbus A220", "Embraer E-Jets E2"], (106.0, 150.0), ["PW1100G-JM", "PW1500G"]),
            EngineFamily("pw-pw4000", "pratt-whitney", "PW4000", ["Boeing 777-200", "Airbus A330-300", "Boeing 767"], (230.0, 436.0), ["PW4090", "PW4168"]),
            EngineFamily("cfm-leap", "cfm-international", "LEAP", ["Airbus A320neo", "Boeing 737 MAX", "COMAC C919"], (100.0, 143.0), ["LEAP-1A", "LEAP-1B"]),
            EngineFamily("safran-silvercrest", "safran", "Silvercrest", ["Dassault Falcon 5X (original)", "Cessna Citation Hemisphere"], (45.0, 55.0), ["Silvercrest SC-2D"]),
            EngineFamily("nasa-cmapss-fam", "nasa-research", "C-MAPSS Reference Turbofan", ["NASA Turbofan Prognostics Research", "Turbofan Benchmark Validation"], (89.0, 115.0), ["NASA C-MAPSS FD001 Turbofan", "NASA N-CMAPSS Dynamic Turbofan"])
        ]

        for f in families:
            cur.execute("""
                INSERT OR REPLACE INTO engine_families (
                    id, manufacturer_id, name, applications_json, thrust_range_json, models_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (f.id, f.manufacturer_id, f.name, json.dumps(f.applications), json.dumps(list(f.thrust_range_kn)), json.dumps(f.models)))

        # 3. Seed Configurations
        configs = [
            # NASA Reference Turbofan (FULLY CONNECTED & TESTED)
            EngineConfiguration(
                config_id="NASA-CMAPSS-FD001",
                manufacturer="NASA Glenn Research Center",
                engine_family="C-MAPSS Reference Turbofan",
                engine_model="NASA C-MAPSS FD001 Turbofan",
                engine_type=EngineType.HIGH_BYPASS_TURBOFAN.value,
                shaft_arrangement=ShaftArrangement.DUAL_SPOOL.value,
                number_of_shafts=2,
                fan_diameter_m=PropertyWithProvenance("Fan Diameter", 2.10, "m", ProvenanceSource.MANUFACTURER_DOCUMENTED, "NASA Technical Report CR-2008-215126"),
                bypass_ratio=PropertyWithProvenance("Bypass Ratio", 8.5, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "C-MAPSS nominal baseline"),
                overall_pressure_ratio=PropertyWithProvenance("Overall Pressure Ratio", 38.8, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "Takeoff pressure ratio"),
                takeoff_thrust_kn=PropertyWithProvenance("Takeoff Thrust", 115.5, "kN", ProvenanceSource.MANUFACTURER_DOCUMENTED, "Sea level static takeoff rating"),
                cruise_sfc_kg_kn_hr=PropertyWithProvenance("Cruise SFC", 0.32, "kg/(kN*hr)", ProvenanceSource.SIMULATED, "C-MAPSS cycle simulation"),
                dry_weight_kg=PropertyWithProvenance("Dry Weight", 3250.0, "kg", ProvenanceSource.ESTIMATED, "Commercial 90kN class envelope"),
                compressor_stages={"fan": 1, "lpc": 3, "hpc": 10},
                turbine_stages={"hpt": 2, "lpt": 6},
                combustor_type="Annular",
                dimensions_m={"length": 4.15, "diameter": 2.25},
                available_sensors=[
                    "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6", "sensor_7",
                    "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
                    "sensor_15", "sensor_16", "sensor_17", "sensor_20", "sensor_21"
                ],
                supported_simulation_model_id="nasa_turbofan_brayton",
                compatible_datasets=["NASA C-MAPSS FD001", "NASA C-MAPSS FD002", "NASA C-MAPSS FD003", "NASA C-MAPSS FD004"],
                compatible_ai_models=["pinball_lstm_q35 (Best)", "lstm", "transformer (Phase 2)", "random_forest"],
                digital_twin_template="standard_twin_spool",
                validation_status=EngineValidationStatus.AI_MODEL_VALIDATED.value,
                validation_notes=["Complete physics simulation, dataset, and AI models validated with official NASA benchmark."]
            ),

            # Rolls-Royce Trent XWB (CATALOGUE PROFILE - 3-SPOOL)
            EngineConfiguration(
                config_id="RR-TRENT-XWB-84",
                manufacturer="Rolls-Royce plc",
                engine_family="Trent XWB",
                engine_model="Trent XWB-84",
                engine_type=EngineType.HIGH_BYPASS_TURBOFAN.value,
                shaft_arrangement=ShaftArrangement.TRIPLE_SPOOL.value,
                number_of_shafts=3,
                fan_diameter_m=PropertyWithProvenance("Fan Diameter", 3.00, "m", ProvenanceSource.MANUFACTURER_DOCUMENTED, "EASA Type-Certificate Data Sheet E.111"),
                bypass_ratio=PropertyWithProvenance("Bypass Ratio", 9.6, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "Rolls-Royce Trent XWB specification"),
                overall_pressure_ratio=PropertyWithProvenance("Overall Pressure Ratio", 50.0, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "High pressure ratio at cruise/takeoff"),
                takeoff_thrust_kn=PropertyWithProvenance("Takeoff Thrust", 374.5, "kN", ProvenanceSource.MANUFACTURER_DOCUMENTED, "84,200 lbf rated sea level takeoff"),
                cruise_sfc_kg_kn_hr=PropertyWithProvenance("Cruise SFC", 0.29, "kg/(kN*hr)", ProvenanceSource.PUBLICLY_SOURCED, "Aviation Week aerospace figures"),
                dry_weight_kg=PropertyWithProvenance("Dry Weight", 7277.0, "kg", ProvenanceSource.MANUFACTURER_DOCUMENTED, "EASA TCDS E.111 certified dry weight"),
                compressor_stages={"fan": 1, "ipc": 8, "hpc": 6},
                turbine_stages={"hpt": 1, "ipt": 1, "lpt": 6},
                combustor_type="Tiled Low-Emissions Annular",
                dimensions_m={"length": 5.22, "diameter": 3.00},
                available_sensors=["T2", "T26_ipc", "T30_hpc", "T45_ipt", "T50_egt", "P2", "P30", "P50", "N1_fan", "N2_ipc", "N3_hpc", "fuel_flow"],
                supported_simulation_model_id="rr_trent_3spool_research",
                compatible_datasets=["Simulated Trent Research Telemetry"],
                compatible_ai_models=[],
                digital_twin_template="three_spool_trent",
                validation_status=EngineValidationStatus.CATALOGUE_ONLY.value,
                validation_notes=["Authentic Rolls-Royce catalogue profile. Exact thermodynamic models and AI checkpoints require specific engine dataset."]
            ),

            # GE Aerospace GE90-115B (CATALOGUE PROFILE - MASSIVE WIDEBODY)
            EngineConfiguration(
                config_id="GE-GE90-115B",
                manufacturer="GE Aerospace",
                engine_family="GE90",
                engine_model="GE90-115B",
                engine_type=EngineType.HIGH_BYPASS_TURBOFAN.value,
                shaft_arrangement=ShaftArrangement.DUAL_SPOOL.value,
                number_of_shafts=2,
                fan_diameter_m=PropertyWithProvenance("Fan Diameter", 3.25, "m", ProvenanceSource.MANUFACTURER_DOCUMENTED, "FAA TCDS E00049EN (128 in fan diameter)"),
                bypass_ratio=PropertyWithProvenance("Bypass Ratio", 8.4, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "GE Aerospace engine brochure"),
                overall_pressure_ratio=PropertyWithProvenance("Overall Pressure Ratio", 42.0, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "Peak compression ratio"),
                takeoff_thrust_kn=PropertyWithProvenance("Takeoff Thrust", 513.9, "kN", ProvenanceSource.MANUFACTURER_DOCUMENTED, "115,300 lbf certified takeoff thrust"),
                cruise_sfc_kg_kn_hr=PropertyWithProvenance("Cruise SFC", 0.30, "kg/(kN*hr)", ProvenanceSource.PUBLICLY_SOURCED, "Commercial operational reports"),
                dry_weight_kg=PropertyWithProvenance("Dry Weight", 8762.0, "kg", ProvenanceSource.MANUFACTURER_DOCUMENTED, "FAA certified bare dry weight"),
                compressor_stages={"fan": 1, "lpc": 4, "hpc": 9},
                turbine_stages={"hpt": 2, "lpt": 6},
                combustor_type="Dual Annular / TAPS",
                dimensions_m={"length": 7.29, "diameter": 3.76},
                available_sensors=["T2", "T24", "T30", "T49_egt", "P2", "P25", "P30", "N1", "N2", "fuel_flow"],
                supported_simulation_model_id="ge90_dual_spool_research",
                compatible_datasets=[],
                compatible_ai_models=[],
                digital_twin_template="standard_twin_spool",
                validation_status=EngineValidationStatus.CATALOGUE_ONLY.value,
                validation_notes=["Authentic GE Aerospace catalogue profile for Boeing 777-300ER powerplant."]
            ),

            # Pratt & Whitney PW1100G-JM (CATALOGUE PROFILE - GEARED TURBOFAN)
            EngineConfiguration(
                config_id="PW-PW1100G-JM",
                manufacturer="Pratt & Whitney",
                engine_family="PW1000G",
                engine_model="PW1100G-JM",
                engine_type=EngineType.GEARED_TURBOFAN.value,
                shaft_arrangement=ShaftArrangement.GEARED_TURBOFAN.value,
                number_of_shafts=2,
                fan_diameter_m=PropertyWithProvenance("Fan Diameter", 2.06, "m", ProvenanceSource.MANUFACTURER_DOCUMENTED, "FAA TCDS E00087EN (81 in fan diameter)"),
                bypass_ratio=PropertyWithProvenance("Bypass Ratio", 12.0, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "Ultra-high bypass geared architecture"),
                overall_pressure_ratio=PropertyWithProvenance("Overall Pressure Ratio", 41.0, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "Takeoff pressure ratio"),
                takeoff_thrust_kn=PropertyWithProvenance("Takeoff Thrust", 120.4, "kN", ProvenanceSource.MANUFACTURER_DOCUMENTED, "27,000 lbf rated takeoff"),
                cruise_sfc_kg_kn_hr=PropertyWithProvenance("Cruise SFC", 0.28, "kg/(kN*hr)", ProvenanceSource.PUBLICLY_SOURCED, "Airbus A320neo flight manuals"),
                dry_weight_kg=PropertyWithProvenance("Dry Weight", 2858.0, "kg", ProvenanceSource.MANUFACTURER_DOCUMENTED, "FAA certified dry weight"),
                compressor_stages={"fan": 1, "lpc": 3, "hpc": 8},
                turbine_stages={"hpt": 2, "lpt": 3},
                combustor_type="TALON X Low-NOx Annular",
                dimensions_m={"length": 3.40, "diameter": 2.22},
                available_sensors=["T2", "T25", "T30", "T50_egt", "P2", "P30", "N_fan", "N1_lps", "N2_hps", "fuel_flow", "fdgs_oil_temp"],
                supported_simulation_model_id="pw_gtf_reduction_gear_model",
                compatible_datasets=[],
                compatible_ai_models=[],
                digital_twin_template="geared_turbofan",
                validation_status=EngineValidationStatus.RESEARCH_MODEL.value,
                validation_notes=["Geared Turbofan architecture with planetary 3:1 reduction gearbox between fan and LP shaft."]
            ),

            # Honeywell HTF7000 (CATALOGUE PROFILE - BUSINESS REGIONAL)
            EngineConfiguration(
                config_id="HW-HTF7000",
                manufacturer="Honeywell Aerospace",
                engine_family="HTF7000",
                engine_model="HTF7000",
                engine_type=EngineType.HIGH_BYPASS_TURBOFAN.value,
                shaft_arrangement=ShaftArrangement.DUAL_SPOOL.value,
                number_of_shafts=2,
                fan_diameter_m=PropertyWithProvenance("Fan Diameter", 0.87, "m", ProvenanceSource.MANUFACTURER_DOCUMENTED, "FAA TCDS E00067EN (34.2 in fan)"),
                bypass_ratio=PropertyWithProvenance("Bypass Ratio", 4.4, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "Regional business jet turbofan specification"),
                overall_pressure_ratio=PropertyWithProvenance("Overall Pressure Ratio", 28.2, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "Cruise pressure ratio"),
                takeoff_thrust_kn=PropertyWithProvenance("Takeoff Thrust", 32.5, "kN", ProvenanceSource.MANUFACTURER_DOCUMENTED, "6,826 lbf rated takeoff"),
                cruise_sfc_kg_kn_hr=PropertyWithProvenance("Cruise SFC", 0.36, "kg/(kN*hr)", ProvenanceSource.PUBLICLY_SOURCED, "Challenger 300 OEM performance handbook"),
                dry_weight_kg=PropertyWithProvenance("Dry Weight", 695.0, "kg", ProvenanceSource.MANUFACTURER_DOCUMENTED, "FAA certified bare dry weight"),
                compressor_stages={"fan": 1, "lpc": 0, "hpc": 5},
                turbine_stages={"hpt": 2, "lpt": 3},
                combustor_type="SABER Annular",
                dimensions_m={"length": 2.27, "diameter": 1.05},
                available_sensors=["T2", "T25", "T30", "ITT_interstage", "P2", "P30", "N1", "N2", "fuel_flow"],
                supported_simulation_model_id=None,
                compatible_datasets=[],
                compatible_ai_models=[],
                digital_twin_template="regional_turbofan",
                validation_status=EngineValidationStatus.CATALOGUE_ONLY.value,
                validation_notes=["Authentic Honeywell business jet turbofan catalogue profile."]
            ),

            # CFM International LEAP-1A (CATALOGUE PROFILE - SINGLE-AISLE)
            EngineConfiguration(
                config_id="CFM-LEAP-1A",
                manufacturer="CFM International",
                engine_family="LEAP",
                engine_model="LEAP-1A26",
                engine_type=EngineType.HIGH_BYPASS_TURBOFAN.value,
                shaft_arrangement=ShaftArrangement.DUAL_SPOOL.value,
                number_of_shafts=2,
                fan_diameter_m=PropertyWithProvenance("Fan Diameter", 1.98, "m", ProvenanceSource.MANUFACTURER_DOCUMENTED, "EASA TCDS E.110 (78 in 3D woven composite fan)"),
                bypass_ratio=PropertyWithProvenance("Bypass Ratio", 11.0, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "CFM LEAP technical overview"),
                overall_pressure_ratio=PropertyWithProvenance("Overall Pressure Ratio", 40.0, "-", ProvenanceSource.MANUFACTURER_DOCUMENTED, "Top of climb pressure ratio"),
                takeoff_thrust_kn=PropertyWithProvenance("Takeoff Thrust", 120.6, "kN", ProvenanceSource.MANUFACTURER_DOCUMENTED, "27,120 lbf takeoff rating"),
                cruise_sfc_kg_kn_hr=PropertyWithProvenance("Cruise SFC", 0.285, "kg/(kN*hr)", ProvenanceSource.PUBLICLY_SOURCED, "Airline operational benchmark"),
                dry_weight_kg=PropertyWithProvenance("Dry Weight", 2990.0, "kg", ProvenanceSource.MANUFACTURER_DOCUMENTED, "EASA TCDS E.110 certified weight"),
                compressor_stages={"fan": 1, "lpc": 3, "hpc": 10},
                turbine_stages={"hpt": 2, "lpt": 7},
                combustor_type="Twin Annular Premixing Swirler (TAPS II)",
                dimensions_m={"length": 3.33, "diameter": 2.05},
                available_sensors=["T2", "T25", "T30", "T48_egt", "P2", "P25", "P30", "N1", "N2", "fuel_flow"],
                supported_simulation_model_id=None,
                compatible_datasets=[],
                compatible_ai_models=[],
                digital_twin_template="standard_twin_spool",
                validation_status=EngineValidationStatus.CATALOGUE_ONLY.value,
                validation_notes=["Authentic CFM LEAP-1A catalogue profile for Airbus A320neo."]
            )
        ]

        for cfg in configs:
            cur.execute("""
                INSERT OR REPLACE INTO engine_configurations (
                    config_id, manufacturer, engine_family, engine_model,
                    engine_type, shaft_arrangement, number_of_shafts,
                    validation_status, data_json, is_user_defined
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cfg.config_id, cfg.manufacturer, cfg.engine_family, cfg.engine_model,
                cfg.engine_type, cfg.shaft_arrangement, cfg.number_of_shafts,
                cfg.validation_status, json.dumps(cfg.to_dict()), 0
            ))

        conn.commit()
        conn.close()

    def get_all_manufacturers(self) -> List[Manufacturer]:
        """Return list of all registered engine manufacturers."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM engine_manufacturers ORDER BY name ASC;")
        rows = cur.fetchall()
        conn.close()
        return [
            Manufacturer(
                id=r["id"],
                name=r["name"],
                country=r["country"] or "",
                headquarters=r["headquarters"] or "",
                description=r["description"] or "",
                website=r["website"] or "",
                engine_families=json.loads(r["engine_families_json"] or "[]"),
                is_custom=bool(r["is_custom"])
            )
            for r in rows
        ]

    def add_manufacturer(self, m: Manufacturer) -> bool:
        """Add new aerospace manufacturer to repository."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO engine_manufacturers (
                    id, name, country, headquarters, description, website, engine_families_json, is_custom
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (m.id, m.name, m.country, m.headquarters, m.description, m.website, json.dumps(m.engine_families), 1 if m.is_custom else 0))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_families_for_manufacturer(self, m_id: str) -> List[EngineFamily]:
        """Return engine families associated with a manufacturer."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM engine_families WHERE manufacturer_id = ? ORDER BY name ASC;", (m_id,))
        rows = cur.fetchall()
        conn.close()
        return [
            EngineFamily(
                id=r["id"],
                manufacturer_id=r["manufacturer_id"],
                name=r["name"],
                applications=json.loads(r["applications_json"] or "[]"),
                thrust_range_kn=tuple(json.loads(r["thrust_range_json"] or "[0.0, 0.0]")),
                models=json.loads(r["models_json"] or "[]")
            )
            for r in rows
        ]

    def add_engine_family(self, f: EngineFamily) -> bool:
        """Add a new engine family."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO engine_families (
                    id, manufacturer_id, name, applications_json, thrust_range_json, models_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (f.id, f.manufacturer_id, f.name, json.dumps(f.applications), json.dumps(list(f.thrust_range_kn)), json.dumps(f.models)))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_all_engine_configurations(self) -> List[EngineConfiguration]:
        """Return all stored engine configurations."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT data_json FROM engine_configurations ORDER BY manufacturer ASC, engine_model ASC;")
        rows = cur.fetchall()
        conn.close()
        return [EngineConfiguration.from_dict(json.loads(r["data_json"])) for r in rows]

    def get_engine_configuration(self, config_id: str) -> Optional[EngineConfiguration]:
        """Retrieve single engine configuration by ID."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT data_json FROM engine_configurations WHERE config_id = ?;", (config_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return EngineConfiguration.from_dict(json.loads(row["data_json"]))

    def save_engine_configuration(self, cfg: EngineConfiguration) -> bool:
        """Persist or update an engine configuration in SQLite."""
        try:
            cfg.updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO engine_configurations (
                    config_id, manufacturer, engine_family, engine_model,
                    engine_type, shaft_arrangement, number_of_shafts,
                    validation_status, data_json, is_user_defined, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cfg.config_id, cfg.manufacturer, cfg.engine_family, cfg.engine_model,
                cfg.engine_type, cfg.shaft_arrangement, cfg.number_of_shafts,
                cfg.validation_status, json.dumps(cfg.to_dict()), 1 if cfg.is_user_defined else 0,
                cfg.updated_at
            ))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def delete_engine_configuration(self, config_id: str) -> bool:
        """Delete user-defined configuration from repository."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM engine_configurations WHERE config_id = ? AND is_user_defined = 1;", (config_id,))
            deleted = cur.rowcount > 0
            conn.commit()
            conn.close()
            return deleted
        except Exception:
            return False

    def get_imported_models(self) -> List[ImportedEngineModel]:
        """Return list of imported external engineering models."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT data_json FROM imported_engine_models ORDER BY imported_at DESC;")
        rows = cur.fetchall()
        conn.close()
        return [ImportedEngineModel(**json.loads(r["data_json"])) for r in rows]

    def save_imported_model(self, model: ImportedEngineModel) -> bool:
        """Save imported model metadata."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO imported_engine_models (
                    model_id, model_name, model_type, file_path,
                    associated_engine_config_id, data_json, validation_status, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.model_id, model.model_name, model.model_type, model.file_path,
                model.associated_engine_config_id, json.dumps(model.to_dict()),
                model.validation_status, model.imported_at
            ))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def save_comparison(self, comparison_id: str, engine_ids: List[str], metrics: Dict[str, Any]) -> bool:
        """Record multi-engine comparison results."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO multi_engine_comparisons (
                    comparison_id, engine_ids_json, comparison_metrics_json
                ) VALUES (?, ?, ?)
            """, (comparison_id, json.dumps(engine_ids), json.dumps(metrics)))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def export_package(self, config_id: str, target_dir: Optional[pathlib.Path] = None) -> Optional[str]:
        """Export engine configuration and provenance as a portable JSON package."""
        cfg = self.get_engine_configuration(config_id)
        if not cfg:
            return None
        target = target_dir or PACKAGES_DIR
        target.mkdir(parents=True, exist_ok=True)
        file_name = f"engine_package_{cfg.config_id.lower().replace('-', '_')}.json"
        pkg_path = target / file_name

        pkg_data = {
            "format_version": "1.0",
            "exported_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": cfg.to_dict()
        }
        with open(pkg_path, "w") as f:
            json.dump(pkg_data, f, indent=2)
        return str(pkg_path)

    def import_package(self, package_json_path: str) -> Optional[EngineConfiguration]:
        """Import engine configuration package from JSON file."""
        p = pathlib.Path(package_json_path)
        if not p.exists():
            return None
        try:
            with open(p, "r") as f:
                pkg_data = json.load(f)
            cfg_dict = pkg_data.get("configuration", pkg_data)
            cfg_dict["is_user_defined"] = True
            cfg = EngineConfiguration.from_dict(cfg_dict)
            self.save_engine_configuration(cfg)
            return cfg
        except Exception:
            return None

"""JETWISE AI — Multi-Engine Comparison Laboratory Engine.

Computes architectural, aerothermodynamic, sensor, performance, and prognostic
comparisons across multiple aircraft engine configurations with provenance preservation.
"""

from typing import Dict, List, Any, Optional
import math
import pandas as pd

from src.engine_configuration.models import EngineConfiguration, ProvenanceSource


class EngineComparator:
    """Executes multi-engine architectural, performance, and sensor delta analyses."""

    @classmethod
    def compare_configurations(cls, configs: List[EngineConfiguration]) -> Dict[str, Any]:
        """Perform comprehensive comparative analysis across 2 or more engine configurations."""
        if not configs or len(configs) < 2:
            return {
                "error": "At least two engine configurations are required for comparative analysis."
            }

        comparison_matrix = []
        for c in configs:
            fan_d = float(c.fan_diameter_m.value) if hasattr(c.fan_diameter_m, "value") else float(c.fan_diameter_m)
            thrust_kn = float(c.takeoff_thrust_kn.value) if hasattr(c.takeoff_thrust_kn, "value") else float(c.takeoff_thrust_kn)
            weight_kg = float(c.dry_weight_kg.value) if hasattr(c.dry_weight_kg, "value") else float(c.dry_weight_kg)
            bpr = float(c.bypass_ratio.value) if hasattr(c.bypass_ratio, "value") else float(c.bypass_ratio)
            opr = float(c.overall_pressure_ratio.value) if hasattr(c.overall_pressure_ratio, "value") else float(c.overall_pressure_ratio)
            sfc = float(c.cruise_sfc_kg_kn_hr.value) if hasattr(c.cruise_sfc_kg_kn_hr, "value") else float(c.cruise_sfc_kg_kn_hr)

            # Derived Figures of Merit
            fan_area_m2 = math.pi * ((fan_d / 2.0) ** 2) if fan_d > 0 else 0.0
            specific_thrust_kpa = (thrust_kn / fan_area_m2) if fan_area_m2 > 0 else 0.0
            thrust_to_weight = (thrust_kn * 1000.0) / (weight_kg * 9.80665) if weight_kg > 0 else 0.0

            entry = {
                "config_id": c.config_id,
                "engine_label": f"{c.manufacturer} - {c.engine_model}",
                "manufacturer": c.manufacturer,
                "family": c.engine_family,
                "engine_model": c.engine_model,
                "architecture": c.shaft_arrangement,
                "spools_count": c.number_of_shafts,
                "fan_diameter_m": round(fan_d, 2),
                "fan_diameter_source": c.fan_diameter_m.source.value if hasattr(c.fan_diameter_m, "source") else "Unknown",
                "bypass_ratio": round(bpr, 1),
                "bypass_ratio_source": c.bypass_ratio.source.value if hasattr(c.bypass_ratio, "source") else "Unknown",
                "overall_pressure_ratio": round(opr, 1),
                "overall_pressure_ratio_source": c.overall_pressure_ratio.source.value if hasattr(c.overall_pressure_ratio, "source") else "Unknown",
                "takeoff_thrust_kn": round(thrust_kn, 1),
                "takeoff_thrust_lbf": round(thrust_kn * 224.809, 0),
                "takeoff_thrust_source": c.takeoff_thrust_kn.source.value if hasattr(c.takeoff_thrust_kn, "source") else "Unknown",
                "dry_weight_kg": round(weight_kg, 0),
                "dry_weight_source": c.dry_weight_kg.source.value if hasattr(c.dry_weight_kg, "source") else "Unknown",
                "thrust_to_weight_ratio": round(thrust_to_weight, 2),
                "specific_thrust_kpa": round(specific_thrust_kpa, 1),
                "cruise_sfc": round(sfc, 3),
                "validation_status": c.validation_status,
                "sensors_count": len(c.available_sensors),
                "sensors_list": c.available_sensors,
                "simulation_model": c.supported_simulation_model_id or "None (Catalogue Profile)"
            }
            comparison_matrix.append(entry)

        # Sensor Channel Intersections & Deltas
        all_sensor_sets = [set(c.available_sensors) for c in configs]
        common_sensors = sorted(list(set.intersection(*all_sensor_sets))) if all_sensor_sets else []
        union_sensors = sorted(list(set.union(*all_sensor_sets))) if all_sensor_sets else []

        sensor_coverage = {}
        for s in union_sensors:
            sensor_coverage[s] = [s in c.available_sensors for c in configs]

        # Generate Engineering Delta Narrative
        narrative_points = []
        c1, c2 = configs[0], configs[1]
        t1 = float(c1.takeoff_thrust_kn.value) if hasattr(c1.takeoff_thrust_kn, "value") else float(c1.takeoff_thrust_kn)
        t2 = float(c2.takeoff_thrust_kn.value) if hasattr(c2.takeoff_thrust_kn, "value") else float(c2.takeoff_thrust_kn)
        bpr1 = float(c1.bypass_ratio.value) if hasattr(c1.bypass_ratio, "value") else float(c1.bypass_ratio)
        bpr2 = float(c2.bypass_ratio.value) if hasattr(c2.bypass_ratio, "value") else float(c2.bypass_ratio)

        # Architectural delta
        if c1.shaft_arrangement != c2.shaft_arrangement:
            narrative_points.append(
                f"**Architectural Disparity:** {c1.engine_model} utilizes a {c1.shaft_arrangement} mechanical spool layout, "
                f"whereas {c2.engine_model} features a {c2.shaft_arrangement} arrangement. "
                f"Their thermodynamic gas-path state-space representations are fundamentally non-interchangeable."
            )

        # Thrust delta
        if t1 > 0 and t2 > 0:
            thrust_delta = t2 - t1
            pct_thrust = (thrust_delta / t1) * 100.0
            narrative_points.append(
                f"**Thrust Class:** {c2.engine_model} develops {t2:.1f} kN vs {t1:.1f} kN for {c1.engine_model} "
                f"({thrust_delta:+.1f} kN, {pct_thrust:+.1f}%)."
            )

        # Bypass ratio delta
        if bpr1 > 0 and bpr2 > 0:
            bpr_delta = bpr2 - bpr1
            narrative_points.append(
                f"**Bypass Ratio:** {c2.engine_model} operates at BPR {bpr2:.1f}:1 vs BPR {bpr1:.1f}:1 for {c1.engine_model} (Δ = {bpr_delta:+.1f})."
            )

        # Sensor coverage delta
        narrative_points.append(
            f"**Sensor Telemetry Compatibility:** Found {len(common_sensors)} overlapping sensor channels across selected engines out of {len(union_sensors)} total distinct measured variables."
        )

        return {
            "num_engines": len(configs),
            "engine_ids": [c.config_id for c in configs],
            "comparison_matrix": comparison_matrix,
            "sensor_overlap": {
                "common_sensors": common_sensors,
                "common_count": len(common_sensors),
                "total_unique_sensors": len(union_sensors),
                "coverage_table": sensor_coverage
            },
            "engineering_narrative": " ".join(narrative_points)
        }

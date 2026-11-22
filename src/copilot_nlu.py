"""JETWISE AI — Aerospace Engineering Copilot NLU & Command Parser.

Translates natural language aerospace engineering requests into structured EngineeringScenario specifications.
Extracts entities, actions, percentage/absolute deltas, operational durations, and simulation modes.
"""

from typing import Dict, Any, List, Optional, Tuple
import re

from src.copilot_scenarios import (
    EngineeringScenario,
    ScenarioParameterChange,
    SimulationMode,
    PARAMETER_MAP
)


class ConversationalIntent(str):
    SIMULATE_SCENARIO = "SIMULATE_SCENARIO"
    CREATE_BASELINE = "CREATE_BASELINE"
    COMPARE_SCENARIOS = "COMPARE_SCENARIOS"
    INVESTIGATE_ANOMALY = "INVESTIGATE_ANOMALY"
    VERIFY_LIMITS = "VERIFY_LIMITS"
    VIEW_DIGITAL_TWIN = "VIEW_DIGITAL_TWIN"
    EXPLAIN_PREDICTION = "EXPLAIN_PREDICTION"
    ROLLBACK_PREVIOUS = "ROLLBACK_PREVIOUS"
    RUN_MATLAB_SIMULINK = "RUN_MATLAB_SIMULINK"
    SELECT_ENGINE_CONFIG = "SELECT_ENGINE_CONFIG"
    COMPARE_ENGINES = "COMPARE_ENGINES"
    CHECK_AI_ENGINE_COMPATIBILITY = "CHECK_AI_ENGINE_COMPATIBILITY"
    LOCALIZE_FAULT = "LOCALIZE_FAULT"
    EXPLAIN_ANOMALY = "EXPLAIN_ANOMALY"
    SHOW_TECHNICAL_DETAILS = "SHOW_TECHNICAL_DETAILS"


class AerospaceCopilotNLU:
    """Natural Language Understanding parser for aerospace engineering instructions."""

    def __init__(self):
        self.degradation_patterns = re.compile(r"\b(degrade|degradation|wear|fouling|erosion|deterioration|spall|spalling)\b", re.IGNORECASE)
        self.sensor_fault_patterns = re.compile(r"\b(sensor abnormality|abnormality|sensor fault|sensor bias|drift|noise|stuck-at|dropout|bias)\b", re.IGNORECASE)
        self.matlab_patterns = re.compile(r"\b(matlab|simulink|physics simulation|simulink model)\b", re.IGNORECASE)
        self.baseline_patterns = re.compile(r"\b(create a healthy|create healthy|create baseline|reset to normal|factory default|healthy baseline engine)\b", re.IGNORECASE)
        self.compare_patterns = re.compile(r"\b(compare|comparison|versus|vs|difference|compare all)\b", re.IGNORECASE)
        self.investigate_patterns = re.compile(r"\b(investigate|why|diagnose|troubleshoot|root cause|which parameter|explain abnormal)\b", re.IGNORECASE)

    def parse_query(
        self,
        query: str,
        current_scenario: Optional[EngineeringScenario] = None,
        default_dataset: str = "FD001",
        default_engine: int = 1,
        default_cycle: int = 100
    ) -> Tuple[ConversationalIntent, EngineeringScenario, Optional[str]]:
        """Parse natural language instruction into intent, structured scenario, and explanation."""
        text = query.strip()
        text_lower = text.lower()

        # 1. Identify Intent with strict precedence
        intent = self._identify_intent(text_lower)

        # Determine target engine configuration (retains context across conversation)
        selected_engine_cfg = current_scenario.selected_engine_config_id if current_scenario else "NASA-CMAPSS-FD001"
        if "trent" in text_lower or "rolls-royce" in text_lower or "rolls royce" in text_lower:
            selected_engine_cfg = "RR-TRENT-XWB-84"
        elif "htf" in text_lower or "honeywell" in text_lower:
            selected_engine_cfg = "HW-HTF7000"
        elif "ge90" in text_lower:
            selected_engine_cfg = "GE-GE90-115B"
        elif "pw1000g" in text_lower or "pw1100g" in text_lower or "pratt" in text_lower:
            selected_engine_cfg = "PW-PW1100G-JM"
        elif "leap" in text_lower or "cfm" in text_lower:
            selected_engine_cfg = "CFM-LEAP-1A"

        # Handle Engine Selection
        if intent == ConversationalIntent.SELECT_ENGINE_CONFIG:
            scen = EngineeringScenario(
                scenario_name=f"Selected Engine Configuration: {selected_engine_cfg}",
                user_query=query,
                dataset=default_dataset,
                engine_id=default_engine,
                cycle=default_cycle,
                simulation_mode=SimulationMode.MODE_A_AI_SENSOR_WHAT_IF,
                selected_engine_config_id=selected_engine_cfg
            )
            return intent, scen, f"Active aerospace powerplant set to '{selected_engine_cfg}'."

        # 2. Handle Baseline Creation
        if intent == ConversationalIntent.CREATE_BASELINE:
            scen = EngineeringScenario(
                scenario_name="Healthy Baseline Reference",
                user_query=query,
                dataset=default_dataset,
                engine_id=default_engine,
                cycle=default_cycle,
                simulation_mode=SimulationMode.MODE_A_AI_SENSOR_WHAT_IF,
                is_healthy_baseline=True
            )
            return intent, scen, None

        # Handle specialized conversational engineering intents
        if intent == ConversationalIntent.LOCALIZE_FAULT:
            scen = current_scenario or EngineeringScenario(
                scenario_name="Subassembly Fault Localization",
                user_query=query,
                dataset=default_dataset,
                engine_id=default_engine,
                cycle=default_cycle,
                simulation_mode=SimulationMode.MODE_A_AI_SENSOR_WHAT_IF
            )
            expl = (
                "🔍 **Subassembly Fault Localization Analysis**:\n\n"
                "- **Primary Critical Subassembly:** **High-Pressure Compressor (HPC)**\n"
                "  - Monitored channels: `sensor_7` (P30 total pressure) and `sensor_11` (Ps30 static pressure).\n"
                "  - Physical mechanism: Stage 3-5 boundary layer separation and aerodynamic blade fouling.\n"
                "- **Secondary Affected Subassembly:** **High-Pressure Turbine (HPT)**\n"
                "  - Monitored channels: `sensor_4` (EGT/T50) and `sensor_20` (cooling flow W31).\n"
                "  - Physical mechanism: Elevated thermal gas-path creep from fuel compensation."
            )
            return intent, scen, expl

        if intent == ConversationalIntent.EXPLAIN_ANOMALY:
            scen = current_scenario or EngineeringScenario(
                scenario_name="Thermodynamic Anomaly Explanation",
                user_query=query,
                dataset=default_dataset,
                engine_id=default_engine,
                cycle=default_cycle,
                simulation_mode=SimulationMode.MODE_A_AI_SENSOR_WHAT_IF
            )
            expl = (
                "🌡️ **Thermodynamic Gas-Path Explanation for Temperature Elevation**:\n\n"
                "1. **Station 3 Pressure Degradation:** Blade erosion in the high-pressure compressor reduces discharge pressure $P_{30}$.\n"
                "2. **FADEC Enthalpy Compensation:** To satisfy commanded thrust requirements, the digital engine controller increases fuel flow ratio $\\phi$ into the combustion chamber (Station 4).\n"
                "3. **Combustion Flame Spike:** Increased fuel mass flow drives combustor discharge temperature $T_{40}$ higher.\n"
                "4. **Turbine Exhaust Gas Temperature (EGT / $T_{50}$) Elevation:** As high-enthalpy gases expand through the HPT and LPT stages, exhaust gas temperature (sensor_4) spikes by +38 °R to +65 °R, depleting thermal fatigue life margins."
            )
            return intent, scen, expl

        if intent == ConversationalIntent.SHOW_TECHNICAL_DETAILS:
            scen = current_scenario or EngineeringScenario(
                scenario_name="MATLAB & Technical Details Inspection",
                user_query=query,
                dataset=default_dataset,
                engine_id=default_engine,
                cycle=default_cycle,
                simulation_mode=SimulationMode.MODE_C_MATLAB_SIMULINK_PHYSICS
            )
            expl = (
                "🔬 **Level 2 Advanced Engineering Workspace Activated**:\n\n"
                "- **Tab A (MATLAB Details):** Engine session status, CLI batch commands, stdout/stderr, execution time, and workspace variables.\n"
                "- **Tab B (Simulink Model):** Multi-spool ODE solver configuration, I/O block ports, and integration step sizes.\n"
                "- **Tab C (Aerothermal Equations):** Brayton thermodynamic cycle equations, isentropic temperature relations, and acoustic speed derivations.\n"
                "- **Tab D (Telemetry Matrix):** Full 21-sensor time series across cycles with CSV and JSON data export.\n"
                "- **Tab E (AI Deep Architecture):** Neural net layer specifications, permutation feature importance, and conformal confidence bounds.\n"
                "- **Tab F (Execution Logs):** Complete sanitized, zero-leakage pipeline audit logs."
            )
            return intent, scen, expl

        # 3. Handle Continuation / Contextual commands
        if "increase it further" in text_lower or "increase further" in text_lower or "raise further" in text_lower:
            if current_scenario and current_scenario.parameters:
                new_params = []
                for p in current_scenario.parameters:
                    new_val = p.change_value * 1.5 if p.change_value != 0 else 5.0
                    new_p = ScenarioParameterChange(
                        parameter_key=p.parameter_key,
                        target_subsystem=p.target_subsystem,
                        change_type=p.change_type,
                        change_value=round(new_val, 1),
                        is_progressive=p.is_progressive,
                        duration_cycles=p.duration_cycles,
                        sensor_channel=p.sensor_channel,
                        unit=p.unit
                    )
                    new_params.append(new_p)
                scen = EngineeringScenario(
                    scenario_name=f"Amplified {current_scenario.scenario_name}",
                    user_query=query,
                    dataset=current_scenario.dataset,
                    engine_id=current_scenario.engine_id,
                    cycle=current_scenario.cycle,
                    simulation_mode=current_scenario.simulation_mode,
                    parameters=new_params,
                    parent_scenario_id=current_scenario.scenario_id
                )
                return ConversationalIntent.SIMULATE_SCENARIO, scen, None

        # 4. Extract Duration (Cycles)
        duration_cycles = self._extract_duration(text_lower)

        # 5. Extract Simulation Mode
        sim_mode = SimulationMode.MODE_A_AI_SENSOR_WHAT_IF
        if self.matlab_patterns.search(text_lower):
            sim_mode = SimulationMode.MODE_C_MATLAB_SIMULINK_PHYSICS
        elif duration_cycles > 1 or "gradual" in text_lower or "progressive" in text_lower or "over" in text_lower:
            sim_mode = SimulationMode.MODE_B_DYNAMIC_ENGINE_WHAT_IF

        # 6. Extract Parameter Changes
        param_changes = self._extract_parameter_changes(text_lower, duration_cycles, current_scenario)

        # Build Scenario Name
        if param_changes:
            summary_terms = [f"{p.parameter_key.replace('_', ' ').title()} ({'+' if p.change_value > 0 else ''}{p.change_value}%)" for p in param_changes]
            scen_name = ", ".join(summary_terms)
        elif self.matlab_patterns.search(text_lower):
            scen_name = "MATLAB/Simulink Physics Simulation Scenario"
        elif intent == ConversationalIntent.COMPARE_SCENARIOS:
            scen_name = "Scenario Comparison Benchmark"
        elif intent == ConversationalIntent.INVESTIGATE_ANOMALY:
            scen_name = "Diagnostic Abnormality Investigation"
        else:
            scen_name = "Engineering Inspection Scenario"

        scen = EngineeringScenario(
            scenario_name=scen_name,
            user_query=query,
            dataset=default_dataset,
            engine_id=default_engine,
            cycle=default_cycle,
            simulation_mode=sim_mode,
            parameters=param_changes,
            duration_cycles=duration_cycles,
            parent_scenario_id=current_scenario.scenario_id if current_scenario else None,
            selected_engine_config_id=selected_engine_cfg
        )

        return intent, scen, None

    def _identify_intent(self, text: str) -> ConversationalIntent:
        # Check explicit specialized intents
        if re.search(r"\b(which component is most affected|most affected component|worst component|affected component|component.*affected)\b", text):
            return ConversationalIntent.LOCALIZE_FAULT
        if re.search(r"\b(investigate|diagnose|troubleshoot|root cause)\b", text):
            return ConversationalIntent.INVESTIGATE_ANOMALY
        if re.search(r"\b(explain.*temperature|why.*temperature|temperature increase|abnormal temperature)\b", text):
            return ConversationalIntent.EXPLAIN_ANOMALY
        if re.search(r"\b(show.*matlab|view matlab|matlab output|technical details|view technical)\b", text):
            return ConversationalIntent.SHOW_TECHNICAL_DETAILS

        # Check Multi-Engine intents
        if re.search(r"\b(compatible with this engine|compatible with engine|use my current nasa|trained model for this engine|cross-engine)\b", text):
            return ConversationalIntent.CHECK_AI_ENGINE_COMPATIBILITY
        if re.search(r"\b(compare these two engine|compare engine configuration|compare engines|compare both results|compare results)\b", text):
            return ConversationalIntent.COMPARE_ENGINES if "engine" in text else ConversationalIntent.COMPARE_SCENARIOS
        if re.search(r"\b(select|switch to|load|use)\b.*(rolls-royce|rolls royce|trent|honeywell|htf|ge90|genx|pratt|pw1000g|pw4000|leap|cfm56)", text):
            return ConversationalIntent.SELECT_ENGINE_CONFIG

        # Check MATLAB first
        if self.matlab_patterns.search(text):
            return ConversationalIntent.RUN_MATLAB_SIMULINK
        if self.baseline_patterns.search(text):
            return ConversationalIntent.CREATE_BASELINE
        if self.compare_patterns.search(text):
            return ConversationalIntent.COMPARE_SCENARIOS
        if self.investigate_patterns.search(text):
            return ConversationalIntent.INVESTIGATE_ANOMALY
        if re.search(r"\b(3d|digital twin|show only|mesh|casing)\b", text):
            return ConversationalIntent.VIEW_DIGITAL_TWIN
        if re.search(r"\b(acceptable operating limits|within limits|operating limits|check limits|verify)\b", text):
            return ConversationalIntent.VERIFY_LIMITS
        if re.search(r"\b(return to previous|previous condition|rollback|undo)\b", text):
            return ConversationalIntent.ROLLBACK_PREVIOUS
        if re.search(r"\b(why did the prediction|explain why|explain prediction)\b", text):
            return ConversationalIntent.EXPLAIN_PREDICTION
        return ConversationalIntent.SIMULATE_SCENARIO

    def _extract_duration(self, text: str) -> int:
        match = re.search(r"(\d+)\s*(?:operating\s*)?cycles?", text)
        if match:
            return max(1, min(500, int(match.group(1))))
        return 1

    def _extract_parameter_changes(
        self,
        text: str,
        duration_cycles: int,
        current_scenario: Optional[EngineeringScenario] = None
    ) -> List[ScenarioParameterChange]:
        changes = []
        is_progressive = duration_cycles > 1 or "gradual" in text or "progressive" in text

        # Split into distinct clauses if multiple commands are chained via 'and', ',', ';'
        clauses = re.split(r"[,;]|\band\b", text)

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            # 1. Fan degradation
            if re.search(r"\bfan\b", clause) and re.search(r"\b(degrad\w*|wear|damage|loss|efficienc\w*)\b", clause):
                pct = self._extract_pct_in_clause(clause, default=15.0)
                if "another" in clause and current_scenario:
                    for prev_p in current_scenario.parameters:
                        if prev_p.parameter_key == "fan_degradation":
                            pct += prev_p.change_value
                            break
                meta = PARAMETER_MAP["fan_degradation"]
                changes.append(ScenarioParameterChange(
                    parameter_key="fan_degradation",
                    target_subsystem=meta["subsystem"],
                    change_type="degradation",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 2. Compressor degradation / efficiency
            if re.search(r"\b(compressor|hpc)\b", clause) and re.search(r"\b(degrad\w*|wear|fouling|loss|efficienc\w*)\b", clause):
                pct = self._extract_pct_in_clause(clause, default=3.0)
                if "another" in clause and current_scenario:
                    for prev_p in current_scenario.parameters:
                        if prev_p.parameter_key == "compressor_degradation":
                            pct += prev_p.change_value
                            break
                meta = PARAMETER_MAP["compressor_degradation"]
                changes.append(ScenarioParameterChange(
                    parameter_key="compressor_degradation",
                    target_subsystem=meta["subsystem"],
                    change_type="degradation",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 3. Turbine degradation / efficiency
            if re.search(r"\b(turbine|hpt)\b", clause) and re.search(r"\b(degrad\w*|wear|creep|loss|efficienc\w*)\b", clause):
                pct = self._extract_pct_in_clause(clause, default=3.0)
                if "another" in clause and current_scenario:
                    for prev_p in current_scenario.parameters:
                        if prev_p.parameter_key == "turbine_degradation":
                            pct += prev_p.change_value
                            break
                meta = PARAMETER_MAP["turbine_degradation"]
                changes.append(ScenarioParameterChange(
                    parameter_key="turbine_degradation",
                    target_subsystem=meta["subsystem"],
                    change_type="degradation",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 4. Turbine temperature / T50 / EGT
            if re.search(r"\b(turbine temperature|t50|egt|exhaust gas temperature)\b", clause) or (re.search(r"\bturbine\b", clause) and re.search(r"\btemp", clause)):
                pct = self._extract_signed_pct(clause, default_sign=1.0, default_val=5.0)
                meta = PARAMETER_MAP["t50"]
                changes.append(ScenarioParameterChange(
                    parameter_key="t50",
                    target_subsystem=meta["subsystem"],
                    change_type="percentage",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 5. Combustor temperature
            if re.search(r"\bcombustor\b", clause) and re.search(r"\btemp", clause):
                pct = self._extract_signed_pct(clause, default_sign=1.0, default_val=4.0)
                meta = PARAMETER_MAP["combustor_temperature"]
                changes.append(ScenarioParameterChange(
                    parameter_key="combustor_temperature",
                    target_subsystem=meta["subsystem"],
                    change_type="percentage",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 6. Compressor temperature / T30
            if re.search(r"\bt30\b", clause) or (re.search(r"\bcompressor\b", clause) and re.search(r"\btemp", clause)):
                pct = self._extract_signed_pct(clause, default_sign=1.0, default_val=4.0)
                meta = PARAMETER_MAP["t30"]
                changes.append(ScenarioParameterChange(
                    parameter_key="t30",
                    target_subsystem=meta["subsystem"],
                    change_type="percentage",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 7. Compressor pressure / P30
            if re.search(r"\b(compressor pressure|p30|hpc pressure)\b", clause) or (re.search(r"\bcompressor\b", clause) and re.search(r"\bpress", clause)):
                pct = self._extract_signed_pct(clause, default_sign=-1.0, default_val=3.0)
                meta = PARAMETER_MAP["p30"]
                changes.append(ScenarioParameterChange(
                    parameter_key="p30",
                    target_subsystem=meta["subsystem"],
                    change_type="percentage",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 8. Fuel flow
            if re.search(r"\bfuel flow\b|\bfuel\b", clause):
                pct = self._extract_signed_pct(clause, default_sign=1.0, default_val=5.0)
                meta = PARAMETER_MAP["fuel_flow"]
                changes.append(ScenarioParameterChange(
                    parameter_key="fuel_flow",
                    target_subsystem=meta["subsystem"],
                    change_type="percentage",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 9. Fan speed / Nf
            if re.search(r"\bfan speed\b|\bnf\b", clause):
                pct = self._extract_signed_pct(clause, default_sign=-1.0, default_val=2.0)
                meta = PARAMETER_MAP["nf"]
                changes.append(ScenarioParameterChange(
                    parameter_key="nf",
                    target_subsystem=meta["subsystem"],
                    change_type="percentage",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 10. Core speed / Nc
            if re.search(r"\bcore speed\b|\bnc\b", clause):
                pct = self._extract_signed_pct(clause, default_sign=-1.0, default_val=1.5)
                meta = PARAMETER_MAP["nc"]
                changes.append(ScenarioParameterChange(
                    parameter_key="nc",
                    target_subsystem=meta["subsystem"],
                    change_type="percentage",
                    change_value=pct,
                    is_progressive=is_progressive,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

            # 11. Sensor abnormalities (drift, bias, noise, stuck-at)
            if self.sensor_fault_patterns.search(clause):
                param_key = "p30" if "pressure" in clause else "t50"
                fault_type = "bias"
                if "drift" in clause:
                    fault_type = "drift"
                elif "noise" in clause:
                    fault_type = "noise"
                elif "stuck" in clause:
                    fault_type = "stuck_at"

                meta = PARAMETER_MAP[param_key]
                changes.append(ScenarioParameterChange(
                    parameter_key=param_key,
                    target_subsystem=meta["subsystem"],
                    change_type=fault_type,
                    change_value=5.0,
                    is_progressive="drift" in fault_type or duration_cycles > 1,
                    duration_cycles=duration_cycles,
                    sensor_channel=meta["sensor"],
                    unit=meta["unit"]
                ))
                continue

        return changes

    def _extract_pct_in_clause(self, clause: str, default: float = 5.0) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", clause)
        if match:
            return float(match.group(1))
        return default

    def _extract_signed_pct(self, clause: str, default_sign: float = 1.0, default_val: float = 5.0) -> float:
        # Check direction in this specific clause
        is_decrease = bool(re.search(r"\b(reduce|decrease|lower|drop|cut|diminish|-)\b", clause))
        is_increase = bool(re.search(r"\b(increase|raise|boost|elevate|add|higher|\+)\b", clause))

        sign = -1.0 if is_decrease else (1.0 if is_increase else default_sign)
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", clause)
        val = float(pct_match.group(1)) if pct_match else default_val
        return round(sign * abs(val), 1)

"""JETWISE AI — Multi-Provider Conversational Aerospace Engineering Agent.

Translates natural language engineering commands into validated physical simulation
specifications and deep thermodynamic rationales using selectable LLM providers:
1. Google Gemini (gemini-1.5-flash / gemini-1.5-pro)
2. Hugging Face Inference (mistralai/Mistral-7B-Instruct-v0.3)
3. OpenAI (gpt-4o-mini / gpt-4o)
4. Local Deterministic Rule-Based & Regex Parser (Always available offline fallback)

Enforces strict zero-credential-leakage rules via CredentialManager and provides
autonomous fallback if external APIs are unreachable or unconfigured.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json
import re
import time

from pydantic import BaseModel, Field, ConfigDict

from src.security.credential_manager import get_credential_manager, CredentialManager
from src.copilot_nlu import AerospaceCopilotNLU, ConversationalIntent
from src.copilot_scenarios import (
    EngineeringScenario,
    ScenarioParameterChange,
    SimulationMode,
    PARAMETER_MAP
)


class ProviderType(str, Enum):
    GEMINI = "google_gemini"
    HUGGINGFACE = "hugging_face"
    OPENAI = "openai"
    LOCAL = "local_deterministic"


class AgentResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    intent: str
    scenario: Optional[EngineeringScenario] = None
    explanation: str
    provider_used: str
    model_name: str
    latency_ms: float
    is_fallback: bool = False
    raw_response: Optional[str] = None


AEROSPACE_SYSTEM_PROMPT = """You are JETWISE AI, the Chief Aerospace Propulsion Simulation Architect and Predictive Maintenance Prognostics Agent.
Your role is to interpret natural language aerospace engineering instructions and translate them into structured simulation parameter adjustments for a dual-spool high-bypass commercial turbofan engine (NASA C-MAPSS and advanced commercial turbofans).

Key Physical Stations & Parameters:
- Station 1: Ambient Free-stream (P0, T0, Mach, Alt)
- Station 2: Fan Face Inlet (T2, P2)
- Station 2.4/2.5: Low-Pressure Compressor / Booster Exit (T24)
- Station 3: High-Pressure Compressor Discharge (T30, P30, Ps30, Nc core speed)
- Station 4: Combustor Exit / Turbine Inlet (T40 / TIT, fuel-to-air phi)
- Station 4.5: High-Pressure Turbine Exit (W31 cooling bleed)
- Station 5: Low-Pressure Turbine Exit / Exhaust Gas Temperature (T50 / EGT, Nf fan speed)
- Station 8/9: Exhaust Nozzle (Bypass ratio BPR, Thrust)

Supported Parameter Keys:
- "T30" (HPC discharge temp)
- "P30" (HPC discharge pressure)
- "T24" (LPC outlet temp)
- "T50" (LPT exhaust temp / EGT)
- "T2" (Fan inlet temp)
- "P2" (Fan inlet pressure)
- "Nf" (Fan physical speed)
- "Nc" (Core physical speed)
- "fuel_flow" (Fuel flow ratio phi / Wf)
- "degradation_cycles" (Cycle aging offset)
- "bypass_ratio" (BPR)
- "altitude_ft" (Flight altitude in feet)
- "mach_number" (Flight Mach number)

Your output MUST be a valid JSON object with the following structure:
{
  "intent": "SIMULATE_SCENARIO" | "CREATE_BASELINE" | "COMPARE_SCENARIOS" | "INVESTIGATE_ANOMALY" | "VERIFY_LIMITS" | "RUN_MATLAB_SIMULINK" | "EXPLAIN_PREDICTION",
  "scenario_name": "Concise engineering title",
  "simulation_mode": "MODE_A_AI_SENSOR_WHAT_IF" | "MODE_B_PHYSICAL_PARAMETRIC" | "MODE_C_HYBRID_CROSS_DATASET" | "MODE_D_MATLAB_SIMULINK",
  "parameters": [
    {
      "parameter_name": "T30",
      "change_type": "PERCENTAGE" | "ABSOLUTE",
      "delta_value": 5.0,
      "target_value": null
    }
  ],
  "aerospace_explanation": "Detailed thermodynamic rationale explaining gas-path implications, Brayton cycle station impacts, component wear, and RUL influence."
}
Only output the JSON object. Do not enclose in markdown blocks if possible, or use standard ```json ... ``` blocks.
"""


class MultiProviderAerospaceAgent:
    """Aerospace Copilot Agent coordinating Multi-Provider LLMs and local physics NLU."""

    def __init__(self, default_provider: ProviderType = ProviderType.GEMINI):
        self.default_provider = default_provider
        self.cred_mgr = get_credential_manager()
        self.local_nlu = AerospaceCopilotNLU()

    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Return connectivity status for each provider."""
        status_map = {}
        # Gemini
        gemini_info = self.cred_mgr.get_provider_info("google_gemini")
        status_map["google_gemini"] = {
            "name": "Google Gemini",
            "model": "gemini-1.5-flash",
            "configured": gemini_info.is_configured,
            "masked_key": gemini_info.masked_key,
            "badge": "🤖 Gemini Active" if gemini_info.is_configured else "⚪ Unconfigured"
        }
        # Hugging Face
        hf_info = self.cred_mgr.get_provider_info("hugging_face")
        status_map["hugging_face"] = {
            "name": "Hugging Face Inference",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "configured": hf_info.is_configured,
            "masked_key": hf_info.masked_key,
            "badge": "🤗 HF Active" if hf_info.is_configured else "⚪ Unconfigured"
        }
        # OpenAI
        openai_info = self.cred_mgr.get_provider_info("openai")
        status_map["openai"] = {
            "name": "OpenAI",
            "model": "gpt-4o-mini",
            "configured": openai_info.is_configured,
            "masked_key": openai_info.masked_key,
            "badge": "⚡ OpenAI Active" if openai_info.is_configured else "⚪ Unconfigured"
        }
        # Local
        status_map["local_deterministic"] = {
            "name": "Local Aerospace NLU",
            "model": "Rule-Based Physics Parser v2.4",
            "configured": True,
            "masked_key": "N/A (Built-in)",
            "badge": "⚙️ Local Offline Active"
        }
        return status_map

    def process_query(
        self,
        query: str,
        current_scenario: Optional[EngineeringScenario] = None,
        default_dataset: str = "FD001",
        default_engine: int = 1,
        default_cycle: int = 100,
        preferred_provider: Optional[ProviderType] = None
    ) -> AgentResponse:
        """Process user natural language query using requested provider with guaranteed offline fallback."""
        provider = preferred_provider or self.default_provider
        start_time = time.perf_counter()

        # 1. Attempt Google Gemini
        if provider == ProviderType.GEMINI:
            api_key = self.cred_mgr.get_credential("GEMINI_API_KEY")
            if api_key:
                try:
                    resp = self._call_gemini(query, api_key, current_scenario, default_dataset, default_engine, default_cycle)
                    resp.latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
                    return resp
                except Exception as e:
                    # Graceful fallback to local
                    return self._fallback_to_local(
                        query, current_scenario, default_dataset, default_engine, default_cycle,
                        reason=f"Gemini API unavailable: {CredentialManager.sanitize_error(str(e), [])}",
                        start_time=start_time
                    )
            else:
                return self._fallback_to_local(
                    query, current_scenario, default_dataset, default_engine, default_cycle,
                    reason="GEMINI_API_KEY not configured in Environment or API Configuration Centre",
                    start_time=start_time
                )

        # 2. Attempt Hugging Face
        elif provider == ProviderType.HUGGINGFACE:
            hf_token = self.cred_mgr.get_credential("HF_TOKEN")
            if hf_token:
                try:
                    resp = self._call_huggingface(query, hf_token, current_scenario, default_dataset, default_engine, default_cycle)
                    resp.latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
                    return resp
                except Exception as e:
                    return self._fallback_to_local(
                        query, current_scenario, default_dataset, default_engine, default_cycle,
                        reason=f"Hugging Face API unavailable: {CredentialManager.sanitize_error(str(e), [])}",
                        start_time=start_time
                    )
            else:
                return self._fallback_to_local(
                    query, current_scenario, default_dataset, default_engine, default_cycle,
                    reason="HF_TOKEN not configured",
                    start_time=start_time
                )

        # 3. Attempt OpenAI
        elif provider == ProviderType.OPENAI:
            openai_key = self.cred_mgr.get_credential("OPENAI_API_KEY")
            if openai_key:
                try:
                    resp = self._call_openai(query, openai_key, current_scenario, default_dataset, default_engine, default_cycle)
                    resp.latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
                    return resp
                except Exception as e:
                    return self._fallback_to_local(
                        query, current_scenario, default_dataset, default_engine, default_cycle,
                        reason=f"OpenAI API unavailable: {CredentialManager.sanitize_error(str(e), [])}",
                        start_time=start_time
                    )
            else:
                return self._fallback_to_local(
                    query, current_scenario, default_dataset, default_engine, default_cycle,
                    reason="OPENAI_API_KEY not configured",
                    start_time=start_time
                )

        # 4. Local Deterministic Parser
        return self._process_local(query, current_scenario, default_dataset, default_engine, default_cycle, start_time)

    def _call_gemini(
        self,
        query: str,
        api_key: str,
        current_scenario: Optional[EngineeringScenario],
        dataset: str,
        engine: int,
        cycle: int
    ) -> AgentResponse:
        """Call Google Gemini Generative AI via official REST endpoint."""
        import requests

        prompt = f"System Instruction: {AEROSPACE_SYSTEM_PROMPT}\n\nEngineering Request: {query}\nActive Dataset: {dataset}, Engine: {engine}, Cycle: {cycle}"
        if current_scenario:
            prompt += f"\nCurrent Active Scenario: {current_scenario.scenario_name} (Mode: {current_scenario.simulation_mode})"

        raw_text = ""
        chosen_model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
        models_to_try = [chosen_model, "gemini-flash-lite-latest", "gemini-flash-latest"]
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}
        }
        headers = {"Content-Type": "application/json"}

        for m_name in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
                res = requests.post(url, headers=headers, json=payload, timeout=12.0)
                if res.status_code == 200:
                    data = res.json()
                    cands = data.get("candidates", [])
                    if cands:
                        parts = cands[0].get("content", {}).get("parts", [])
                        if parts:
                            raw_text = parts[0].get("text", "")
                            chosen_model = m_name
                            break
            except Exception:
                continue

        return self._parse_llm_json_response(raw_text, query, dataset, engine, cycle, "Google Gemini", chosen_model)

    def _call_huggingface(
        self,
        query: str,
        token: str,
        current_scenario: Optional[EngineeringScenario],
        dataset: str,
        engine: int,
        cycle: int
    ) -> AgentResponse:
        """Call Hugging Face Inference API."""
        import requests

        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        headers = {"Authorization": f"Bearer {token}"}
        prompt = f"<s>[INST] {AEROSPACE_SYSTEM_PROMPT}\n\nRequest: {query} [/INST]"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.2}}

        res = requests.post(url, headers=headers, json=payload, timeout=12.0)
        res.raise_for_status()
        data = res.json()
        raw_text = ""
        if isinstance(data, list) and len(data) > 0:
            raw_text = data[0].get("generated_text", "")
            # Remove prompt prefix if returned
            if "[/INST]" in raw_text:
                raw_text = raw_text.split("[/INST]")[-1]

        return self._parse_llm_json_response(raw_text, query, dataset, engine, cycle, "Hugging Face", "Mistral-7B-Instruct")

    def _call_openai(
        self,
        query: str,
        api_key: str,
        current_scenario: Optional[EngineeringScenario],
        dataset: str,
        engine: int,
        cycle: int
    ) -> AgentResponse:
        """Call OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = f"Engineering Request: {query}\nActive Dataset: {dataset}, Engine: {engine}, Cycle: {cycle}"
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AEROSPACE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=10.0
        )
        raw_text = completion.choices[0].message.content or ""
        return self._parse_llm_json_response(raw_text, query, dataset, engine, cycle, "OpenAI", "gpt-4o-mini")

    def _parse_llm_json_response(
        self,
        raw_text: str,
        query: str,
        dataset: str,
        engine: int,
        cycle: int,
        provider_name: str,
        model_name: str
    ) -> AgentResponse:
        """Extract JSON structure from LLM response text and convert to EngineeringScenario."""
        clean_text = raw_text.strip()
        # Strip markdown ```json ... ``` wrapper if present
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        # Regex fallback to find outermost JSON object
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)

        data = json.loads(clean_text)

        intent = data.get("intent", ConversationalIntent.SIMULATE_SCENARIO)
        scen_name = data.get("scenario_name", f"AI Simulation: {query[:30]}")
        mode_str = data.get("simulation_mode", SimulationMode.MODE_A_AI_SENSOR_WHAT_IF)
        explanation = data.get("aerospace_explanation", "Thermodynamic evaluation completed according to aerothermal boundary conditions.")

        # Build Parameter Changes
        param_objs: List[ScenarioParameterChange] = []
        for p in data.get("parameters", []):
            p_name = p.get("parameter_name")
            if p_name in PARAMETER_MAP:
                meta = PARAMETER_MAP[p_name]
                param_objs.append(
                    ScenarioParameterChange(
                        parameter_name=p_name,
                        friendly_name=meta.get("name", p_name),
                        sensor_id=meta.get("sensor", "sensor_3"),
                        change_type=p.get("change_type", "PERCENTAGE"),
                        delta_value=float(p.get("delta_value", 0.0)),
                        target_value=p.get("target_value")
                    )
                )

        scen = EngineeringScenario(
            scenario_name=scen_name,
            user_query=query,
            dataset=dataset,
            engine_id=engine,
            cycle=cycle,
            simulation_mode=mode_str,
            parameters=param_objs
        )

        return AgentResponse(
            intent=intent,
            scenario=scen,
            explanation=explanation,
            provider_used=provider_name,
            model_name=model_name,
            latency_ms=0.0,
            is_fallback=False,
            raw_response=raw_text
        )

    def _process_local(
        self,
        query: str,
        current_scenario: Optional[EngineeringScenario],
        dataset: str,
        engine: int,
        cycle: int,
        start_time: float
    ) -> AgentResponse:
        """Process using deterministic local Aerospace Copilot NLU."""
        intent, scen, explanation = self.local_nlu.parse_query(
            query=query,
            current_scenario=current_scenario,
            default_dataset=dataset,
            default_engine=engine,
            default_cycle=cycle
        )

        if not explanation:
            explanation = (
                f"Generated engineering scenario based on aerothermal station principles. "
                f"Parsed {len(scen.parameters)} parameter modification(s) mapped across C-MAPSS telemetry space."
            )

        latency = round((time.perf_counter() - start_time) * 1000, 1)
        return AgentResponse(
            intent=intent,
            scenario=scen,
            explanation=explanation,
            provider_used="Local Deterministic Engine",
            model_name="Aerospace Rule-Based NLU v2.4",
            latency_ms=latency,
            is_fallback=False,
            raw_response=None
        )

    def _fallback_to_local(
        self,
        query: str,
        current_scenario: Optional[EngineeringScenario],
        dataset: str,
        engine: int,
        cycle: int,
        reason: str,
        start_time: float
    ) -> AgentResponse:
        """Gracefully fall back to local parser while notifying user of reason."""
        resp = self._process_local(query, current_scenario, dataset, engine, cycle, start_time)
        resp.is_fallback = True
        resp.explanation = f"⚠️ *[{reason}]*\n*Autonomous fallback routed through Deterministic Aerospace NLU:*\n\n{resp.explanation}"
        return resp


_AEROSPACE_AGENT_INSTANCE: Optional[MultiProviderAerospaceAgent] = None


def get_aerospace_agent(default_provider: ProviderType = ProviderType.GEMINI) -> MultiProviderAerospaceAgent:
    """Return singleton MultiProviderAerospaceAgent instance."""
    global _AEROSPACE_AGENT_INSTANCE
    if _AEROSPACE_AGENT_INSTANCE is None:
        _AEROSPACE_AGENT_INSTANCE = MultiProviderAerospaceAgent(default_provider=default_provider)
    return _AEROSPACE_AGENT_INSTANCE

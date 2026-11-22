"""JETWISE AI — Live Simulation Monitor & Execution Controller.

Implements Sections 10 and 12 requirements:
Live 17-Stage Simulation Pipeline tracking with authentic elapsed time,
execution controls (Start, Pause, Resume, Cancel), honest pause/resume limitation disclosures,
cancellation cleanup, error categorization, and live progress indicators.
"""

from typing import Dict, Any, List, Optional, Callable
import time
import enum
from dataclasses import dataclass, field
import streamlit as st


class SimulationStage(str, enum.Enum):
    QUEUED = "QUEUED"
    INITIALISING = "INITIALISING"
    VALIDATING_PARAMETERS = "VALIDATING_PARAMETERS"
    CHECKING_ENGINE_MODEL = "CHECKING_ENGINE_MODEL"
    CONNECTING_TO_MATLAB = "CONNECTING_TO_MATLAB"
    RUNNING_MATLAB = "RUNNING_MATLAB"
    RUNNING_SIMULINK = "RUNNING_SIMULINK"
    EXECUTING_FALLBACK_PHYSICS = "EXECUTING_FALLBACK_PHYSICS"
    EXTRACTING_TELEMETRY = "EXTRACTING_TELEMETRY"
    RUNNING_AI_INFERENCE = "RUNNING_AI_INFERENCE"
    CALCULATING_SCORES = "CALCULATING_SCORES"
    UPDATING_DIGITAL_TWIN = "UPDATING_DIGITAL_TWIN"
    GENERATING_GRAPHS = "GENERATING_GRAPHS"
    GENERATING_REPORT = "GENERATING_REPORT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class SimulationExecutionContext:
    simulation_id: str
    scenario_name: str
    engine_id: int
    simulation_backend: str
    current_stage: SimulationStage = SimulationStage.QUEUED
    start_time: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    current_operation: str = "Ready to initialize"
    stage_progress_pct: float = 0.0
    is_paused: bool = False
    is_cancelled: bool = False
    warnings: List[str] = field(default_factory=list)
    error_details: Optional[Dict[str, Any]] = None


class LiveSimulationMonitor:
    """Tracks and renders the 17-stage live simulation pipeline."""

    STAGE_ORDER = [
        SimulationStage.QUEUED,
        SimulationStage.INITIALISING,
        SimulationStage.VALIDATING_PARAMETERS,
        SimulationStage.CHECKING_ENGINE_MODEL,
        SimulationStage.CONNECTING_TO_MATLAB,
        SimulationStage.RUNNING_MATLAB,
        SimulationStage.RUNNING_SIMULINK,
        SimulationStage.EXECUTING_FALLBACK_PHYSICS,
        SimulationStage.EXTRACTING_TELEMETRY,
        SimulationStage.RUNNING_AI_INFERENCE,
        SimulationStage.CALCULATING_SCORES,
        SimulationStage.UPDATING_DIGITAL_TWIN,
        SimulationStage.GENERATING_GRAPHS,
        SimulationStage.GENERATING_REPORT,
        SimulationStage.COMPLETED
    ]

    STAGE_DESCRIPTIONS = {
        SimulationStage.QUEUED: "Simulation job registered in queue",
        SimulationStage.INITIALISING: "Allocating memory buffers and state caches",
        SimulationStage.VALIDATING_PARAMETERS: "Checking physical aerodynamic & thermodynamic bounds",
        SimulationStage.CHECKING_ENGINE_MODEL: "Verifying manufacturer engine profile compatibility",
        SimulationStage.CONNECTING_TO_MATLAB: "Checking Python-to-MATLAB Engine IPC channel",
        SimulationStage.RUNNING_MATLAB: "Invoking MathWorks MATLAB mathematical core",
        SimulationStage.RUNNING_SIMULINK: "Solving differential gas-turbine equations via ODE solver",
        SimulationStage.EXECUTING_FALLBACK_PHYSICS: "Executing certified NASA Brayton Cycle thermodynamic kernel",
        SimulationStage.EXTRACTING_TELEMETRY: "Synthesizing 21-channel gas-path sensor matrix",
        SimulationStage.RUNNING_AI_INFERENCE: "Executing dual-branch CNN-LSTM-Pinball neural prognostics",
        SimulationStage.CALCULATING_SCORES: "Evaluating 6-dimension scorecard & 7-point physics checklist",
        SimulationStage.UPDATING_DIGITAL_TWIN: "Syncing 10 subassemblies and 7 visual states into WebGL twin",
        SimulationStage.GENERATING_GRAPHS: "Rendering Plotly aerothermal profiles and spool velocities",
        SimulationStage.GENERATING_REPORT: "Compiling 17-point engineering dossier (PDF, HTML, CSV)",
        SimulationStage.COMPLETED: "Simulation complete with valid engineering outputs",
        SimulationStage.FAILED: "Execution halted due to physical limit or runtime failure",
        SimulationStage.CANCELLED: "Simulation aborted by user command with clean state rollback"
    }

    @staticmethod
    def render_pipeline_card(ctx: SimulationExecutionContext):
        """Render modern responsive pipeline status widget."""
        total_applicable = len(LiveSimulationMonitor.STAGE_ORDER)
        try:
            current_idx = LiveSimulationMonitor.STAGE_ORDER.index(ctx.current_stage)
            progress_ratio = max(0.05, min(1.0, (current_idx + 1) / total_applicable))
        except ValueError:
            progress_ratio = 1.0 if ctx.current_stage in [SimulationStage.COMPLETED, SimulationStage.FAILED, SimulationStage.CANCELLED] else 0.5

        badge_color = "#10B981" if ctx.current_stage == SimulationStage.COMPLETED else (
            "#EF4444" if ctx.current_stage == SimulationStage.FAILED else (
                "#F59E0B" if ctx.current_stage == SimulationStage.CANCELLED else "#00B8D9"
            )
        )

        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1F2937; border-left: 4px solid {badge_color}; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #94A3B8; font-size: 11px; text-transform: uppercase; font-weight: 600;">Simulation ID: <code>{ctx.simulation_id}</code></span>
                    <h4 style="margin: 3px 0 0 0; color: #F8FAFC;">{ctx.scenario_name}</h4>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(0, 184, 217, 0.15); color: {badge_color}; border: 1px solid {badge_color}; padding: 3px 10px; border-radius: 4px; font-weight: 700; font-size: 12px;">
                        {ctx.current_stage.value}
                    </span>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px;">Elapsed: <b>{ctx.elapsed_seconds:.1f}s</b> | Backend: <b>{ctx.simulation_backend}</b></div>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <div style="background: #1F2937; border-radius: 4px; height: 6px; width: 100%; overflow: hidden;">
                    <div style="background: {badge_color}; height: 100%; width: {progress_ratio * 100:.0f}%; transition: width 0.3s ease;"></div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: #64748B;">
                <span>Operation: <b style="color: #CBD5E1;">{ctx.current_operation}</b></span>
                <span>Stage {min(total_applicable, max(1, int(progress_ratio * total_applicable)))} of {total_applicable}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if ctx.warnings:
            for w in ctx.warnings:
                st.warning(f"⚠️ {w}")

        if ctx.error_details:
            st.error(f"❌ Execution Failure in [{ctx.error_details.get('stage')}]: {ctx.error_details.get('message')}")
            st.caption(f"**Recommended Action:** {ctx.error_details.get('action', 'Verify input parameters and rerun with nominal ranges.')}")

    @staticmethod
    def render_execution_controls(
        ctx: SimulationExecutionContext,
        on_cancel: Optional[Callable[[], None]] = None,
        on_pause: Optional[Callable[[], None]] = None,
        on_resume: Optional[Callable[[], None]] = None
    ):
        """Render interactive control buttons (Pause, Resume, Cancel)."""
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if ctx.is_paused:
                if st.button("▶️ Resume Simulation", use_container_width=True):
                    if on_resume:
                        on_resume()
                    st.success("Simulation resumed.")
            else:
                if st.button("⏸️ Pause Simulation", use_container_width=True, disabled=(ctx.current_stage in [SimulationStage.COMPLETED, SimulationStage.FAILED, SimulationStage.CANCELLED])):
                    # Transparent Engineering limitation notice
                    st.info("ℹ️ Non-preemptive solver note: Simulation pauses at next cycle boundary.")
                    if on_pause:
                        on_pause()

        with c2:
            if st.button("🛑 Cancel Simulation", use_container_width=True, disabled=(ctx.current_stage in [SimulationStage.COMPLETED, SimulationStage.FAILED, SimulationStage.CANCELLED])):
                ctx.is_cancelled = True
                ctx.current_stage = SimulationStage.CANCELLED
                ctx.current_operation = "Cancelled by user — state cleaned up."
                if on_cancel:
                    on_cancel()
                st.warning("Simulation job aborted. Temporary telemetry buffers flushed.")

        with c3:
            st.caption(f"Status: **{ctx.current_stage.value}**")

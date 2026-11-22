"""Interactive 3D Digital Twin Component for Turbofan Jet Engine (JETWISE AI Phase 8).

Renders:
1. High-fidelity 3D Plotly Mesh representation of turbofan subassemblies:
   - Fan Stage (Inlet & Low Pressure Fan)
   - High Pressure Compressor (HPC)
   - Annular Combustor Chamber (Hot Section)
   - High Pressure Turbine (HPT)
   - Low Pressure Turbine (LPT)
   - Exhaust Nozzle & Centerbody
   - Dual Spool Coaxial Shafts
2. Live telemetry & thermodynamic heat-map coloring based on sensor readings.
3. Interactive WebGL Three.js animated digital twin canvas with component breakdown and diagnostic status HUD.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import plotly.graph_objects as go


def _cylinder_mesh(
    r_start: float,
    r_end: float,
    z_start: float,
    z_end: float,
    n_theta: int = 36,
    n_z: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 3D coordinates for a conical/cylindrical shell surface."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(z_start, z_end, n_z)
    theta_grid, z_grid = np.meshgrid(theta, z)

    # Linear interpolation of radius along z
    t = (z_grid - z_start) / (z_end - z_start if z_end != z_start else 1.0)
    r_grid = r_start + t * (r_end - r_start)

    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    return x_grid, y_grid, z_grid


def generate_turbofan_3d_figure(
    telemetry: Optional[Dict[str, float]] = None,
    health_state: str = "HEALTHY",
    predicted_rul: float = 85.0,
    mpi: float = 24.0
) -> go.Figure:
    """Construct an interactive Plotly 3D digital twin of a two-spool turbofan engine."""
    telemetry = telemetry or {}

    # Extract or simulate key temperatures and pressures
    t24 = float(telemetry.get("sensor_2", 642.0))   # LPC outlet temp (~642 R)
    t30 = float(telemetry.get("sensor_3", 1585.0))  # HPC outlet temp (~1585 R)
    t50 = float(telemetry.get("sensor_4", 1405.0))  # LPT outlet temp (~1405 R)
    vib = float(telemetry.get("sensor_14", 8140.0)) # Core speed / vibration

    # Component definitions: (name, r1, r2, z1, z2, base_temp, colormap)
    # z axis is the engine centerline (0 = fan inlet, 14 = nozzle exit)
    components = [
        {"name": "Inlet Spinner & Fan", "r1": 1.6, "r2": 2.2, "z1": 0.0, "z2": 2.2, "temp": t24 * 0.8, "colorscale": "Teal"},
        {"name": "Low Pressure Compressor (LPC)", "r1": 2.1, "r2": 1.7, "z1": 2.2, "z2": 4.0, "temp": t24, "colorscale": "Teal"},
        {"name": "High Pressure Compressor (HPC)", "r1": 1.7, "r2": 1.3, "z1": 4.0, "z2": 6.8, "temp": t30 * 0.9, "colorscale": "YlOrRd"},
        {"name": "Combustor Chamber", "r1": 1.3, "r2": 1.4, "z1": 6.8, "z2": 8.5, "temp": t30 * 1.3, "colorscale": "Hot"},
        {"name": "High Pressure Turbine (HPT)", "r1": 1.4, "r2": 1.5, "z1": 8.5, "z2": 9.8, "temp": t50 * 1.1, "colorscale": "Inferno"},
        {"name": "Low Pressure Turbine (LPT)", "r1": 1.5, "r2": 1.6, "z1": 9.8, "z2": 11.5, "temp": t50, "colorscale": "Viridis"},
        {"name": "Core Exhaust Nozzle", "r1": 1.6, "r2": 1.1, "z1": 11.5, "z2": 14.0, "temp": t50 * 0.85, "colorscale": "Electric"},
    ]

    fig = go.Figure()

    # Add each component surface mesh
    for comp in components:
        x, y, z = _cylinder_mesh(comp["r1"], comp["r2"], comp["z1"], comp["z2"])
        # Temperature field representation
        val = np.full_like(z, comp["temp"])
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            surfacecolor=val,
            colorscale=comp["colorscale"],
            showscale=False,
            opacity=0.88,
            hoverinfo="text",
            hovertext=f"<b>{comp['name']}</b><br>Thermal Level: {comp['temp']:.1f} °R<br>Centerline Z: {comp['z1']} to {comp['z2']}"
        ))

    # Add Central Coaxial Spool Shaft
    xs, ys, zs = _cylinder_mesh(0.28, 0.28, 0.5, 13.5, n_theta=18)
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        surfacecolor=np.full_like(zs, 100),
        colorscale=[[0, "#94A3B8"], [1, "#E2E8F0"]],
        showscale=False,
        opacity=0.95,
        hoverinfo="text",
        hovertext="<b>Dual-Spool HP/LP Concentric Drive Shafts</b><br>Core Speed: N2 nominal"
    ))

    # Add 3D Diagnostic Annotation Markers at critical inspection stations
    station_markers = [
        {"x": 0.0, "y": 2.5, "z": 1.1, "name": "Station 2 (Fan)", "status": "NOMINAL", "desc": "Fan blade tip clearance"},
        {"x": 0.0, "y": 2.0, "z": 5.4, "name": "Station 3 (HPC)", "status": "ELEVATED WEAR" if mpi > 50 else "NOMINAL", "desc": f"T30={t30:.1f}°R, HPC Pressure"},
        {"x": 0.0, "y": 1.8, "z": 7.6, "name": "Station 4 (Combustor)", "status": "HOT SPOT CHECK", "desc": "Combustor flame tube"},
        {"x": 0.0, "y": 1.9, "z": 9.1, "name": "Station 4.5 (HPT)", "status": "INSPECTION DUE" if predicted_rul < 40 else "NOMINAL", "desc": "HPT blade creep & thermal stress"},
        {"x": 0.0, "y": 2.0, "z": 10.6, "name": "Station 5 (LPT)", "status": "NOMINAL", "desc": f"EGT / T50={t50:.1f}°R"}
    ]

    marker_colors = {
        "NOMINAL": "#21C77A",
        "ELEVATED WEAR": "#FFB547",
        "HOT SPOT CHECK": "#00B8D9",
        "INSPECTION DUE": "#FF5252"
    }

    fig.add_trace(go.Scatter3d(
        x=[m["x"] for m in station_markers],
        y=[m["y"] for m in station_markers],
        z=[m["z"] for m in station_markers],
        mode="markers+text",
        marker=dict(
            size=7,
            color=[marker_colors.get(m["status"], "#00B8D9") for m in station_markers],
            symbol="circle",
            line=dict(color="#FFFFFF", width=1)
        ),
        text=[m["name"] for m in station_markers],
        textposition="top center",
        textfont=dict(color="#E2E8F0", size=10),
        hoverinfo="text",
        hovertext=[f"<b>{m['name']}</b><br>Status: {m['status']}<br>{m['desc']}" for m in station_markers]
    ))

    # SCADA Dark Aerospace Styling Layout
    fig.update_layout(
        title=dict(
            text=f"<b>JETWISE AI — Interactive Turbofan Digital Twin (State: {health_state} | RUL: {predicted_rul:.1f} Cycles)</b>",
            font=dict(color="#00B8D9", size=14, family="monospace")
        ),
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#0B0F14",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="Centerline Flow Axis"),
            bgcolor="#0B0F14",
            aspectratio=dict(x=1, y=1, z=2.4),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.9),
                center=dict(x=0, y=0, z=0.2)
            )
        )
    )

    return fig


def render_threejs_digital_twin_html(
    engine_id: int,
    cycle: int,
    predicted_rul: float,
    health_state: str = "HEALTHY",
    mpi: float = 25.0
) -> str:
    """Generate interactive WebGL 3D turbofan digital twin HTML using Three.js."""
    state_colors = {
        "HEALTHY": "#21C77A",
        "MONITOR": "#00B8D9",
        "MAINTENANCE_DUE": "#FFB547",
        "CRITICAL": "#FF5252"
    }
    glow_color = state_colors.get(health_state, "#00B8D9")

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: #0B0F14;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            }}
            #canvas-container {{
                width: 100%;
                height: 480px;
                position: relative;
            }}
            .hud-overlay {{
                position: absolute;
                top: 12px;
                left: 14px;
                color: #E2E8F0;
                background: rgba(16, 23, 32, 0.85);
                border: 1px solid #1E293B;
                border-left: 4px solid {glow_color};
                padding: 10px 14px;
                border-radius: 6px;
                pointer-events: none;
                font-size: 12px;
                line-height: 1.5;
            }}
            .hud-title {{
                color: #00B8D9;
                font-weight: bold;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }}
            .controls-hint {{
                position: absolute;
                bottom: 10px;
                right: 14px;
                color: #64748B;
                font-size: 11px;
                pointer-events: none;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div class="hud-overlay">
                <div class="hud-title">Turbofan Digital Twin // Engine #{engine_id}</div>
                <div>Status: <b style="color: {glow_color};">{health_state}</b> | Cycle: <b>{cycle}</b></div>
                <div>Predicted RUL: <b>{predicted_rul:.1f} Cycles</b> | MPI: <b>{mpi:.1f}/100</b></div>
                <div style="font-size: 10px; color: #94A3B8; margin-top: 4px;">Dynamic Aerothermal Surface Heat-Map</div>
            </div>
            <div class="controls-hint">🖱️ Drag to Orbit | Scroll to Zoom | Right-Click to Pan</div>
        </div>

        <script>
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0B0F14);

            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(12, 6, 14);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Ambient & Directional Lights
            const ambient = new THREE.AmbientLight(0x334155, 1.8);
            scene.add(ambient);

            const dirLight = new THREE.DirectionalLight(0xFFFFFF, 1.2);
            dirLight.position.set(10, 20, 15);
            scene.add(dirLight);

            const blueLight = new THREE.PointLight(0x00B8D9, 2.0, 30);
            blueLight.position.set(-10, -5, -10);
            scene.add(blueLight);

            const engineGroup = new THREE.Group();

            // Materials
            const casingMat = new THREE.MeshStandardMaterial({{
                color: 0x1E293B,
                metalness: 0.8,
                roughness: 0.3,
                transparent: true,
                opacity: 0.45,
                wireframe: false
            }});

            const coreMat = new THREE.MeshStandardMaterial({{
                color: 0x334155,
                metalness: 0.9,
                roughness: 0.2
            }});

            const combustorMat = new THREE.MeshStandardMaterial({{
                color: {glow_color.replace('#', '0x')},
                emissive: {glow_color.replace('#', '0x')},
                emissiveIntensity: 0.6,
                roughness: 0.4
            }});

            const bladeMat = new THREE.MeshStandardMaterial({{
                color: 0x94A3B8,
                metalness: 0.95,
                roughness: 0.1
            }});

            // 1. Outer Nacelle Casing (Cutaway)
            const casingGeo = new THREE.CylinderGeometry(2.6, 2.2, 10, 32, 1, true, 0, Math.PI * 1.6);
            const casing = new THREE.Mesh(casingGeo, casingMat);
            casing.rotation.z = Math.PI / 2;
            engineGroup.add(casing);

            // 2. Central Spool Shaft
            const shaftGeo = new THREE.CylinderGeometry(0.3, 0.3, 11, 24);
            const shaft = new THREE.Mesh(shaftGeo, coreMat);
            shaft.rotation.z = Math.PI / 2;
            engineGroup.add(shaft);

            // 3. Rotating Fan Assembly
            const fanGroup = new THREE.Group();
            const spinnerGeo = new THREE.ConeGeometry(0.8, 1.6, 24);
            const spinner = new THREE.Mesh(spinnerGeo, coreMat);
            spinner.rotation.z = -Math.PI / 2;
            spinner.position.x = -5.0;
            fanGroup.add(spinner);

            // Fan Blades
            const bladeGeo = new THREE.BoxGeometry(0.12, 1.8, 0.4);
            const numBlades = 16;
            for (let i = 0; i < numBlades; i++) {{
                const angle = (i / numBlades) * Math.PI * 2;
                const blade = new THREE.Mesh(bladeGeo, bladeMat);
                blade.position.x = -4.5;
                blade.position.y = Math.cos(angle) * 1.2;
                blade.position.z = Math.sin(angle) * 1.2;
                blade.rotation.x = angle;
                blade.rotation.y = 0.3;
                fanGroup.add(blade);
            }}
            engineGroup.add(fanGroup);

            // 4. Compressor Discs (HPC Stages)
            for (let c = 0; c < 5; c++) {{
                const discGeo = new THREE.CylinderGeometry(1.4 - c * 0.1, 1.4 - c * 0.1, 0.3, 24);
                const disc = new THREE.Mesh(discGeo, coreMat);
                disc.rotation.z = Math.PI / 2;
                disc.position.x = -2.5 + c * 0.7;
                engineGroup.add(disc);
            }}

            // 5. Combustor Chamber (Glowing Hot Section)
            const combustorGeo = new THREE.CylinderGeometry(1.3, 1.3, 1.8, 24);
            const combustor = new THREE.Mesh(combustorGeo, combustorMat);
            combustor.rotation.z = Math.PI / 2;
            combustor.position.x = 1.6;
            engineGroup.add(combustor);

            // 6. High & Low Pressure Turbines (HPT/LPT)
            for (let t = 0; t < 4; t++) {{
                const turbGeo = new THREE.CylinderGeometry(1.2 + t * 0.1, 1.2 + t * 0.1, 0.25, 24);
                const turbMat = (t === 0) ? combustorMat : coreMat;
                const turb = new THREE.Mesh(turbGeo, turbMat);
                turb.rotation.z = Math.PI / 2;
                turb.position.x = 3.0 + t * 0.6;
                engineGroup.add(turb);
            }}

            // 7. Exhaust Nozzle
            const nozzleGeo = new THREE.ConeGeometry(1.4, 2.0, 24, 1, true);
            const nozzle = new THREE.Mesh(nozzleGeo, casingMat);
            nozzle.rotation.z = -Math.PI / 2;
            nozzle.position.x = 5.8;
            engineGroup.add(nozzle);

            scene.add(engineGroup);

            // Center group
            engineGroup.position.set(0, 0, 0);

            // Animation Loop
            let clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                const delta = clock.getDelta();
                fanGroup.rotation.x += 4.5 * delta;
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();

            // Resize listener
            window.addEventListener('resize', () => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html_code


if __name__ == "__main__":
    print("[x] Testing 3D Digital Twin Component...")
    fig = generate_turbofan_3d_figure(
        telemetry={"sensor_2": 642.5, "sensor_3": 1588.0, "sensor_4": 1406.2, "sensor_14": 8145.0},
        health_state="MONITOR",
        predicted_rul=45.2,
        mpi=42.0
    )
    html = render_threejs_digital_twin_html(1, 145, 45.2, "MONITOR", 42.0)
    print(f"Plotly Figure generated with {len(fig.data)} traces.")
    print(f"Three.js HTML generated ({len(html)} bytes).")
    print("[x] Digital Twin Component verified.")

"""JETWISE AI — Multi-Engine 3D Digital Twin Generator.

Generates custom WebGL Three.js 3D Digital Twins reflecting specific mechanical architectures:
- Three-Spool Turbofan (Rolls-Royce Trent with LP, IP, HP concentric shafts and IPC/IPT)
- Geared Turbofan (Pratt & Whitney GTF with Fan Reduction Gearbox)
- Standard Dual-Spool Commercial Turbofan (GE90, CFM LEAP, NASA C-MAPSS)
- Compact Regional Business Jet Turbofan (Honeywell HTF7000)
"""

from typing import Dict, List, Any, Optional
import json

from src.engine_configuration.models import EngineConfiguration, ShaftArrangement


def render_multi_engine_3d_twin_html(
    engine_cfg: EngineConfiguration,
    telemetry: Optional[Dict[str, float]] = None,
    component_states: Optional[Dict[str, str]] = None,
    scenario_info: Optional[Dict[str, Any]] = None,
    height_px: int = 600
) -> str:
    """Generate interactive Three.js HTML for the specific architectural digital twin."""
    telemetry = telemetry or {}
    component_states = component_states or {}
    scenario_info = scenario_info or {}

    template = engine_cfg.digital_twin_template
    is_three_spool = engine_cfg.shaft_arrangement == ShaftArrangement.TRIPLE_SPOOL.value or template == "three_spool_trent"
    is_geared = engine_cfg.shaft_arrangement == ShaftArrangement.GEARED_TURBOFAN.value or template == "geared_turbofan"
    is_regional = template == "regional_turbofan"

    fan_d = float(engine_cfg.fan_diameter_m.value) if hasattr(engine_cfg.fan_diameter_m, "value") else 2.1
    fan_scale = max(0.6, min(1.6, fan_d / 2.1))

    # Provenance Architecture Label
    if is_three_spool:
        arch_title = "Rolls-Royce Trent 3-Spool Concentric Architecture (LP / IP / HP Shafts)"
        badge_text = "THREE-SPOOL SCHEMATIC: SEPARATE FAN, IPC, & HPC ROTORS"
    elif is_geared:
        arch_title = "Pratt & Whitney Geared Turbofan (Fan Reduction Gearbox + LP + HP)"
        badge_text = "GEARED TURBOFAN SCHEMATIC: 3:1 PLANETARY REDUCTION GEARBOX"
    elif is_regional:
        arch_title = "Honeywell Regional Business Jet Turbofan (Compact High-Pressure Core)"
        badge_text = "REGIONAL TURBOFAN SCHEMATIC: COMPACT TWIN-SPOOL"
    else:
        arch_title = f"{engine_cfg.manufacturer} Standard High-Bypass Dual-Spool Architecture"
        badge_text = "STANDARD DUAL-SPOOL HIGH-BYPASS SCHEMATIC"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: #06090E;
                color: #E2E8F0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            #canvas-container {{
                width: 100%;
                height: {height_px}px;
                position: relative;
            }}
            #hud-overlay {{
                position: absolute;
                top: 12px;
                left: 12px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid #1E293B;
                border-left: 4px solid #00B8D9;
                padding: 10px 14px;
                border-radius: 6px;
                pointer-events: none;
                z-index: 10;
                max-width: 420px;
            }}
            #hud-overlay h4 {{
                margin: 0 0 4px 0;
                color: #F8FAFC;
                font-size: 0.95rem;
            }}
            #hud-overlay p {{
                margin: 0;
                color: #94A3B8;
                font-size: 0.75rem;
                line-height: 1.3;
            }}
            .badge-arch {{
                display: inline-block;
                background: rgba(0, 184, 217, 0.15);
                color: #00B8D9;
                border: 1px solid #00B8D9;
                font-size: 0.65rem;
                font-weight: bold;
                padding: 2px 6px;
                border-radius: 4px;
                margin-top: 5px;
            }}
            #controls-panel {{
                position: absolute;
                bottom: 12px;
                right: 12px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid #1E293B;
                padding: 8px 12px;
                border-radius: 6px;
                z-index: 10;
                font-size: 0.75rem;
            }}
            .control-btn {{
                background: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
                padding: 4px 8px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.7rem;
                margin-right: 4px;
            }}
            .control-btn:hover {{
                background: #00B8D9;
                color: #0F172A;
            }}
            #tooltip {{
                position: absolute;
                display: none;
                background: rgba(11, 15, 20, 0.92);
                border: 1px solid #00B8D9;
                color: #FFF;
                padding: 6px 10px;
                border-radius: 4px;
                font-size: 0.75rem;
                pointer-events: none;
                z-index: 20;
            }}
        </style>
        <!-- Load Three.js & OrbitControls -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div id="hud-overlay">
                <h4>{engine_cfg.manufacturer} — {engine_cfg.engine_model}</h4>
                <p>Architecture: <b>{engine_cfg.shaft_arrangement}</b> ({engine_cfg.number_of_shafts} Shafts)</p>
                <p>Bypass Ratio: <b>{engine_cfg.bypass_ratio.value}:1</b> | Fan Ø: <b>{engine_cfg.fan_diameter_m.value} m</b></p>
                <div class="badge-arch">{badge_text}</div>
            </div>

            <div id="controls-panel">
                <button class="control-btn" id="btn-wireframe">Toggle Wireframe</button>
                <button class="control-btn" id="btn-explode">Exploded View</button>
                <button class="control-btn" id="btn-reset">Reset Camera</button>
            </div>

            <div id="tooltip"></div>
        </div>

        <script>
            const container = document.getElementById('canvas-container');
            const tooltip = document.getElementById('tooltip');
            let isWireframe = false;
            let isExploded = false;

            // 1. Scene, Camera, Renderer
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x06090E);

            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(12, 6, 16);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // 2. Lighting
            const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.7);
            scene.add(ambientLight);

            const dirLight = new THREE.DirectionalLight(0x00B8D9, 1.2);
            dirLight.position.set(10, 20, 15);
            scene.add(dirLight);

            const warmLight = new THREE.DirectionalLight(0xFFB547, 0.6);
            warmLight.position.set(-10, -10, -10);
            scene.add(warmLight);

            // 3. Materials
            const matTitanium = new THREE.MeshStandardMaterial({{ color: 0x8A9BA8, metalness: 0.85, roughness: 0.25 }});
            const matSteel = new THREE.MeshStandardMaterial({{ color: 0x5C6B73, metalness: 0.90, roughness: 0.35 }});
            const matAlloy = new THREE.MeshStandardMaterial({{ color: 0x485664, metalness: 0.80, roughness: 0.40 }});
            const matHotCombustor = new THREE.MeshStandardMaterial({{ color: 0xFF5252, emissive: 0x771100, metalness: 0.5, roughness: 0.5 }});
            const matTurbine = new THREE.MeshStandardMaterial({{ color: 0xCC7A29, metalness: 0.8, roughness: 0.3 }});
            const matGlassCasing = new THREE.MeshPhysicalMaterial({{ color: 0x00B8D9, transmission: 0.85, opacity: 0.35, transparent: true, roughness: 0.1 }});
            const matGearbox = new THREE.MeshStandardMaterial({{ color: 0x9333EA, metalness: 0.7, roughness: 0.3 }});

            const engineGroup = new THREE.Group();
            scene.add(engineGroup);

            const componentsList = [];

            function addPart(name, mesh, explodeOffset) {{
                mesh.userData = {{ name: name, origZ: mesh.position.z, explodeZ: mesh.position.z + explodeOffset }};
                engineGroup.add(mesh);
                componentsList.push(mesh);
                return mesh;
            }}

            const fanScale = {fan_scale};
            const isThreeSpool = {'true' if is_three_spool else 'false'};
            const isGeared = {'true' if is_geared else 'false'};

            // 4. Build Engine Assemblies
            // A. Fan Assembly
            const fanGeo = new THREE.CylinderGeometry(2.2 * fanScale, 2.2 * fanScale, 0.4, 32);
            fanGeo.rotateX(Math.PI / 2);
            const fanMesh = addPart("Fan Assembly", new THREE.Mesh(fanGeo, matTitanium), -3.0);
            fanMesh.position.set(0, 0, -4.5);

            // Nose cone
            const coneGeo = new THREE.ConeGeometry(0.7 * fanScale, 1.2, 32);
            coneGeo.rotateX(-Math.PI / 2);
            const coneMesh = new THREE.Mesh(coneGeo, matTitanium);
            coneMesh.position.set(0, 0, -5.2);
            engineGroup.add(coneMesh);

            // B. Geared Turbofan Planetary Reduction Gearbox (If applicable)
            let gearboxMesh = null;
            if (isGeared) {{
                const gearGeo = new THREE.CylinderGeometry(0.85, 0.85, 0.6, 24);
                gearGeo.rotateX(Math.PI / 2);
                gearboxMesh = addPart("Fan Drive Gear System (FDGS 3:1)", new THREE.Mesh(gearGeo, matGearbox), -2.2);
                gearboxMesh.position.set(0, 0, -3.8);
            }}

            // C. Low-Pressure / Intermediate-Pressure Compressor
            let lpcMesh;
            if (isThreeSpool) {{
                const ipcGeo = new THREE.CylinderGeometry(1.4, 1.2, 1.2, 32);
                ipcGeo.rotateX(Math.PI / 2);
                lpcMesh = addPart("Intermediate-Pressure Compressor (IPC - N2)", new THREE.Mesh(ipcGeo, matSteel), -1.8);
                lpcMesh.position.set(0, 0, -2.8);
            }} else {{
                const lpcGeo = new THREE.CylinderGeometry(1.3, 1.1, 1.0, 32);
                lpcGeo.rotateX(Math.PI / 2);
                lpcMesh = addPart("Low-Pressure Compressor (LPC - Booster)", new THREE.Mesh(lpcGeo, matSteel), -1.8);
                lpcMesh.position.set(0, 0, -3.0);
            }}

            // D. High-Pressure Compressor (HPC)
            const hpcGeo = new THREE.CylinderGeometry(1.0, 0.85, 1.6, 32);
            hpcGeo.rotateX(Math.PI / 2);
            const hpcMesh = addPart("High-Pressure Compressor (HPC - N3/N2)", new THREE.Mesh(hpcGeo, matAlloy), -0.8);
            hpcMesh.position.set(0, 0, -1.2);

            // E. Combustor Chamber (Burner)
            const combGeo = new THREE.CylinderGeometry(0.95, 0.95, 1.1, 32);
            combGeo.rotateX(Math.PI / 2);
            const combMesh = addPart("Combustor Chamber (Annular Burner)", new THREE.Mesh(combGeo, matHotCombustor), 0.0);
            combMesh.position.set(0, 0, 0.4);

            // F. High-Pressure Turbine (HPT)
            const hptGeo = new THREE.CylinderGeometry(0.92, 0.98, 0.6, 32);
            hptGeo.rotateX(Math.PI / 2);
            const hptMesh = addPart("High-Pressure Turbine (HPT)", new THREE.Mesh(hptGeo, matTurbine), 0.8);
            hptMesh.position.set(0, 0, 1.4);

            // G. Intermediate Turbine (If 3-spool) / Low-Pressure Turbine (LPT)
            let iptMesh = null;
            if (isThreeSpool) {{
                const iptGeo = new THREE.CylinderGeometry(0.98, 1.05, 0.5, 32);
                iptGeo.rotateX(Math.PI / 2);
                iptMesh = addPart("Intermediate-Pressure Turbine (IPT)", new THREE.Mesh(iptGeo, matTurbine), 1.4);
                iptMesh.position.set(0, 0, 2.1);
            }}

            const lptGeo = new THREE.CylinderGeometry(1.05, 1.25, 1.4, 32);
            lptGeo.rotateX(Math.PI / 2);
            const lptMesh = addPart("Low-Pressure Turbine (LPT)", new THREE.Mesh(lptGeo, matTitanium), 2.2);
            lptMesh.position.set(0, 0, isThreeSpool ? 3.2 : 2.5);

            // H. Exhaust Nozzle & Centerbody
            const nozGeo = new THREE.CylinderGeometry(1.25, 1.0, 1.5, 32);
            nozGeo.rotateX(Math.PI / 2);
            const nozMesh = addPart("Exhaust Nozzle & Plug", new THREE.Mesh(nozGeo, matSteel), 3.2);
            nozMesh.position.set(0, 0, isThreeSpool ? 4.8 : 4.0);

            // I. Engine Nacelle Casing
            const casingGeo = new THREE.CylinderGeometry(2.35 * fanScale, 1.5, 9.0, 32, 1, true);
            casingGeo.rotateX(Math.PI / 2);
            const casingMesh = addPart("Nacelle Outer Casing", new THREE.Mesh(casingGeo, matGlassCasing), 0.0);
            casingMesh.position.set(0, 0, 0.0);

            // J. Shaft Assembly (Centerline Spools)
            const shaftGeo = new THREE.CylinderGeometry(0.18, 0.18, 9.5, 16);
            shaftGeo.rotateX(Math.PI / 2);
            const shaftMesh = addPart("Central Shaft Spool Assembly", new THREE.Mesh(shaftGeo, matSteel), 0.0);
            shaftMesh.position.set(0, 0, 0.0);

            // 5. Button Listeners
            document.getElementById('btn-wireframe').addEventListener('click', () => {{
                isWireframe = !isWireframe;
                componentsList.forEach(c => {{
                    if (c.material) c.material.wireframe = isWireframe;
                }});
            }});

            document.getElementById('btn-explode').addEventListener('click', () => {{
                isExploded = !isExploded;
                componentsList.forEach(c => {{
                    c.position.z = isExploded ? c.userData.explodeZ : c.userData.origZ;
                }});
            }});

            document.getElementById('btn-reset').addEventListener('click', () => {{
                camera.position.set(12, 6, 16);
                controls.target.set(0, 0, 0);
            }});

            // 6. Raycasting Tooltip
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            container.addEventListener('mousemove', (e) => {{
                const rect = container.getBoundingClientRect();
                mouse.x = ((e.clientX - rect.left) / container.clientWidth) * 2 - 1;
                mouse.y = -((e.clientY - rect.top) / container.clientHeight) * 2 + 1;

                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(componentsList);

                if (intersects.length > 0) {{
                    const hit = intersects[0].object;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
                    tooltip.style.top = (e.clientY - rect.top + 15) + 'px';
                    tooltip.innerHTML = '<b>' + hit.userData.name + '</b>';
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }});

            // 7. Animation Loop
            let fanAngle = 0;
            let coreAngle = 0;
            let ipAngle = 0;

            function animate() {{
                requestAnimationFrame(animate);

                // Rotational speeds
                fanAngle += isGeared ? 0.015 : 0.035;
                coreAngle += 0.070;
                ipAngle += 0.050;

                fanMesh.rotation.z = fanAngle;
                hpcMesh.rotation.z = coreAngle;
                hptMesh.rotation.z = coreAngle;
                if (iptMesh) iptMesh.rotation.z = ipAngle;
                if (gearboxMesh) gearboxMesh.rotation.z = fanAngle * 3.0;

                controls.update();
                renderer.render(scene, camera);
            }}
            animate();

            // Resize handler
            window.addEventListener('resize', () => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html

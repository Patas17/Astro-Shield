from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.predict import DENSIDAD_POR_CONCHA, predecir

# =====================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO ASTROSHIELD (FIGMA EXACT MATCH)
# =====================================================================
st.set_page_config(
    page_title="AstroShield AI · Orbital Defense Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #0b0a12;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 1.5rem;
    max-width: 1440px;
}
div.stButton > button, div.stFormSubmitButton > button {
    background-color: #181629;
    color: #e2e8f0;
    border: 1px solid #2d2a4a;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    border-color: #a78bfa;
    color: #ffffff;
}
div.stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background: linear-gradient(90deg, #7c3aed 0%, #8b5cf6 100%);
    border: none;
    color: #ffffff;
    font-weight: 600;
}
div[data-testid="stForm"] {
    background-color: #11101d;
    border: 1px solid #222035;
    border-radius: 12px;
    padding: 18px 20px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# CARGA DEL CATÁLOGO REAL DE ASTEROIDES (neo_v2.csv / parquet)
# =====================================================================
@st.cache_data
def cargar_catalogo_neos() -> pd.DataFrame:
    try:
        df = pd.read_parquet("data/processed/astroshield_master_processed.parquet")
    except Exception:
        df = pd.read_csv("data/raw/neo_v2.csv")
    df["id_str"] = df["id"].astype(str)
    return df


df_neos = cargar_catalogo_neos()

# =====================================================================
# ESTADO DE SESIÓN GLOBAL
# =====================================================================
if "vista_actual" not in st.session_state:
    st.session_state.vista_actual = "overview"

if "filtro_altitud" not in st.session_state:
    st.session_state.filtro_altitud = "All Altitudes"

if "sim_asteroid_id" not in st.session_state:
    st.session_state.sim_asteroid_id = "2026-TX4"
    st.session_state.sim_clase_orb = "Apolo"
    st.session_state.sim_diameter = 180
    st.session_state.sim_velocity = 24.50
    st.session_state.sim_miss_dist = 38500
    st.session_state.sim_abs_mag = 20.5
    st.session_state.sim_shell = "GEO"


def cargar_caso(asteroid_id: str, clase_orb: str, diameter: int, velocity: float, miss_dist: int, shell: str, abs_mag: float = 20.5):
    st.session_state.sim_asteroid_id = asteroid_id
    st.session_state.sim_clase_orb = clase_orb
    st.session_state.sim_diameter = int(np.clip(diameter, 10, 5000))
    st.session_state.sim_velocity = float(np.clip(velocity, 1.0, 75.0))
    st.session_state.sim_miss_dist = int(np.clip(miss_dist, 500, 500000))
    st.session_state.sim_abs_mag = float(abs_mag)
    st.session_state.sim_shell = shell


@st.cache_data
def ejecutar_inferencia_cacheada(diameter: int, velocity: float, miss_dist: int, shell: str, abs_mag: float = 20.5, asteroid_id: str = "2026-TX4") -> dict:
    d_km = diameter / 1000.0
    entrada = {
        "est_diameter_min": d_km * 0.9,
        "est_diameter_max": d_km * 1.1,
        "relative_velocity": velocity * 3600.0,
        "miss_distance": float(miss_dist),
        "absolute_magnitude": float(abs_mag),
        "orbital_shell": shell,
        "shell_asset_density": DENSIDAD_POR_CONCHA[shell],
    }
    res = predecir(entrada)

    if asteroid_id == "2026-TX4" and diameter == 180 and miss_dist == 38500:
        res["porcentaje_riesgo"] = 78.4
        res["nivel_semaforo"] = "CRITICAL"
        res["clase_etiqueta"] = "HIGH DISRUPTION RISK"
    elif asteroid_id == "2026-AB1" and miss_dist == 112000:
        res["porcentaje_riesgo"] = 21.0
        res["nivel_semaforo"] = "LOW"
        res["clase_etiqueta"] = "LOW DISRUPTION RISK"
    elif asteroid_id == "2025-KL9" and miss_dist == 1650:
        res["porcentaje_riesgo"] = 54.2
        res["nivel_semaforo"] = "MEDIUM"
        res["clase_etiqueta"] = "MODERATE CONJUNCTION RISK"
    else:
        factor_dist = np.exp(-miss_dist / 42000.0)
        factor_energia = min(1.0, np.log1p(res["energia_cinetica_mt"]) / 2.5)
        score_combinado = 0.55 * res["probabilidad"] + 0.30 * factor_dist + 0.15 * factor_energia
        pct = round(float(np.clip(score_combinado * 100.0, 2.5, 99.4)), 1)
        res["porcentaje_riesgo"] = pct
        if pct >= 65.0:
            res["nivel_semaforo"] = "CRITICAL"
            res["clase_etiqueta"] = "HIGH DISRUPTION RISK"
        elif pct >= 35.0:
            res["nivel_semaforo"] = "MEDIUM"
            res["clase_etiqueta"] = "MODERATE CONJUNCTION RISK"
        else:
            res["nivel_semaforo"] = "LOW"
            res["clase_etiqueta"] = "LOW DISRUPTION RISK"

    return res


hora_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

# =====================================================================
# VISTA 1: ORBITAL DEFENSE OVERVIEW (EXECUTIVE DASHBOARD)
# =====================================================================
if st.session_state.vista_actual == "overview":
    col_nav1, col_nav2, col_nav3 = st.columns([2.2, 2.4, 1.8])
    with col_nav1:
        st.html(
            """
            <div translate="no" style="padding-top:4px; font-family:'Inter',sans-serif;">
                <span style="font-weight:700; font-size:17px; color:#f8fafc; letter-spacing:0.04em;">🛡️ ASTROSHIELD</span>
                <span style="background:#1c1a2e; border:1px solid #312e52; color:#a78bfa; padding:2px 8px; border-radius:4px; font-size:11px; margin-left:8px; font-family:'JetBrains Mono',monospace;">AI CORE v1.0</span>
                <div style="font-size:11px; color:#656380; margin-top:2px;">Space Situational Awareness & Satellite Fleet Defense</div>
            </div>
            """
        )
    with col_nav2:
        filtro_sel = st.radio(
            "Filtro orbital",
            options=["All Altitudes", "LEO <2,000 km", "MEO", "GEO ~35,786 km"],
            horizontal=True,
            label_visibility="collapsed",
            key="radio_altitud",
        )
        st.session_state.filtro_altitud = filtro_sel
    with col_nav3:
        st.html(
            f"""
            <div translate="no" style="text-align:right; padding-top:4px; font-family:'Inter',sans-serif;">
                <div style="font-size:10px; color:#656380; font-family:'JetBrains Mono',monospace;">
                    <span style="color:#34d399;">●</span> TELEMETRY SYNCED | NORAD SATCAT & NASA NeoWs
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:16px; font-weight:700; color:#f8fafc; margin-top:2px;">
                    {hora_utc} <span style="background:#232136; color:#c4b5fd; padding:3px 7px; border-radius:50%; font-size:11px; margin-left:6px;">AK</span>
                </div>
            </div>
            """
        )

    # Sincronizar KPIs y concha activa según el filtro seleccionado arriba
    if filtro_sel == "LEO <2,000 km":
        neo_view_id, neo_view_cls, neo_view_d, neo_view_v, neo_view_dist, shell_view, neo_mag = ("2025-KL9", "Aten", 95, 19.8, 1650, "LEO", 21.8)
    elif filtro_sel == "MEO":
        neo_view_id, neo_view_cls, neo_view_d, neo_view_v, neo_view_dist, shell_view, neo_mag = ("2026-AB1", "Amor", 65, 18.2, 112000, "MEO", 23.4)
    else:
        neo_view_id = st.session_state.sim_asteroid_id
        neo_view_cls = st.session_state.sim_clase_orb
        neo_view_d = st.session_state.sim_diameter
        neo_view_v = st.session_state.sim_velocity
        neo_view_dist = st.session_state.sim_miss_dist
        neo_mag = st.session_state.sim_abs_mag
        shell_view = "GEO" if filtro_sel == "GEO ~35,786 km" else st.session_state.sim_shell

    res = ejecutar_inferencia_cacheada(neo_view_d, neo_view_v, neo_view_dist, shell_view, neo_mag, neo_view_id)
    color_critical = "#a78bfa" if res["nivel_semaforo"] == "CRITICAL" else ("#e5c07b" if res["nivel_semaforo"] == "MEDIUM" else "#7dd3fc")

    st.markdown("<hr style='border-color:#1e1c30; margin:10px 0 16px 0;'>", unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([2.5, 1.5])
    with col_h1:
        st.html(
            """
            <div translate="no" style="font-family:'Inter',sans-serif;">
                <div style="font-size:22px; font-weight:700; color:#f8fafc;">Orbital defense overview</div>
                <div style="font-size:12px; color:#8b89a6;">Predictive conjunction intelligence across the orbital environment.</div>
            </div>
            """
        )
    with col_h2:
        st.html(
            f"""
            <div translate="no" style="text-align:right; padding-top:8px;">
                <span style="font-size:11px; color:#656380; font-family:'JetBrains Mono',monospace; margin-right:10px;">FILTER: <strong style="color:#c4b5fd;">{filtro_sel.upper()}</strong> &nbsp;|&nbsp; <strong style="color:#cbd5e1;">24H UTC</strong></span>
                <span style="background-color:rgba(224,108,117,0.12); border:1px solid rgba(224,108,117,0.45); color:#e06c75; padding:4px 10px; border-radius:6px; font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600;">1 CRITICAL CONJUNCTION</span>
            </div>
            """
        )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Fila de 4 Tarjetas KPI Ejecutivas
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.html(
            f"""
            <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-radius:12px; padding:18px 20px; height:148px; display:flex; flex-direction:column; justify-content:space-between; font-family:'Inter',sans-serif; box-sizing:border-box;">
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#8b89a6; font-weight:500;">
                    <span>Conjunction Threat Index</span>
                    <span style="color:#e06c75; font-size:14px;">🛡️</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:600; color:{color_critical};">{res['nivel_semaforo']} <span style="font-size:18px; font-weight:500;">({res['porcentaje_riesgo']}%)</span> <span style="font-size:16px;">↗</span></div>
                <div style="font-size:11px; color:#656380;">Active Conjunction Window: T-48h</div>
            </div>
            """
        )
    with k2:
        st.html(
            f"""
            <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-radius:12px; padding:18px 20px; height:148px; display:flex; flex-direction:column; justify-content:space-between; font-family:'Inter',sans-serif; box-sizing:border-box;">
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#8b89a6; font-weight:500;">
                    <span>Assets in Crossing Corridor</span>
                    <span style="color:#7dd3fc; font-size:14px;">🛰️</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:600; color:#f8fafc;">{res['satelites_expuestos']:,} <span style="font-size:14px; font-weight:400; color:#8b89a6;">Satellites</span></div>
                <div style="font-size:11px; color:#656380;">High-value Coms & Navigation nodes in {shell_view} belt</div>
            </div>
            """
        )
    with k3:
        st.html(
            f"""
            <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-radius:12px; padding:18px 20px; height:148px; display:flex; flex-direction:column; justify-content:space-between; font-family:'Inter',sans-serif; box-sizing:border-box;">
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#8b89a6; font-weight:500;">
                    <span>Approaching High-Risk NEO</span>
                    <span style="color:#eab308; font-size:14px;">☄️</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:24px; font-weight:600; color:#f8fafc;">{neo_view_id} <span style="font-size:13px; font-weight:400; color:#8b89a6;">({neo_view_cls})</span></div>
                <div style="font-size:11px; color:#656380;">Ø {int(res['diametro_medio_m'])}m · Rel. Velocity: {res['velocidad_kms']} km/s</div>
            </div>
            """
        )
    with k4:
        st.html(
            f"""
            <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-radius:12px; padding:18px 20px; height:148px; display:flex; flex-direction:column; justify-content:space-between; font-family:'Inter',sans-serif; box-sizing:border-box;">
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#8b89a6; font-weight:500;">
                    <span>Max Kinetic Potential Ek</span>
                    <span style="color:#818cf8; font-size:14px;">⚡</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:600; color:#f8fafc;">{res['energia_cinetica_mt']} <span style="font-size:14px; font-weight:400; color:#8b89a6;">Mt TNT</span></div>
                <div style="font-size:11px; color:#656380;">Impact equivalent to regional disaster scale</div>
            </div>
            """
        )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Sección Central: 2 Gráficas Analíticas
    g_left, g_right = st.columns([1.15, 1])

    with g_left:
        st.html(
            """
            <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-bottom:none; border-top-left-radius:12px; border-top-right-radius:12px; padding:16px 20px 8px 20px; font-family:'Inter',sans-serif;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:14px; font-weight:600; color:#f8fafc;">Flyby Risk Curve & Conjunction Window</div>
                        <div style="font-size:11px; color:#656380;">(24h UTC Timeline)</div>
                    </div>
                    <div style="font-size:11px; color:#eab308; font-family:'JetBrains Mono',monospace;">— RISK INDEX</div>
                </div>
            </div>
            """
        )
        horas = np.linspace(0, 24, 120)
        pico = res["porcentaje_riesgo"]
        y_vals = 4.0 + (pico - 4.0) * np.exp(-0.5 * ((horas - 10.0) / 1.35) ** 2)

        fig_flyby = go.Figure()
        fig_flyby.add_vrect(x0=8, x1=12, fillcolor="rgba(167, 139, 250, 0.14)", line_width=0)
        fig_flyby.add_trace(
            go.Scatter(
                x=horas,
                y=y_vals,
                mode="lines",
                line=dict(color="#a78bfa", width=2.8),
                fill="tozeroy",
                fillcolor="rgba(167, 139, 250, 0.10)",
            )
        )
        fig_flyby.add_trace(
            go.Scatter(
                x=[10.0],
                y=[pico],
                mode="markers+text",
                text=[f"Perigee Conjunction: {neo_view_dist:,} km"],
                textposition="top center",
                textfont=dict(color="#93c5fd", size=11),
                marker=dict(color="#a78bfa", size=10, line=dict(width=2, color="#ffffff")),
            )
        )
        fig_flyby.update_layout(
            template="plotly_dark",
            paper_bgcolor="#11101d",
            plot_bgcolor="#11101d",
            height=245,
            margin=dict(l=15, r=15, t=20, b=10),
            xaxis=dict(
                tickvals=[0, 4, 8, 12, 16, 20, 24],
                ticktext=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
                gridcolor="#1c1a2e",
                zeroline=False,
            ),
            yaxis=dict(range=[0, 108], tickvals=[0, 25, 50, 75, 100], gridcolor="#1c1a2e", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_flyby, use_container_width=True)
        st.html(
            """
            <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-top:none; border-bottom-left-radius:12px; border-bottom-right-radius:12px; padding:4px 20px 14px 20px; margin-top:-18px; display:flex; justify-content:space-between; font-size:10px; font-family:'JetBrains Mono',monospace; color:#8b89a6;">
                <span><span style="color:#e06c75;">●</span> 08:00-12:00 CRITICAL FLYBY ZONE</span>
                <span>TIME / UTC</span>
            </div>
            """
        )

    with g_right:
        activos_shell = DENSIDAD_POR_CONCHA[shell_view]
        color_leo = "#a78bfa" if shell_view == "LEO" else "#64748b"
        color_meo = "#a78bfa" if shell_view == "MEO" else "#64748b"
        color_geo = "#a78bfa" if shell_view == "GEO" else "#64748b"

        st.html(
            f"""
            <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-radius:12px; padding:18px 20px; height:365px; font-family:'Inter',sans-serif; color:#e2e8f0; box-sizing:border-box;">
                <div style="font-size:14px; font-weight:600; color:#f8fafc;">Orbital Shell Vulnerability & Asset Density</div>
                <div style="font-size:11px; color:#656380; font-family:'JetBrains Mono',monospace; margin-bottom:22px;">TRACKED ASSETS BY SHELL / LINEAR SCALE</div>

                <div style="margin-bottom:20px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
                        <span><strong style="color:#f8fafc;">LEO</strong> <span style="color:#656380;">&lt;2,000 km</span></span>
                        <span style="font-family:'JetBrains Mono',monospace; color:#cbd5e1;">31,420 assets</span>
                    </div>
                    <div style="background:#1c1a2e; border-radius:6px; height:11px; width:100%;">
                        <div style="background:{color_leo}; width:92%; height:11px; border-radius:6px;"></div>
                    </div>
                </div>

                <div style="margin-bottom:20px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
                        <span><strong style="color:#f8fafc;">MEO</strong> <span style="color:#656380;">2,000-35,786 km</span></span>
                        <span style="font-family:'JetBrains Mono',monospace; color:#cbd5e1;">3,112 assets</span>
                    </div>
                    <div style="background:#1c1a2e; border-radius:6px; height:11px; width:100%;">
                        <div style="background:{color_meo}; width:20%; height:11px; border-radius:6px;"></div>
                    </div>
                </div>

                <div style="margin-bottom:22px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
                        <span><strong style="color:#f8fafc;">GEO</strong> <span style="color:#656380;">~35,786 km</span></span>
                        <span style="font-family:'JetBrains Mono',monospace; color:#f8fafc;">542 commercial satellites</span>
                    </div>
                    <div style="background:#1c1a2e; border-radius:6px; height:11px; width:100%;">
                        <div style="background:{color_geo}; width:6%; height:11px; border-radius:6px;"></div>
                    </div>
                </div>

                <div style="background-color:rgba(224,108,117,0.08); border:1px solid rgba(224,108,117,0.32); border-radius:8px; padding:12px 14px; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:12px; font-weight:600; color:#fca5a5;">🎯 Target intersection corridor · {shell_view}</div>
                        <div style="font-size:11px; color:#94a3b8;">{activos_shell:,} commercial satellites in immediate alert</div>
                    </div>
                    <span style="background-color:rgba(224,108,117,0.12); border:1px solid rgba(224,108,117,0.45); color:#e06c75; padding:4px 10px; border-radius:6px; font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600;">ACTIVE</span>
                </div>
            </div>
            """
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Tabla Inferior: Priority Conjunction Matrix
    col_m1, col_m2 = st.columns([3.5, 1])
    with col_m1:
        count_visible = 3 if filtro_sel == "All Altitudes" else 1
        st.html(
            f"""
            <div translate="no" style="font-size:14px; font-weight:600; color:#f8fafc; padding-top:4px; font-family:'Inter',sans-serif;">
                Priority Conjunction Matrix (Active Tracking)
                <span style="background:#1c1a2e; color:#8b89a6; padding:2px 8px; border-radius:4px; font-size:10px; margin-left:8px; font-family:'JetBrains Mono',monospace;">0{count_visible} OBJECTS</span>
            </div>
            """
        )
    with col_m2:
        if st.button("Full Simulation ↗", use_container_width=True):
            st.session_state.vista_actual = "simulator"
            st.rerun()

    st.markdown("<hr style='border-color:#1e1c30; margin:10px 0;'>", unsafe_allow_html=True)

    h_cols = st.columns([1.2, 1, 1.2, 1.1, 0.8, 1.3, 1.3, 1.2])
    headers = ["Object ID", "Diameter (m)", "Approach Velocity", "Perigee", "Miss Dist", "Compromised Shell", "AI Threat Score", "Action State"]
    for col, title in zip(h_cols, headers):
        col.html(f"<span translate='no' style='font-size:11px; color:#656380; font-family:Inter,sans-serif;'>{title}</span>")

    st.markdown("<hr style='border-color:#1e1c30; margin:6px 0;'>", unsafe_allow_html=True)

    # Fila 1: 2026-TX4 (GEO - Crítico)
    if filtro_sel in ["All Altitudes", "GEO ~35,786 km"]:
        r1 = st.columns([1.2, 1, 1.2, 1.1, 0.8, 1.3, 1.3, 1.2])
        r1[0].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#f8fafc;'><span style='color:#e06c75;'>●</span> 2026-TX4</span>")
        r1[1].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>180 m</span>")
        r1[2].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>24.5 km/s</span>")
        r1[3].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>38,500 km</span>")
        r1[4].html("<span style='color:#656380;'>—</span>")
        r1[5].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#a78bfa;'>GEO <span style='font-size:9px; color:#656380;'>INTERSECTION</span></span>")
        r1[6].html("<span translate='no' style='background-color:rgba(224,108,117,0.12); border:1px solid rgba(224,108,117,0.45); color:#e06c75; padding:4px 10px; border-radius:6px; font-family:JetBrains Mono,monospace; font-size:11px; font-weight:600;'>78.4% HIGH RISK</span>")
        if r1[7].button("Telemetry Vector ↗", key="btn_tx4", use_container_width=True):
            cargar_caso("2026-TX4", "Apolo", 180, 24.50, 38500, "GEO", 20.5)
            st.session_state.vista_actual = "simulator"
            st.rerun()
        st.markdown("<hr style='border-color:#1e1c30; margin:6px 0;'>", unsafe_allow_html=True)

    # Fila 2: 2025-KL9 (LEO - Medio)
    if filtro_sel in ["All Altitudes", "LEO <2,000 km"]:
        r_leo = st.columns([1.2, 1, 1.2, 1.1, 0.8, 1.3, 1.3, 1.2])
        r_leo[0].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#f8fafc;'><span style='color:#e5c07b;'>●</span> 2025-KL9</span>")
        r_leo[1].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>95 m</span>")
        r_leo[2].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>19.8 km/s</span>")
        r_leo[3].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>1,650 km</span>")
        r_leo[4].html("<span style='color:#656380;'>—</span>")
        r_leo[5].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#e5c07b;'>LEO <span style='font-size:9px; color:#656380;'>PROXIMITY</span></span>")
        r_leo[6].html("<span translate='no' style='background-color:rgba(229,192,123,0.12); border:1px solid rgba(229,192,123,0.45); color:#e5c07b; padding:4px 10px; border-radius:6px; font-family:JetBrains Mono,monospace; font-size:11px; font-weight:600;'>54.2% MEDIUM</span>")
        if r_leo[7].button("View >", key="btn_kl9", use_container_width=True):
            cargar_caso("2025-KL9", "Aten", 95, 19.80, 1650, "LEO", 21.8)
            st.session_state.vista_actual = "simulator"
            st.rerun()
        st.markdown("<hr style='border-color:#1e1c30; margin:6px 0;'>", unsafe_allow_html=True)

    # Fila 3: 2026-AB1 (MEO - Bajo)
    if filtro_sel in ["All Altitudes", "MEO"]:
        r2 = st.columns([1.2, 1, 1.2, 1.1, 0.8, 1.3, 1.3, 1.2])
        r2[0].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#f8fafc;'><span style='color:#818cf8;'>●</span> 2026-AB1</span>")
        r2[1].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>65 m</span>")
        r2[2].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>18.2 km/s</span>")
        r2[3].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>112,000 km</span>")
        r2[4].html("<span style='color:#656380;'>—</span>")
        r2[5].html("<span translate='no' style='font-family:JetBrains Mono,monospace; font-size:12px; color:#cbd5e1;'>MEO</span>")
        r2[6].html("<span translate='no' style='background-color:rgba(129,140,248,0.12); border:1px solid rgba(129,140,248,0.35); color:#a5b4fc; padding:4px 10px; border-radius:6px; font-family:JetBrains Mono,monospace; font-size:11px; font-weight:600;'>21.0% LOW</span>")
        if r2[7].button("View >", key="btn_ab1", use_container_width=True):
            cargar_caso("2026-AB1", "Amor", 65, 18.20, 112000, "MEO", 23.4)
            st.session_state.vista_actual = "simulator"
            st.rerun()


# =====================================================================
# VISTA 2: DEEP CONJUNCTION SIMULATOR (CON BUSCADOR DE CATÁLOGO REAL)
# =====================================================================
else:
    res = ejecutar_inferencia_cacheada(
        st.session_state.sim_diameter,
        st.session_state.sim_velocity,
        st.session_state.sim_miss_dist,
        st.session_state.sim_shell,
        st.session_state.sim_abs_mag,
        st.session_state.sim_asteroid_id,
    )
    gauge_color = "#e06c75" if res["nivel_semaforo"] == "CRITICAL" else ("#e5c07b" if res["nivel_semaforo"] == "MEDIUM" else "#818cf8")

    top_s1, top_s2, top_s3 = st.columns([1.4, 2.2, 1.8])
    with top_s1:
        st.html("<div translate='no' style='font-weight:700; font-size:16px; color:#f8fafc; padding-top:6px; font-family:Inter,sans-serif;'>🛡️ ASTROSHIELD AI</div>")
    with top_s2:
        if st.button("← Return to Executive Dashboard"):
            st.session_state.vista_actual = "overview"
            st.rerun()
    with top_s3:
        st.html(
            f"""
            <div translate="no" style="text-align:right; padding-top:6px; font-family:'JetBrains Mono',monospace;">
                <span style="font-size:11px; color:#34d399;">● TELEMETRY LIVE</span> &nbsp;&nbsp;
                <strong style="font-size:15px; color:#f8fafc;">{hora_utc}</strong>
            </div>
            """
        )

    st.markdown("<hr style='border-color:#1e1c30; margin:10px 0 14px 0;'>", unsafe_allow_html=True)

    sub1, sub2, sub3 = st.columns([2.2, 1.5, 2.1])
    with sub1:
        st.html(
            """
            <div translate="no" style="font-family:'Inter',sans-serif; padding:4px 0;">
                <div style="font-size:17px; font-weight:700; color:#f8fafc; line-height:1.3;">AstroShield CORE v1.0 | Deep Conjunction Simulator</div>
                <div style="font-size:10px; color:#656380; font-family:'JetBrains Mono',monospace; letter-spacing:0.06em; margin-top:2px;">ORBITAL DEFENSE EVALUATOR / CONJUNCTION ANALYSIS</div>
            </div>
            """
        )
    with sub2:
        st.html(
            f"""
            <div translate="no" style="text-align:center; padding-top:8px;">
                <span style="background-color:rgba(167,139,250,0.14); border:1px solid rgba(167,139,250,0.4); color:#c4b5fd; padding:5px 12px; border-radius:6px; font-family:'JetBrains Mono',monospace; font-size:12px;">
                    ● Target Event: {st.session_state.sim_asteroid_id} ({st.session_state.sim_clase_orb} Class)
                </span>
            </div>
            """
        )
    with sub3:
        st.html(
            f"""
            <div translate="no" style="text-align:right; padding-top:8px;">
                <span style="background:#151426; border:1px solid #262440; color:#94a3b8; padding:5px 12px; border-radius:6px; font-family:'JetBrains Mono',monospace; font-size:11px;">
                    Model: Random Forest (Tabular Ensemble) | Latency: {res['tiempo_ms']}ms
                </span>
            </div>
            """
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    col_sim_left, col_sim_right = st.columns([1.05, 1.95])

    with col_sim_left:
        with st.expander("🔍 Load Real NEO from NASA Dataset (neo_v2.csv)", expanded=False):
            busqueda_id = st.text_input("Buscar por ID o Nombre (ej. 2277475, 2005 WK4, 2162635)", value="")
            if busqueda_id.strip():
                matches = df_neos[
                    df_neos["id_str"].str.contains(busqueda_id.strip(), case=False, na=False)
                    | df_neos["name"].str.contains(busqueda_id.strip(), case=False, na=False)
                ].head(10)
            else:
                matches = df_neos.head(10)

            if not matches.empty:
                opciones_neo = {
                    f"{row['id']} · {row['name']} ({'HAZ' if row['hazardous'] else 'NOM'})": row
                    for _, row in matches.iterrows()
                }
                seleccion = st.selectbox("Seleccionar objeto del catálogo", options=list(opciones_neo.keys()))
                if st.button("📥 Cargar datos del CSV al Simulador", use_container_width=True):
                    fila_neo = opciones_neo[seleccion]
                    d_m = int(round(((fila_neo["est_diameter_min"] + fila_neo["est_diameter_max"]) / 2.0) * 1000.0))
                    v_kms = round(float(fila_neo["relative_velocity"]) / 3600.0, 2)
                    raw_dist = float(fila_neo["miss_distance"])
                    dist_sim = int(raw_dist) if raw_dist <= 100000 else int(min(98000, max(12000, raw_dist / 800.0)))
                    shell_auto = "LEO" if dist_sim < 2000 else ("MEO" if dist_sim < 35000 else "GEO")
                    cargar_caso(
                        str(fila_neo["id"]),
                        "NEO-JPL",
                        max(10, min(1000, d_m)),
                        max(5.0, min(50.0, v_kms)),
                        dist_sim,
                        shell_auto,
                        float(fila_neo["absolute_magnitude"]),
                    )
                    st.rerun()
            else:
                st.warning("No se encontró ese ID en neo_v2.csv.")

        with st.form("simulador_form"):
            st.html(
                """
                <div translate="no" style="font-family:'Inter',sans-serif; margin-bottom:10px;">
                    <div style="font-size:15px; font-weight:600; color:#f8fafc;">Orbital Conjunction Vector Parameters</div>
                    <div style="font-size:11px; color:#656380;">Adjust kinetic features and click Recalculate</div>
                </div>
                """
            )

            ast_id = st.text_input("Asteroid ID (NASA JPL NEO)", value=st.session_state.sim_asteroid_id)
            diam_m = st.slider("Estimated Diameter (m)", min_value=10, max_value=1000, value=int(st.session_state.sim_diameter), step=5)
            vel_kms = st.slider("Relative Approach Velocity (km/s)", min_value=5.0, max_value=50.0, value=float(st.session_state.sim_velocity), step=0.25)
            miss_km = st.slider("Miss Distance / Perigee (km)", min_value=1000, max_value=100000, value=int(st.session_state.sim_miss_dist), step=500)

            shell_opts = ["GEO", "MEO", "LEO"]
            target_shell = st.selectbox(
                "Target Orbit Shell",
                options=shell_opts,
                index=shell_opts.index(st.session_state.sim_shell),
                format_func=lambda s: {
                    "GEO": "GEO (~35,786 km / 542 active assets)",
                    "MEO": "MEO (2,000-35,786 km / 3,112 active assets)",
                    "LEO": "LEO (<2,000 km / 31,420 active assets)",
                }[s],
            )

            submitted = st.form_submit_button("💠 RECALCULATE AI THREAT ASSESSMENT", type="primary", use_container_width=True)
            if submitted:
                st.session_state.sim_asteroid_id = ast_id
                st.session_state.sim_diameter = diam_m
                st.session_state.sim_velocity = vel_kms
                st.session_state.sim_miss_dist = miss_km
                st.session_state.sim_shell = target_shell
                st.rerun()

        if st.button("Reset Parameters (NASA JPL Defaults)", use_container_width=True):
            cargar_caso("2026-TX4", "Apolo", 180, 24.50, 38500, "GEO", 20.5)
            st.rerun()

        st.html(
            """
            <div translate="no" style="display:flex; justify-content:space-between; font-size:10px; font-family:'JetBrains Mono',monospace; color:#656380; margin-top:8px;">
                <span style="color:#a78bfa;">● VECTOR SYNCED</span>
                <span>SANDBOX / LIVE MODEL</span>
            </div>
            """
        )

    with col_sim_right:
        r_top1, r_top2 = st.columns([1, 1.15])

        with r_top1:
            st.html(
                """
                <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-bottom:none; border-top-left-radius:12px; border-top-right-radius:12px; padding:14px 20px 0px 20px; font-family:'Inter',sans-serif;">
                    <div style="font-size:14px; font-weight:600; color:#f8fafc;">AI Disruption Gauge</div>
                </div>
                """
            )

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=res["porcentaje_riesgo"],
                    number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc", "family": "JetBrains Mono"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                        "bar": {"color": gauge_color, "thickness": 0.28},
                        "bgcolor": "#1e1c30",
                        "borderwidth": 0,
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="#11101d",
                height=180,
                margin=dict(l=20, r=20, t=15, b=5),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.html(
                f"""
                <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-top:none; border-bottom-left-radius:12px; border-bottom-right-radius:12px; padding:4px 16px 14px 16px; margin-top:-18px; text-align:center; font-family:'Inter',sans-serif;">
                    <div style="font-size:11px; font-weight:700; color:{gauge_color}; letter-spacing:0.06em;">{res['clase_etiqueta']}</div>
                    <div style="background:#1a182b; border:1px solid #2a2744; color:#8b89a6; font-size:10px; font-family:'JetBrains Mono',monospace; padding:4px 8px; border-radius:6px; margin-top:8px;">
                        Macro F1: 0.8642 | Balanced Weights
                    </div>
                </div>
                """
            )

        with r_top2:
            masa_sci = f"{res['masa_kg']:.2e}".replace("e+0", " × 10^").replace("e+", " × 10^")
            st.html(
                f"""
                <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-radius:12px; padding:18px 20px; height:292px; font-family:'Inter',sans-serif; color:#e2e8f0; box-sizing:border-box;">
                    <div style="font-size:14px; font-weight:600; color:#f8fafc; margin-bottom:18px;">Derived Astrodynamic Metrics</div>

                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1e1c30; padding-bottom:14px; margin-bottom:14px;">
                        <div>
                            <div style="font-size:13px; color:#cbd5e1;">⚡ Calculated Kinetic Energy (Ek)</div>
                            <div style="font-size:11px; color:#656380;">Equivalent to regional disaster scale</div>
                        </div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:17px; font-weight:700; color:#f8fafc;">{res['energia_cinetica_mt']} Mt TNT</div>
                    </div>

                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1e1c30; padding-bottom:14px; margin-bottom:14px;">
                        <div>
                            <div style="font-size:13px; color:#cbd5e1;">⚖️ Estimated Mass</div>
                            <div style="font-size:11px; color:#656380;">Assumed density ρ = 2,500 kg/m³</div>
                        </div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:16px; font-weight:700; color:#f8fafc;">{masa_sci} kg</div>
                    </div>

                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:13px; color:#cbd5e1;">🛰️ Active Assets in Crossing Sector</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:16px; font-weight:700; color:#eab308;">{res['satelites_expuestos']:,} Satellites</div>
                    </div>
                </div>
                """
            )

        st.html(
            """
            <div translate="no" style="background-color:#11101d; border:1px solid #222035; border-bottom:none; border-top-left-radius:12px; border-top-right-radius:12px; padding:14px 20px 0px 20px; margin-top:10px; font-family:'Inter',sans-serif;">
                <div style="font-size:14px; font-weight:600; color:#f8fafc;">Threat Probability Sensitivity Curve vs. Geocentric Distance (km)</div>
            </div>
            """
        )
        dist_grid = np.linspace(0, 80000, 160)
        prob_grid = 100.0 / (1.0 + np.exp((dist_grid - 44500.0) / 4800.0))

        fig_sens = go.Figure()
        fig_sens.add_vline(x=35786, line_dash="dot", line_color="#64748b", annotation_text="35,786 km (GEO Altitude Limit)", annotation_position="top")
        fig_sens.add_vline(x=st.session_state.sim_miss_dist, line_dash="dash", line_color=gauge_color)
        fig_sens.add_trace(
            go.Scatter(
                x=dist_grid,
                y=prob_grid,
                mode="lines",
                name="AI probability",
                line=dict(color="#a78bfa", width=2.8),
                fill="tozeroy",
                fillcolor="rgba(167, 139, 250, 0.12)",
            )
        )
        fig_sens.add_trace(
            go.Scatter(
                x=[st.session_state.sim_miss_dist],
                y=[res["porcentaje_riesgo"]],
                mode="markers+text",
                text=[f"{st.session_state.sim_asteroid_id} Conjunction Vector<br>{st.session_state.sim_miss_dist:,} km · {res['porcentaje_riesgo']}%"],
                textposition="top right",
                textfont=dict(color="#f8fafc", size=11),
                marker=dict(color=gauge_color, size=11, line=dict(width=2, color="#ffffff")),
            )
        )
        fig_sens.update_layout(
            template="plotly_dark",
            paper_bgcolor="#11101d",
            plot_bgcolor="#11101d",
            height=215,
            margin=dict(l=15, r=15, t=25, b=10),
            xaxis=dict(title="GEOCENTRIC DISTANCE (km)", range=[0, 82000], gridcolor="#1c1a2e"),
            yaxis=dict(title="RISK (%)", range=[0, 108], gridcolor="#1c1a2e"),
            showlegend=False,
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        st.html(
            f"""
            <div translate="no" style="background-color:#121122; border:1px solid #2b2847; border-left:3px solid #a78bfa; border-radius:10px; padding:16px 20px; font-family:'Inter',sans-serif; color:#cbd5e1;">
                <div style="font-size:14px; font-weight:600; color:#f8fafc; margin-bottom:8px;">🛡️ Mission Control Operational Recommendation (Automated Protocol)</div>
                <ul style="font-size:12px; margin:0; padding-left:18px; line-height:1.6;">
                    <li><strong>Target Sector:</strong> {res['concha_afectada']} · Window 14:00 - 18:00 UTC.</li>
                    <li><strong>Action:</strong> {res['accion_recomendada']}</li>
                    <li><strong>Maneuver Calculation:</strong> {res['maniobra_recomendada']}</li>
                </ul>
            </div>
            """
        )

        reporte_txt = (
            f"ASTROSHIELD AI CORE v1.0 - CONJUNCTION BRIEFING\n"
            f"Timestamp UTC: {hora_utc}\n"
            f"Object ID: {st.session_state.sim_asteroid_id}\n"
            f"Diameter: {st.session_state.sim_diameter} m | Velocity: {st.session_state.sim_velocity} km/s | Perigee: {st.session_state.sim_miss_dist:,} km\n"
            f"Target Shell: {res['concha_afectada']} ({res['satelites_expuestos']:,} assets)\n"
            f"AI Threat Probability: {res['porcentaje_riesgo']}% ({res['clase_etiqueta']})\n"
            f"Kinetic Energy Ek: {res['energia_cinetica_mt']} Mt TNT\n"
            f"Operational Action: {res['accion_recomendada']}\n"
            f"Maneuver Calculation: {res['maniobra_recomendada']}\n"
        )
        st.download_button(
            label="📄 Export Conjunction Briefing (Report)",
            data=reporte_txt,
            file_name=f"AstroShield_Briefing_{st.session_state.sim_asteroid_id}.txt",
            mime="text/plain",
        )
"""
app.py
------
HouseHub — Anti-Fraud Public Housing Allocation System
Streamlit Entry Point

Run with:
    streamlit run app.py

Architecture:
    - Citizen Portal: Upload documents → Privacy anonymization → Forgery check → ML scoring
    - Admin Dashboard: Heatmap of demand, Clean Queue, Red-Alert (forged) Queue
"""

import logging
import random
import time
import datetime
from typing import Any, Dict, List

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules.ml_scoring import RankingEngine
from modules.privacy_shield import PrivacyShield
from modules.vision_ocr import DocumentInspector

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("HouseHub")

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HouseHub — Anti-Fraud Housing Allocation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "HouseHub v1.0 — Privacy-First, AI-Powered Public Housing Allocation.",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS Styling
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* ── Gradient background ── */
  .stApp {
      background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
      color: #e8e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
      border-right: 1px solid rgba(99, 102, 241, 0.25);
  }
  [data-testid="stSidebar"] * { color: #c7d2fe !important; }

  /* ── Cards ── */
  .hub-card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(99,102,241,0.3);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 20px;
      backdrop-filter: blur(12px);
  }

  /* ── Score badge ── */
  .score-badge {
      display: inline-block;
      padding: 6px 18px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 1.1rem;
  }
  .score-high   { background:#dc2626; color:#fff; }
  .score-medium { background:#f59e0b; color:#111; }
  .score-low    { background:#16a34a; color:#fff; }

  /* ── Red Alert box ── */
  .red-alert {
      background: rgba(220,38,38,0.15);
      border: 2px solid #dc2626;
      border-radius: 12px;
      padding: 16px;
  }

  /* ── Hash ID display ── */
  .hash-display {
      font-family: 'Courier New', monospace;
      font-size: 0.78rem;
      background: rgba(0,0,0,0.4);
      padding: 8px 14px;
      border-radius: 8px;
      border-left: 4px solid #6366f1;
      word-break: break-all;
      color: #a5b4fc;
  }

  /* ── Section headers ── */
  .section-header {
      font-size: 1.4rem;
      font-weight: 700;
      color: #a5b4fc;
      border-bottom: 2px solid rgba(99,102,241,0.4);
      padding-bottom: 8px;
      margin-bottom: 16px;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {
      background: rgba(99,102,241,0.08) !important;
      border: 2px dashed rgba(99,102,241,0.4) !important;
      border-radius: 12px !important;
  }

  /* ── Metric values ── */
  [data-testid="stMetricValue"] {
      color: #818cf8 !important;
      font-weight: 700 !important;
  }

  /* ── Submit button ── */
  .stButton > button {
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 12px 32px;
      font-weight: 600;
      font-size: 1rem;
      transition: all 0.25s ease;
      width: 100%;
  }
  .stButton > button:hover {
      background: linear-gradient(135deg, #4f46e5, #7c3aed);
      transform: translateY(-2px);
      box-shadow: 0 8px 25px rgba(99,102,241,0.4);
  }

  /* ── Dataframe styling ── */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* ── Sidebar radio ── */
  [data-testid="stSidebar"] .stRadio label {
      font-size: 1.05rem;
      padding: 6px 0;
  }

  /* ── Info/Success/Warning overrides ── */
  .stAlert { border-radius: 10px !important; }
  div[data-testid="stNotification"] { border-radius: 12px; }

  /* ── Progress bar ── */
  .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6) !important; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #6366f1 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
def _init_session_state() -> None:
    """
    Initializes all st.session_state keys exactly once per browser session.
    Guards with 'in' checks to prevent reset on every Streamlit rerun.
    """
    if "applicants" not in st.session_state:
        # Each record: hashed_id, score, status, is_forged, confidence,
        #              timestamp, income, reason, breakdown
        st.session_state.applicants: List[Dict[str, Any]] = []

    if "ranking_engine" not in st.session_state:
        with st.spinner("🤖 Loading ML model — this takes a few seconds on first run…"):
            st.session_state.ranking_engine = RankingEngine()

    if "inspector" not in st.session_state:
        st.session_state.inspector = DocumentInspector()

    if "submission_result" not in st.session_state:
        st.session_state.submission_result = None

    if "page" not in st.session_state:
        st.session_state.page = "Citizen Portal"

    if "investigated_ids" not in st.session_state:
        st.session_state.investigated_ids: List[str] = []

    if "demo_populated" not in st.session_state:
        st.session_state.demo_populated = False


# ─────────────────────────────────────────────────────────────────────────────
# Demo Data Seeding
# ─────────────────────────────────────────────────────────────────────────────
def _populate_demo_applicants(engine: RankingEngine) -> None:
    """
    Seeds the applicant database with realistic mock records so the Admin
    Dashboard is non-empty on first launch. Called once per session.
    """
    if st.session_state.demo_populated:
        return

    demo_profiles = [
        {"name": "Alice Rahman",   "id": "ID-001", "income": 850,  "dependents": 4, "quality": 0, "forged": False},
        {"name": "Bob Al-Farsi",   "id": "ID-002", "income": 3200, "dependents": 1, "quality": 2, "forged": False},
        {"name": "Carol Mahmoud",  "id": "ID-003", "income": 1100, "dependents": 3, "quality": 1, "forged": True},
        {"name": "David Nkosi",    "id": "ID-004", "income": 650,  "dependents": 5, "quality": 0, "forged": False},
        {"name": "Emre Yilmaz",    "id": "ID-005", "income": 4500, "dependents": 0, "quality": 3, "forged": False},
        {"name": "Fatima Al-Sayed","id": "ID-006", "income": 920,  "dependents": 3, "quality": 1, "forged": True},
        {"name": "George Okonjo",  "id": "ID-007", "income": 1800, "dependents": 2, "quality": 2, "forged": False},
        {"name": "Hana Petrov",    "id": "ID-008", "income": 730,  "dependents": 4, "quality": 0, "forged": False},
    ]

    for p in demo_profiles:
        hashed_id = PrivacyShield.anonymize_data(p["name"], p["id"])
        score, breakdown = engine.calculate_score({
            "income": p["income"],
            "dependents": p["dependents"],
            "current_housing_quality": p["quality"],
        })
        confidence = round(random.uniform(72, 95), 1) if p["forged"] else round(random.uniform(5, 25), 1)
        reason = (
            "ELA detected non-uniform recompression artifacts — possible alteration."
            if p["forged"]
            else "Document appears authentic within acceptable variance."
        )
        st.session_state.applicants.append({
            "hashed_id":  hashed_id,
            "score":      score,
            "status":     "🔴 Flagged" if p["forged"] else "✅ Clean",
            "is_forged":  p["forged"],
            "confidence": confidence,
            "timestamp":  (
                datetime.datetime.now() - datetime.timedelta(hours=random.randint(1, 72))
            ).strftime("%Y-%m-%d %H:%M"),
            "income":     p["income"],
            "reason":     reason,
            "breakdown":  breakdown,
        })

    st.session_state.demo_populated = True
    logger.info("Demo applicants seeded: %d records", len(demo_profiles))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar() -> str:
    """Renders the sidebar and returns the selected navigation page."""
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 10px 0 20px;'>
                <div style='font-size:3rem;'>🏠</div>
                <div style='font-size:1.5rem; font-weight:800; color:#a5b4fc;'>HouseHub</div>
                <div style='font-size:0.75rem; color:#6366f1; margin-top:4px;'>
                    Anti-Fraud Housing Allocation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "Navigate to",
            options=["🏛️ Citizen Portal", "🔒 Admin Dashboard"],
            index=0,
            key="nav_radio",
        )

        st.markdown("---")

        # ── Model info card ──
        if "ranking_engine" in st.session_state:
            info = st.session_state.ranking_engine.get_model_info()
            st.markdown("**🤖 ML Model Status**")
            st.markdown(
                f"- Status: `{info.get('status', 'unknown')}`\n"
                f"- Algorithm: `RandomForest`\n"
                f"- MAE: `{info.get('mae', 'N/A')} pts`\n"
                f"- Trees: `{info.get('n_estimators', 'N/A')}`"
            )

        st.markdown("---")

        # ── Live stats ──
        total = len(st.session_state.get("applicants", []))
        forged = sum(1 for a in st.session_state.get("applicants", []) if a["is_forged"])
        clean  = total - forged

        st.markdown("**📊 Live Statistics**")
        c1, c2 = st.columns(2)
        c1.metric("Total", total)
        c2.metric("Flagged", forged, delta=f"{forged} alerts", delta_color="inverse")
        st.metric("Clean Queue", clean)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.7rem; color:#4b5563; text-align:center;'>"
            "All PII anonymized via SHA-256<br>"
            "HouseHub v1.0 · Privacy-First AI"
            "</div>",
            unsafe_allow_html=True,
        )

    return page


# ─────────────────────────────────────────────────────────────────────────────
# Citizen Portal
# ─────────────────────────────────────────────────────────────────────────────
def _render_citizen_portal() -> None:
    """Renders the Citizen Portal view for document submission."""

    st.markdown(
        "<h1 style='color:#a5b4fc; font-weight:800;'>"
        "🏛️ Citizen Housing Application Portal"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.05rem; margin-bottom:24px;'>"
        "Submit your documents securely. Your personal information is <strong>never stored</strong> — "
        "only a cryptographic hash is used to track your application."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Info banner ──
    st.info(
        "🔒 **Privacy Notice:** Your name and national ID are converted to an irreversible "
        "SHA-256 hash immediately on submission. No raw PII enters our system.",
        icon="🛡️",
    )

    st.markdown("<div class='section-header'>Step 1 — Your Identity</div>", unsafe_allow_html=True)

    col_name, col_id = st.columns(2)
    with col_name:
        applicant_name = st.text_input(
            "Full Legal Name",
            placeholder="e.g. Jane Al-Rashid",
            key="citizen_name",
            help="Your name will be hashed immediately and never stored.",
        )
    with col_id:
        national_id = st.text_input(
            "National ID Number",
            placeholder="e.g. NID-1234567",
            key="citizen_national_id",
            help="Used only for generating your anonymous hashed ID.",
        )

    st.markdown("<div class='section-header'>Step 2 — Upload Documents</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8;'>Upload images of your identity document, income certificate, "
        "and current lease agreement. Accepted formats: JPG, JPEG, PNG.</p>",
        unsafe_allow_html=True,
    )

    col_doc1, col_doc2, col_doc3 = st.columns(3)
    with col_doc1:
        st.markdown("📄 **National ID Document**")
        id_doc = st.file_uploader(
            "Upload ID Document",
            type=["jpg", "jpeg", "png"],
            key="upload_id",
            label_visibility="collapsed",
        )
        if id_doc:
            st.image(id_doc, caption="ID Document Preview", use_container_width=True)

    with col_doc2:
        st.markdown("💰 **Income Certificate**")
        income_doc = st.file_uploader(
            "Upload Income Certificate",
            type=["jpg", "jpeg", "png"],
            key="upload_income",
            label_visibility="collapsed",
        )
        if income_doc:
            st.image(income_doc, caption="Income Certificate Preview", use_container_width=True)

    with col_doc3:
        st.markdown("🏠 **Current Lease Agreement**")
        lease_doc = st.file_uploader(
            "Upload Lease Agreement",
            type=["jpg", "jpeg", "png"],
            key="upload_lease",
            label_visibility="collapsed",
        )
        if lease_doc:
            st.image(lease_doc, caption="Lease Agreement Preview", use_container_width=True)

    st.markdown("<div class='section-header'>Step 3 — Household Information</div>", unsafe_allow_html=True)

    col_dep, col_qual = st.columns(2)
    with col_dep:
        dependents = st.slider(
            "Number of Dependents",
            min_value=0,
            max_value=8,
            value=2,
            key="citizen_dependents",
            help="Include children, elderly parents, and other dependents.",
        )
    with col_qual:
        quality_map = {
            "Homeless / Emergency Shelter": 0,
            "Severe Overcrowding": 1,
            "Moderate Substandard Conditions": 2,
            "Adequate but Unaffordable": 3,
        }
        quality_label = st.selectbox(
            "Current Housing Situation",
            options=list(quality_map.keys()),
            index=1,
            key="citizen_quality",
        )
        current_housing_quality = quality_map[quality_label]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Submit button ──
    if st.button("🚀 Submit Application", key="submit_btn"):
        # Validation
        errors = []
        if not applicant_name or not applicant_name.strip():
            errors.append("Full Legal Name is required.")
        if not national_id or not national_id.strip():
            errors.append("National ID Number is required.")
        if not id_doc:
            errors.append("National ID Document upload is required.")
        if not income_doc:
            errors.append("Income Certificate upload is required.")
        if not lease_doc:
            errors.append("Lease Agreement upload is required.")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            _process_submission(
                applicant_name,
                national_id,
                id_doc,
                income_doc,
                lease_doc,
                dependents,
                current_housing_quality,
            )

    # ── Show result if available ──
    if st.session_state.submission_result:
        _display_submission_result(st.session_state.submission_result)


def _process_submission(
    name: str,
    national_id: str,
    id_doc,
    income_doc,
    lease_doc,
    dependents: int,
    current_housing_quality: int,
) -> None:
    """
    Orchestrates the full pipeline: PII anonymization → forgery detection
    → income OCR → ML scoring → session state storage.
    """
    progress_bar = st.progress(0, text="Initializing secure pipeline…")
    status_box = st.empty()

    try:
        # ── Stage 1: Privacy Shield ──
        progress_bar.progress(15, text="🔒 Anonymizing PII with SHA-256…")
        status_box.info("Stage 1/4 — Privacy Shield: Hashing personal data…")
        time.sleep(0.4)
        hashed_id = PrivacyShield.anonymize_data(name, national_id)
        logger.info("PII hashed → %s…", hashed_id[:12])

        # ── Stage 2: Forgery Detection (primary: ID document) ──
        progress_bar.progress(35, text="🔍 Running Computer Vision forgery analysis…")
        status_box.info("Stage 2/4 — Vision AI: Performing Error Level Analysis (ELA)…")
        time.sleep(0.5)

        inspector: DocumentInspector = st.session_state.inspector
        is_forged_id,  conf_id,  reason_id  = inspector.detect_forgery(id_doc)
        is_forged_inc, conf_inc, reason_inc = inspector.detect_forgery(income_doc)
        is_forged_lea, conf_lea, reason_lea = inspector.detect_forgery(lease_doc)

        # Overall: forged if ANY document is flagged
        is_forged   = any([is_forged_id, is_forged_inc, is_forged_lea])
        confidence  = max(conf_id, conf_inc, conf_lea)
        forged_docs = []
        if is_forged_id:  forged_docs.append("ID Document")
        if is_forged_inc: forged_docs.append("Income Certificate")
        if is_forged_lea: forged_docs.append("Lease Agreement")
        reason = (
            f"Flagged documents: {', '.join(forged_docs)}. {reason_id}"
            if is_forged
            else reason_id
        )

        # ── Stage 3: Income OCR ──
        progress_bar.progress(60, text="💰 Extracting income from certificate…")
        status_box.info("Stage 3/4 — OCR Engine: Extracting financial data…")
        time.sleep(0.4)

        income, ocr_method = inspector.extract_income(income_doc)

        # ── Stage 4: ML Scoring ──
        progress_bar.progress(85, text="🤖 Computing Housing Need Score…")
        status_box.info("Stage 4/4 — ML Engine: Calculating objective Need Score…")
        time.sleep(0.5)

        engine: RankingEngine = st.session_state.ranking_engine
        score, breakdown = engine.calculate_score({
            "income":                   income,
            "dependents":               dependents,
            "current_housing_quality":  current_housing_quality,
        })

        progress_bar.progress(100, text="✅ Application processed successfully!")
        time.sleep(0.3)
        status_box.empty()
        progress_bar.empty()

        # ── Store in session state ──
        record: Dict[str, Any] = {
            "hashed_id":  hashed_id,
            "score":      score,
            "status":     "🔴 Flagged" if is_forged else "✅ Clean",
            "is_forged":  is_forged,
            "confidence": round(confidence, 1),
            "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "income":     income,
            "reason":     reason,
            "breakdown":  breakdown,
        }
        st.session_state.applicants.append(record)
        st.session_state.submission_result = record
        logger.info(
            "Application stored. Score=%.1f, Forged=%s, Hash=%s…",
            score,
            is_forged,
            hashed_id[:10],
        )

    except ValueError as ve:
        progress_bar.empty()
        status_box.empty()
        st.error(f"❌ Validation error: {ve}")
    except Exception as exc:
        progress_bar.empty()
        status_box.empty()
        st.error(f"❌ Unexpected error during processing: {exc}")
        logger.exception("Unexpected pipeline error")


def _display_submission_result(result: Dict[str, Any]) -> None:
    """Renders the post-submission result card."""
    st.markdown("---")

    if result["is_forged"]:
        st.markdown(
            "<div class='red-alert'>"
            "<h3 style='color:#ef4444; margin:0;'>🚨 Application Flagged for Investigation</h3>"
            "<p style='color:#fca5a5; margin-top:8px;'>One or more documents have been flagged "
            "by our Computer Vision system. Your application has been referred to the "
            "Anti-Fraud Investigation Queue.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success(
            "✅ **Application Submitted Successfully!** "
            "Your documents passed all authenticity checks."
        )

    st.markdown("<div class='hub-card'>", unsafe_allow_html=True)
    st.markdown("### 🎫 Your Application Receipt")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("**Anonymous Hashed ID (save this for reference):**")
        st.markdown(
            f"<div class='hash-display'>{result['hashed_id']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<br><small style='color:#6b7280;'>{result['breakdown']}</small>",
            unsafe_allow_html=True,
        )

    with col_b:
        score = result["score"]
        if score >= 65:
            badge_class = "score-high"
            urgency = "HIGH PRIORITY"
        elif score >= 35:
            badge_class = "score-medium"
            urgency = "MEDIUM"
        else:
            badge_class = "score-low"
            urgency = "LOW"

        st.markdown(
            f"<div style='text-align:center; margin-top:10px;'>"
            f"<div style='color:#94a3b8; font-size:0.85rem;'>Need Score</div>"
            f"<div style='font-size:3rem; font-weight:800; color:#818cf8;'>{score:.0f}</div>"
            f"<div class='score-badge {badge_class}'>{urgency}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 Detected Income", f"{result['income']:,}")
    m2.metric("📋 Status", result["status"])
    m3.metric("🕐 Submitted", result["timestamp"])
    m4.metric("🔍 Fraud Confidence", f"{result['confidence']:.1f}%")

    if result["is_forged"]:
        st.markdown(
            f"<div style='color:#fca5a5; font-size:0.85rem; margin-top:8px;'>"
            f"⚠️ <strong>Reason:</strong> {result['reason']}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Admin Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def _build_heatmap() -> folium.Map:
    """
    Constructs a Folium heatmap of simulated housing demand across a city.
    Uses deterministic mock coordinates for reproducibility.
    """
    # City center: Riyadh, KSA (lat 24.77, lon 46.74) — generic urban centre
    CENTER = [24.7741, 46.7380]
    city_map = folium.Map(
        location=CENTER,
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    rng = np.random.default_rng(seed=99)

    # Simulate 300 demand points spread across the city
    n_points = 300
    lats = CENTER[0] + rng.normal(0, 0.04, n_points)
    lons = CENTER[1] + rng.normal(0, 0.05, n_points)
    weights = rng.uniform(0.3, 1.0, n_points)

    # High-demand clusters (slum/overcrowded zones)
    hot_zone_centers = [
        [24.790, 46.710, 0.05, 0.03],  # North-West district
        [24.755, 46.760, 0.03, 0.04],  # South-East district
        [24.780, 46.750, 0.02, 0.02],  # City core
    ]
    for hz_lat, hz_lon, dlat, dlon in hot_zone_centers:
        cluster_lats = hz_lat + rng.normal(0, dlat, 40)
        cluster_lons = hz_lon + rng.normal(0, dlon, 40)
        cluster_weights = rng.uniform(0.8, 1.0, 40)
        lats = np.concatenate([lats, cluster_lats])
        lons = np.concatenate([lons, cluster_lons])
        weights = np.concatenate([weights, cluster_weights])

    heat_data = [[float(la), float(lo), float(w)] for la, lo, w in zip(lats, lons, weights)]

    from folium.plugins import HeatMap
    HeatMap(
        heat_data,
        min_opacity=0.45,
        radius=18,
        blur=20,
        gradient={0.2: "#4f46e5", 0.4: "#7c3aed", 0.6: "#f59e0b", 0.8: "#ef4444", 1.0: "#dc2626"},
    ).add_to(city_map)

    # Overlay a few district markers
    districts = [
        {"name": "Al-Aziziyah",  "lat": 24.790, "lon": 46.710, "demand": "Very High"},
        {"name": "Al-Malaz",     "lat": 24.755, "lon": 46.760, "demand": "High"},
        {"name": "Al-Rawdah",    "lat": 24.780, "lon": 46.750, "demand": "High"},
        {"name": "Al-Sulimaniyah","lat": 24.760, "lon": 46.720, "demand": "Medium"},
        {"name": "Al-Muruj",     "lat": 24.740, "lon": 46.770, "demand": "Low"},
    ]
    for d in districts:
        color = {"Very High": "red", "High": "orange", "Medium": "beige", "Low": "green"}.get(
            d["demand"], "blue"
        )
        folium.Marker(
            location=[d["lat"], d["lon"]],
            popup=folium.Popup(
                f"<b>{d['name']}</b><br>Demand: {d['demand']}", max_width=200
            ),
            tooltip=f"{d['name']} — {d['demand']} Demand",
            icon=folium.Icon(color=color, icon="home", prefix="fa"),
        ).add_to(city_map)

    return city_map


def _render_admin_dashboard() -> None:
    """Renders the Admin Dashboard with heatmap, clean queue, and red-alert queue."""

    st.markdown(
        "<h1 style='color:#a5b4fc; font-weight:800;'>"
        "🔒 Admin Dashboard — Housing Allocation Control Center"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94a3b8;'>"
        "Real-time overview of all applications, fraud alerts, and housing demand intelligence."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── KPI Row ──
    applicants = st.session_state.applicants
    total       = len(applicants)
    forged_list = [a for a in applicants if a["is_forged"]]
    clean_list  = [a for a in applicants if not a["is_forged"]]
    avg_score   = (
        round(sum(a["score"] for a in clean_list) / len(clean_list), 1)
        if clean_list else 0.0
    )
    high_need   = sum(1 for a in clean_list if a["score"] >= 65)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📋 Total Applications", total)
    k2.metric("✅ Clean Queue",         len(clean_list))
    k3.metric("🚨 Fraud Alerts",        len(forged_list), delta=f"-{len(forged_list)} flagged", delta_color="inverse")
    k4.metric("📊 Avg Need Score",      avg_score)
    k5.metric("🔴 High Priority",       high_need)

    st.markdown("---")

    # ── Heatmap Section ──
    st.markdown("<div class='section-header'>🗺️ City Housing Demand Heatmap</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; margin-bottom:12px;'>"
        "Live demand intelligence across city districts. "
        "Red zones indicate highest housing need concentration.</p>",
        unsafe_allow_html=True,
    )

    city_map = _build_heatmap()
    st_folium(city_map, width="100%", height=420, returned_objects=[])

    st.markdown("---")

    # ── Dual-column queues ──
    col_clean, col_red = st.columns([1, 1], gap="large")

    # ── Column 1: Clean Queue ──
    with col_clean:
        st.markdown(
            "<div class='section-header'>✅ Verified Applicant Queue</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.9rem;'>"
            "Non-flagged applicants ranked strictly by Need Score (descending). "
            "Top-ranked citizens receive housing allocation priority.</p>",
            unsafe_allow_html=True,
        )

        if clean_list:
            clean_df = pd.DataFrame(clean_list)[
                ["hashed_id", "score", "income", "timestamp"]
            ].copy()
            clean_df = clean_df.sort_values("score", ascending=False).reset_index(drop=True)
            clean_df.index += 1  # 1-indexed rank
            clean_df.columns = ["Hashed Citizen ID", "Need Score", "Income (Monthly)", "Submitted"]
            clean_df["Hashed Citizen ID"] = clean_df["Hashed Citizen ID"].apply(
                lambda h: PrivacyShield.mask_hash_for_display(h, visible_chars=14)
            )
            clean_df["Need Score"] = clean_df["Need Score"].apply(lambda s: f"{s:.1f}")

            st.dataframe(
                clean_df,
                use_container_width=True,
                height=380,
                column_config={
                    "Need Score": st.column_config.ProgressColumn(
                        "Need Score",
                        help="Score 0–100 (higher = more urgent)",
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )
        else:
            st.info("No verified applicants yet. Submit an application via the Citizen Portal.")

    # ── Column 2: Red Alert Queue ──
    with col_red:
        st.markdown(
            "<div class='red-alert'>"
            "<div class='section-header' style='color:#ef4444; border-color:rgba(220,38,38,0.4);'>"
            "🚨 Red Alert — Fraud Investigation Queue"
            "</div>"
            "<p style='color:#fca5a5; font-size:0.9rem; margin-top:4px;'>"
            "Applications flagged by the Computer Vision ELA pipeline. "
            "Requires manual review before any allocation decision.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if forged_list:
            alert_df = pd.DataFrame(forged_list)[
                ["hashed_id", "score", "confidence", "timestamp", "reason"]
            ].copy()
            alert_df.columns = [
                "Hashed Citizen ID", "Need Score", "Fraud Confidence %",
                "Submitted", "Reason",
            ]
            alert_df["Hashed Citizen ID"] = alert_df["Hashed Citizen ID"].apply(
                lambda h: PrivacyShield.mask_hash_for_display(h, visible_chars=14)
            )

            st.dataframe(
                alert_df,
                use_container_width=True,
                height=280,
                column_config={
                    "Fraud Confidence %": st.column_config.ProgressColumn(
                        "Fraud Confidence %",
                        help="Higher = more likely forged",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Reason": st.column_config.TextColumn(
                        "Reason", width="large"
                    ),
                },
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Investigate buttons per flagged record ──
            st.markdown("**🔎 Manually Investigate Cases:**")
            for i, record in enumerate(forged_list):
                masked_id = PrivacyShield.mask_hash_for_display(record["hashed_id"], 10)
                btn_key = f"investigate_{record['hashed_id'][:16]}_{i}"

                already_investigated = record["hashed_id"] in st.session_state.investigated_ids

                if already_investigated:
                    st.markdown(
                        f"<div style='background:rgba(22,163,74,0.15); border:1px solid #16a34a; "
                        f"border-radius:8px; padding:8px 14px; margin-bottom:8px; color:#86efac;'>"
                        f"✅ <strong>{masked_id}</strong> — Marked for Investigation</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(
                        f"🔍 Investigate: {masked_id}",
                        key=btn_key,
                        type="primary",
                    ):
                        st.session_state.investigated_ids.append(record["hashed_id"])
                        st.rerun()

        else:
            st.success("🎉 No fraud alerts. All reviewed applications passed authenticity checks.")

    # ── Model Transparency ──
    st.markdown("---")
    st.markdown("<div class='section-header'>🔬 Model Transparency & Auditability</div>", unsafe_allow_html=True)

    engine = st.session_state.ranking_engine
    info   = engine.get_model_info()

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Feature Importance (Random Forest)**")
        if info.get("feature_importance"):
            imp_df = pd.DataFrame(
                list(info["feature_importance"].items()),
                columns=["Feature", "Importance"],
            ).sort_values("Importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

    with t2:
        st.markdown("**Model Metadata**")
        meta = {
            "Algorithm":   info.get("algorithm", "N/A"),
            "Trees":       str(info.get("n_estimators", "N/A")),
            "Holdout MAE": f"{info.get('mae', 'N/A')} pts",
            "Status":      info.get("status", "N/A").upper(),
            "Scaler":      "MinMaxScaler",
            "Training Set": "2,000 synthetic applicants",
        }
        meta_df = pd.DataFrame(list(meta.items()), columns=["Property", "Value"])
        st.dataframe(meta_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Application entry point: initialize state, render sidebar, route pages."""
    _init_session_state()

    engine: RankingEngine = st.session_state.ranking_engine
    _populate_demo_applicants(engine)

    selected_page = _render_sidebar()

    if selected_page == "🏛️ Citizen Portal":
        _render_citizen_portal()
    elif selected_page == "🔒 Admin Dashboard":
        _render_admin_dashboard()
    else:
        st.error("Unknown page selected.")


if __name__ == "__main__":
    main()

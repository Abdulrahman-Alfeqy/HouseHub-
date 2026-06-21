"""
app.py
------
HouseHub+ — Anti-Fraud Public Housing Allocation System
Streamlit Entry Point
"""

import logging
import streamlit as st
import folium
from streamlit_folium import st_folium
import os
import pandas as pd

from modules.database import DatabaseManager
from modules.models import IntelligenceEngine
from modules.vision import DocumentInspector
from modules.utils import PrivacyShield

# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HouseHub+")

# Configure Page
st.set_page_config(
    page_title="HouseHub+ | Smart Housing Allocation",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #e8e8f0; }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); border-right: 1px solid rgba(99, 102, 241, 0.25); }
  [data-testid="stSidebar"] * { color: #c7d2fe !important; }
  .hub-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(99,102,241,0.3); border-radius: 16px; padding: 24px; margin-bottom: 20px; backdrop-filter: blur(12px); }
  .score-badge { display: inline-block; padding: 6px 18px; border-radius: 999px; font-weight: 700; font-size: 1.1rem; }
  .score-high { background:#dc2626; color:#fff; }
  .score-medium { background:#f59e0b; color:#111; }
  .score-low { background:#16a34a; color:#fff; }
  .red-alert { background: rgba(220,38,38,0.15); border: 2px solid #dc2626; border-radius: 12px; padding: 16px; }
  .section-header { font-size: 1.4rem; font-weight: 700; color: #a5b4fc; border-bottom: 2px solid rgba(99,102,241,0.4); padding-bottom: 8px; margin-bottom: 16px; }
  .stButton > button { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; border: none; border-radius: 10px; padding: 12px 32px; font-weight: 600; width: 100%; }
  .stButton > button:hover { background: linear-gradient(135deg, #4f46e5, #7c3aed); transform: translateY(-2px); box-shadow: 0 8px 25px rgba(99,102,241,0.4); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
@st.cache_resource
def get_db():
    return DatabaseManager()

@st.cache_resource
def get_ai_engine():
    return IntelligenceEngine()

@st.cache_resource
def get_vision_engine():
    return DocumentInspector()

if 'db' not in st.session_state:
    st.session_state.db = get_db()
if 'ai_engine' not in st.session_state:
    st.session_state.ai_engine = get_ai_engine()
if 'vision_engine' not in st.session_state:
    st.session_state.vision_engine = get_vision_engine()
if 'citizen_data' not in st.session_state:
    st.session_state.citizen_data = {}

db: DatabaseManager = st.session_state.db
ai_engine: IntelligenceEngine = st.session_state.ai_engine
vision_engine: DocumentInspector = st.session_state.vision_engine

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🏠 HouseHub+</h2>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏛️ Citizen Portal", "🔒 Admin Dashboard"])
    st.markdown("---")
    st.markdown("**🛡️ Responsible AI Principles**")
    st.markdown("- **Privacy-by-Design**: Names encrypted & IDs hashed.")
    st.markdown("- **Human-in-the-Loop**: AI scores priority, humans make decisions.")

# ─────────────────────────────────────────────────────────────────────────────
# CITIZEN PORTAL
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏛️ Citizen Portal":
    st.title("🏛️ Citizen Housing Application Portal")
    st.info("🔒 **Privacy Notice:** Your name and national ID are immediately hashed or encrypted on submission. No raw PII is visible to the ML model or during initial admin review.")

    # 1. Identity & Auto-Reload
    st.markdown("<div class='section-header'>Step 1 — Identity Verification</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        nid = st.text_input("National ID Number", key="nid", help="Enter to retrieve previous application if applicable.")
    with col2:
        name = st.text_input("Full Legal Name", key="cname")

    if nid:
        user_token = PrivacyShield.hash_national_id(nid)
        existing = db.get_applicant_by_token(user_token)
        if existing and not st.session_state.get('reloaded'):
            st.success("✅ Previous application found. Reloading data...")
            st.session_state.citizen_data = existing
            st.session_state.reloaded = True

    # Pre-fill from session if reloaded
    c_data = st.session_state.get('citizen_data', {})
    
    # 2. Demographics
    st.markdown("<div class='section-header'>Step 2 — Household Details</div>", unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        family_size = st.number_input("Family Size", min_value=1, max_value=20, value=int(c_data.get('family_size', 1)))
    with c4:
        marital_options = ["Single", "Married", "Divorced", "Widowed"]
        m_idx = marital_options.index(c_data.get('marital_status', 'Single')) if c_data.get('marital_status') in marital_options else 0
        marital_status = st.selectbox("Marital Status", marital_options, index=m_idx)
    with c5:
        special_conditions = st.checkbox("Special Medical/Social Conditions?", value=bool(c_data.get('special_conditions', False)))

    # 3. Location Tracking
    st.markdown("<div class='section-header'>Step 3 — Current Location (GPS)</div>", unsafe_allow_html=True)
    
    # Default Center: Riyadh
    map_center = [c_data.get('lat', 24.7741), c_data.get('lon', 46.7380)]
    m = folium.Map(location=map_center, zoom_start=12)
    m.add_child(folium.LatLngPopup())
    if 'lat' in c_data:
        folium.Marker(map_center, popup="Saved Location").add_to(m)
        
    map_data = st_folium(m, height=300, width="100%")
    
    lat, lon = map_center[0], map_center[1]
    if map_data['last_clicked']:
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        st.success(f"📍 Location Captured: Lat {lat:.4f}, Lon {lon:.4f}")

    # 4. Documents
    st.markdown("<div class='section-header'>Step 4 — Document Upload</div>", unsafe_allow_html=True)
    st.markdown("<small>Supported formats: JPG, PNG. Documents will be automatically scanned by AI.</small>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        id_doc = st.file_uploader("Upload National ID", type=['jpg', 'jpeg', 'png'])
    with d2:
        income_doc = st.file_uploader("Upload Proof of Income", type=['jpg', 'jpeg', 'png'])
    with d3:
        lease_doc = st.file_uploader("Upload Lease Agreement", type=['jpg', 'jpeg', 'png'])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Securely Submit Application"):
        if not (nid and name and id_doc and income_doc and lease_doc):
            st.error("❌ Please complete all fields and upload all documents.")
        else:
            with st.spinner("Analyzing documents & computing Priority Score..."):
                # Vision AI Check
                id_bytes = id_doc.read()
                inc_bytes = income_doc.read()
                lease_bytes = lease_doc.read()
                
                f_id, c_id, r_id = vision_engine.detect_forgery(id_bytes)
                f_inc, c_inc, r_inc = vision_engine.detect_forgery(inc_bytes)
                f_lea, c_lea, r_lea = vision_engine.detect_forgery(lease_bytes)
                
                is_forged = f_id or f_inc or f_lea
                forgery_conf = max(c_id, c_inc, c_lea)
                reasons = []
                if f_id: reasons.append(f"ID: {r_id}")
                if f_inc: reasons.append(f"Income: {r_inc}")
                if f_lea: reasons.append(f"Lease: {r_lea}")
                forgery_reason = " | ".join(reasons) if is_forged else ""
                
                # Extract numbers
                extracted_income = vision_engine.extract_numbers(inc_bytes)
                extracted_rent = vision_engine.extract_numbers(lease_bytes)
                
                if extracted_income == 0.0 or extracted_rent == 0.0:
                    is_forged = True
                    forgery_reason += " | OCR Validation Failed: Unreadable values."
                    st.warning("⚠️ OCR could not clearly extract income or rent. Routing to Red Alert Queue.")
                
                # Default to 0 if extraction fails
                inc_val = extracted_income if extracted_income > 0 else float(c_data.get('income', 0))
                rent_val = extracted_rent if extracted_rent > 0 else float(c_data.get('rent', 0))
                
                if inc_val == 0 or rent_val == 0:
                    st.warning("⚠️ OCR could not clearly extract income or rent. Proceeding with default values.")
                
                special_cond_int = 1 if special_conditions else 0
                
                # ML Scoring
                score, tier, breakdown = ai_engine.calculate_priority(
                    income=inc_val, 
                    rent=rent_val, 
                    family_size=family_size, 
                    special_conditions=special_cond_int, 
                    lat=lat, 
                    lon=lon
                )
                
                # Save to DB
                cluster = ai_engine.get_district_cluster(lat, lon)
                
                token = db.upsert_applicant(
                    name=name, national_id=nid, income=inc_val, rent=rent_val,
                    family_size=family_size, marital_status=marital_status,
                    special_conditions=special_cond_int, lat=lat, lon=lon,
                    district_cluster=cluster, need_score=score, tier=tier,
                    is_forged=is_forged, forgery_confidence=forgery_conf,
                    forgery_reason=forgery_reason
                )
                
                st.success("✅ Application Submitted Successfully!")
                st.markdown(f"**Your Anonymous ID:** `{token}`")
                st.markdown(f"**Priority Score:** `{score}` ({tier} Tier)")
                
                if is_forged:
                    st.error("🚨 Notice: Our system flagged inconsistencies in your documents. It will be reviewed by an investigator.")
                    # Save images for Human-in-the-Loop review
                    os.makedirs("data/flagged_docs", exist_ok=True)
                    with open(f"data/flagged_docs/{token}_id.jpg", "wb") as f:
                        f.write(id_bytes)
                    with open(f"data/flagged_docs/{token}_income.jpg", "wb") as f:
                        f.write(inc_bytes)
                    with open(f"data/flagged_docs/{token}_lease.jpg", "wb") as f:
                        f.write(lease_bytes)

# ─────────────────────────────────────────────────────────────────────────────
# ADMIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔒 Admin Dashboard":
    st.title("🔒 Admin Dashboard")
    
    # Admin Authentication
    if not st.session_state.get("admin_authenticated", False):
        st.markdown("### Restricted Access")
        with st.form("admin_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
                ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")
                if username == ADMIN_USER and password == ADMIN_PASS:
                    st.session_state.admin_authenticated = True
                    st.success("Authenticated successfully!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        st.stop()

    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.markdown("Real-time Housing Intelligence & Queue Management.")
    with col_logout:
        if st.button("Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()
    
    # KPIs
    clean_q = db.get_queue(forged_only=False)
    red_q = db.get_queue(forged_only=True)
    
    k1, k2, k3 = st.columns(3)
    k1.metric("✅ Clean Applications", len(clean_q))
    k2.metric("🚨 Red Alert (Suspicious)", len(red_q))
    k3.metric("🗺️ Total Valid Mapped Points", len(db.get_heatmap_data()))
    
    st.markdown("---")
    
    # Sub-tabs
    tab1, tab2, tab3 = st.tabs(["✅ Clean Queue", "🚨 Red Alert Queue", "🗺️ Geo-AI Heatmap"])
    
    def render_queue(queue_data, is_red=False):
        if not queue_data:
            st.info("Queue is empty.")
            return
            
        for app in queue_data:
            with st.container():
                css_class = 'red-alert' if is_red else 'hub-card'
                st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
                
                col_info, col_act = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**Anonymous Token:** `{app['user_token']}`")
                    st.markdown(f"**Score:** `{app['need_score']}` | **Tier:** `{app['tier']}` | **Status:** `{app['status']}`")
                    st.markdown(f"**Details:** Income: {app['income']}, Rent: {app['rent']}, Family: {app['family_size']}, Cluster: {app['district_cluster']}")
                    if is_red:
                        st.markdown(f"**🔴 Forgery Reason:** {app['forgery_reason']} (Confidence: {app['forgery_confidence']:.2f}%)")
                        id_path = f"data/flagged_docs/{app['user_token']}_id.jpg"
                        inc_path = f"data/flagged_docs/{app['user_token']}_income.jpg"
                        lease_path = f"data/flagged_docs/{app['user_token']}_lease.jpg"
                        
                        cols = st.columns(3)
                        if os.path.exists(id_path): cols[0].image(id_path, caption="National ID", use_container_width=True)
                        if os.path.exists(inc_path): cols[1].image(inc_path, caption="Income Doc", use_container_width=True)
                        if os.path.exists(lease_path): cols[2].image(lease_path, caption="Lease Doc", use_container_width=True)
                    st.markdown(f"*Rejections: {app['rejection_count']}*")
                    
                with col_act:
                    if app['status'] in ['Pending', 'Rejected']:
                        if st.button("✅ Approve", key=f"app_{app['user_token']}"):
                            real_name = db.approve_applicant(app['user_token'])
                            st.success(f"Approved! Citizen Name: **{real_name}**")
                            st.rerun()
                        
                        reject_reason = st.text_input("Rejection Reason", key=f"r_res_{app['user_token']}")
                        if st.button("❌ Reject", key=f"rej_{app['user_token']}"):
                            if not reject_reason:
                                st.warning("Please provide a reason to reject.")
                            else:
                                res = db.reject_applicant(app['user_token'], reject_reason)
                                st.info(f"Updated: {res['status']}")
                                st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    with tab1:
        st.markdown("### Process clean applications ordered by Priority Need.")
        render_queue(clean_q, is_red=False)
        
    with tab2:
        st.markdown("### Applications flagged by Vision AI Error Level Analysis.")
        render_queue(red_q, is_red=True)
        
    with tab3:
        st.markdown("### AI District Clustering (Demand Heatmap)")
        heatmap_data = db.get_heatmap_data()
        if heatmap_data:
            from folium.plugins import HeatMap
            map_center = [24.7741, 46.7380] # Default
            city_map = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB dark_matter")
            
            heat_points = []
            for pt in heatmap_data:
                # Weight by need_score
                weight = pt['need_score'] / 100.0
                heat_points.append([pt['lat'], pt['lon'], weight])
                
            HeatMap(heat_points, radius=15, blur=15, max_zoom=1).add_to(city_map)
            st_folium(city_map, width="100%", height=500)
        else:
            st.info("No approved or pending safe locations to map.")

# app.py
# FoodBridge – Clean Full-Stack (Ops + AI) Streamlit App
# Neat UI, compact navigation, single Analytics page with tabs
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import networkx as nx

from geopy.distance import geodesic
from datetime import datetime as dt, timedelta, time
import random

# ===================== THEME & PAGE CONFIG =====================
PRIMARY = "#186049"     # deep green
ACCENT = "#0EA5E9"      # blue accent
BG_SOFT = "#F5FAF7"     # soft greenish background
CARD_BG = "#FFFFFF"

st.set_page_config(
    page_title="🍲 FoodBridge",
    page_icon="🍲",
    layout="wide",
)

st.markdown(
    f"""
<style>
  .appview-container .main .block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    }}
  :root {{
    --primary:{PRIMARY};
    --accent:{ACCENT};
  }}
  .metric-card {{
    background: linear-gradient(135deg, var(--primary), #3aa981);
    padding: 1rem;
    border-radius: 16px;
    color: #fff;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,.06);
  }}
  .prediction-card {{
    background: linear-gradient(135deg, #f093fb, #f5576c);
    padding: 1rem;
    border-radius: 16px;
    color: #fff;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,.06);
  }}
  .soft-section {{
    background: {CARD_BG};
    border: 1px solid #E8EEF0;
    border-radius: 14px;
    padding: 1rem 1rem;
  }}
  .food-card {{
    border: 1px solid #E9EEF3;
    border-radius: 14px;
    padding: 1rem;
    margin: 12px 0;
    background: #FCFEFF;
  }}
  .pill {{
    display: inline-block;
    padding: .25rem .7rem;
    border-radius: 999px;
    font-size: .8rem;
    }}
  .pill-available {{ background:#10B981; color:#fff; }}
  .pill-claimed {{ background:#F59E0B; color:#fff; }}
  .pill-delivered {{ background:#3B82F6; color:#fff; }}
  .alert-urgent {{
    background:#FFF5F5;
    border-left: 6px solid #EF4444;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin: 8px 0 4px;
  }}
  .title-soft {{
    color: var(--primary);
    font-weight: 800;
    letter-spacing: .2px;
  }}
</style>
""",
    unsafe_allow_html=True,
)

# ===================== SESSION BOOTSTRAP =====================
ss = st.session_state
defaults = {
    "food_donations": [],
    "volunteers": [],
    "organizations": [],
    "current_user_type": "Visitor",
    "current_user": None,
    "ai_ready": False,
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v


def init_data():
    """Seed some lightweight sample data for quick demos."""
    if not ss.food_donations:
        now = dt.now()
        ss.food_donations += [
            {
                "id": "FD001",
                "donor_name": "Green Valley Restaurant",
                "donor_phone": "+91-9876543210",
                "donor_address": "Race Course Road, Coimbatore",
                "donor_lat": 11.0168,
                "donor_lng": 76.9558,
                "food_type": "Cooked Meals",
                "quantity": "50 servings",
                "description": "Fresh vegetarian meals",
                "expiry_time": now + timedelta(hours=4),
                "status": "Available",
                "created_at": now - timedelta(hours=1),
                "claimed_by": None,
                "delivered_to": None,
                "special_instructions": "Contains dairy",
                "food_category": "Vegetarian",
            },
            {
                "id": "FD002",
                "donor_name": "City Bakery",
                "donor_phone": "+91-9876543211",
                "donor_address": "RS Puram, Coimbatore",
                "donor_lat": 11.0096,
                "donor_lng": 76.9750,
                "food_type": "Bakery Items",
                "quantity": "30 pieces",
                "description": "Bread & pastries (today)",
                "expiry_time": now + timedelta(hours=12),
                "status": "Claimed",
                "created_at": now - timedelta(hours=3),
                "claimed_by": "VOL001",
                "delivered_to": None,
                "special_instructions": "Best consumed fresh",
                "food_category": "Vegetarian",
            },
            {
                "id": "FD003",
                "donor_name": "Spice Garden Hotel",
                "donor_phone": "+91-9876543212",
                "donor_address": "Gandhipuram, Coimbatore",
                "donor_lat": 11.0183,
                "donor_lng": 76.9725,
                "food_type": "Cooked Meals",
                "quantity": "80 servings",
                "description": "Mixed veg & non-veg meals",
                "expiry_time": now + timedelta(hours=2),
                "status": "Available",
                "created_at": now - timedelta(minutes=30),
                "claimed_by": None,
                "delivered_to": None,
                "special_instructions": "Urgent pickup",
                "food_category": "Mixed",
            },
        ]
    if not ss.volunteers:
        ss.volunteers += [
            {
                "id": "VOL001",
                "name": "Rajesh Kumar",
                "phone": "+91-9876543220",
                "area": "Race Course",
                "vehicle": "Two Wheeler",
                "availability": "Available",
                "rating": 4.8,
                "completed_deliveries": 25,
                "location_lat": 11.0150,
                "location_lng": 76.9600,
            },
            {
                "id": "VOL002",
                "name": "Priya Sharma",
                "phone": "+91-9876543221",
                "area": "RS Puram",
                "vehicle": "Four Wheeler",
                "availability": "Busy",
                "rating": 4.9,
                "completed_deliveries": 42,
                "location_lat": 11.0100,
                "location_lng": 76.9750,
            },
        ]
    if not ss.organizations:
        ss.organizations += [
            {
                "id": "ORG001",
                "name": "Hope Foundation",
                "contact_person": "Maria Joseph",
                "phone": "+91-9876543230",
                "address": "Peelamedu, Coimbatore",
                "type": "NGO",
                "beneficiaries": 150,
                "location_lat": 11.0296,
                "location_lng": 76.9378,
                "requirements": "Veg meals preferred; elderly & children",
            },
            {
                "id": "ORG002",
                "name": "Street Children Shelter",
                "contact_person": "David Wilson",
                "phone": "+91-9876543231",
                "address": "Saibaba Colony, Coimbatore",
                "type": "Shelter Home",
                "beneficiaries": 80,
                "location_lat": 11.0240,
                "location_lng": 76.9350,
                "requirements": "Any food; urgent breakfast",
            },
        ]


init_data()

# ===================== AI / DATA-SCI ENGINE =====================
class FoodBridgeAI:
    """
    Compact AI backend:
    - Generates synthetic donors (300), donations (365 days), volunteers
    - Trains RandomForest (demand), XGBClassifier (success), KMeans (segments), IsolationForest (anomalies)
    """

    def __init__(self):
        self._init()
        self._train()

    def _init(self):
        np.random.seed(42)

        # donors
        donor_types = ["Restaurant", "Bakery", "Grocery Store", "Catering", "Hotel", "Cafe"]
        locs = [
            (12.9716, 77.5946),
            (13.0827, 80.2707),
            (11.0168, 76.9558),
            (15.2993, 74.1240),
            (17.3850, 78.4867),
        ]
        donors = []
        for i in range(300):
            lat, lon = locs[i % len(locs)]
            t = np.random.choice(donor_types)
            donors.append(
                {
                    "donor_id": f"D_{i:03d}",
                    "name": f"{t}_{i}",
                    "type": t,
                    "latitude": lat + np.random.normal(0, 0.08),
                    "longitude": lon + np.random.normal(0, 0.08),
                    "size_score": np.random.uniform(1, 10),
                    "sustainability_score": np.random.uniform(1, 10),
                    "avg_daily_footfall": np.random.randint(50, 1000),
                    "years_operating": np.random.randint(1, 20),
                    "donation_frequency": np.random.choice(["Daily", "Weekly", "Monthly"]),
                    "preferred_pickup_time": np.random.choice(["Morning", "Afternoon", "Evening"]),
                    "max_donation_capacity": np.random.randint(10, 200),
                }
            )
        self.donors = pd.DataFrame(donors)

        # donations – 1 year for faster startup
        rows = []
        start = dt.now() - timedelta(days=365)
        cats = ["Prepared Food", "Bakery Items", "Fruits", "Vegetables", "Dairy", "Packaged Food", "Beverages"]
        urg = {
            "Prepared Food": 4,
            "Dairy": 6,
            "Bakery Items": 24,
            "Fruits": 48,
            "Vegetables": 72,
            "Packaged Food": 168,
            "Beverages": 240,
        }
        for d in range(365):
            cur = start + timedelta(days=d)
            mult = (1.4 if cur.weekday() >= 5 else 1.0) * (1.15 if cur.month in [11, 12, 1] else 1.0)
            for _ in range(int(np.random.poisson(18) * mult)):
                did = np.random.choice(self.donors["donor_id"])
                di = self.donors[self.donors.donor_id == did].iloc[0]
                ft = np.random.choice(cats)
                base = di.size_score * np.random.uniform(0.5, 2)
                qty = max(1, np.random.normal(base, base * 0.3))
                hrs = max(1, np.random.normal(urg[ft], urg[ft] * 0.3))
                p = min(0.95, 0.5 + (di.sustainability_score / 20) + (1 / max(1, hrs / 24)) * 0.3)
                rows.append(
                    {
                        "donation_id": f"DN_{len(rows):06d}",
                        "donor_id": did,
                        "date": cur,
                        "food_type": ft,
                        "quantity_kg": round(qty, 2),
                        "hours_to_expire": round(hrs, 1),
                        "pickup_success": np.random.random() < p,
                        "people_fed_estimate": int(qty * np.random.uniform(2, 4)),
                        "day_of_week": cur.weekday(),
                        "month": cur.month,
                        "hour_posted": np.random.randint(6, 22),
                        "latitude": di.latitude,
                        "longitude": di.longitude,
                    }
                )
        self.donations = pd.DataFrame(rows)

        # external factors
        ext = []
        for i in range(365):
            cur = start + timedelta(days=i)
            ext.append(
                {
                    "date": cur,
                    "temperature": np.random.normal(25, 7),
                    "rainfall": max(0, np.random.exponential(2)),
                    "humidity": np.random.uniform(40, 90),
                    "festival_day": np.random.random() < 0.05,
                    "public_holiday": np.random.random() < 0.03,
                }
            )
        self.external = pd.DataFrame(ext)

        # volunteers + network
        vol = []
        vloc = [(12.9716, 77.5946), (13.0827, 80.2707), (11.0168, 76.9558), (15.2993, 74.1240)]
        for i in range(150):
            lat, lon = random.choice(vloc)
            vol.append(
                {
                    "volunteer_id": f"V_{i:03d}",
                    "latitude": lat + np.random.normal(0, 0.06),
                    "longitude": lon + np.random.normal(0, 0.06),
                }
            )
        self.volunteers = pd.DataFrame(vol)

        edges = []
        for _, dn in self.donations.iterrows():
            if dn.pickup_success:
                edges.append(
                    {
                        "donor_id": dn.donor_id,
                        "volunteer_id": np.random.choice(self.volunteers.volunteer_id),
                        "connection_strength": np.random.uniform(0.1, 1.0),
                    }
                )
        self.network = pd.DataFrame(edges)

    def _train(self):
        # merge for ML
        d = self.donations.copy()
        d["date_only"] = pd.to_datetime(d["date"]).dt.date
        e = self.external.copy()
        e["date_only"] = pd.to_datetime(e["date"]).dt.date
        ml = d.merge(
            e[["date_only", "temperature", "rainfall", "humidity", "festival_day", "public_holiday"]],
            on="date_only",
            how="left",
        )
        ml = ml.merge(
            self.donors[["donor_id", "size_score", "sustainability_score", "avg_daily_footfall", "type"]],
            on="donor_id",
            how="left",
        )
        self.ml = ml

        # demand model (RandomForest)
        agg = (
            ml.groupby("date_only")
            .agg(
                quantity_kg=("quantity_kg", "sum"),
                temperature=("temperature", "mean"),
                rainfall=("rainfall", "mean"),
                humidity=("humidity", "mean"),
                festival_day=("festival_day", "first"),
                public_holiday=("public_holiday", "first"),
            )
            .reset_index()
        )
        agg["day_of_week"] = pd.to_datetime(agg["date_only"]).dt.dayofweek
        agg["month"] = pd.to_datetime(agg["date_only"]).dt.month
        X = agg[["temperature", "rainfall", "humidity", "festival_day", "public_holiday", "day_of_week", "month"]]
        y = agg["quantity_kg"]
        self.demand = RandomForestRegressor(n_estimators=120, random_state=42).fit(X, y)

        # success model (XGBoost)
        le = LabelEncoder()
        fd = self.ml.copy()
        fd["food_type_encoded"] = le.fit_transform(fd["food_type"])
        fd["type_encoded"] = le.fit_transform(fd["type"])
        feats = [
            "quantity_kg",
            "hours_to_expire",
            "size_score",
            "sustainability_score",
            "temperature",
            "rainfall",
            "humidity",
            "day_of_week",
            "hour_posted",
            "food_type_encoded",
            "type_encoded",
        ]
        Xs = fd[feats].fillna(0)
        ys = fd["pickup_success"].astype(int)
        self.success = xgb.XGBClassifier(random_state=42)
        self.success.fit(Xs, ys)
        self.feat_names = feats

        # clustering
        donor = self.donors[["size_score", "sustainability_score", "avg_daily_footfall", "years_operating"]].fillna(0)
        sc = StandardScaler().fit(donor)
        km = KMeans(n_clusters=5, random_state=42).fit(sc.transform(donor))
        self.cluster = {"labels": km.labels_, "scaler": sc}

        # anomalies
        A = self.ml[["quantity_kg", "hours_to_expire", "people_fed_estimate", "day_of_week", "hour_posted"]].fillna(0)
        self.anom = IsolationForest(contamination=0.1, random_state=42).fit(A)


if not ss.ai_ready:
    with st.spinner("Loading AI models…"):
        ss.ai = FoodBridgeAI()
        ss.ai_ready = True

ai = ss.ai

# ===================== AUTH / NAV =====================
st.sidebar.title("🔑 Sign in")
choice = st.sidebar.selectbox(
    "I am a:",
    ["Visitor", "Restaurant/Donor", "Volunteer", "NGO/Organization", "Admin"],
    index=0,
)

ss.current_user_type = choice
ss.current_user = {
    "Restaurant/Donor": "Green Valley Restaurant",
    "Volunteer": "Rajesh Kumar",
    "NGO/Organization": "Hope Foundation",
    "Admin": "System Admin",
}.get(choice, None)

if choice != "Visitor":
    st.sidebar.success(f"Signed in as {ss.current_user}")

# Minimal, clean navigation
PAGES = ["🏠 Dashboard", "🍽 Donate", "🚗 Pickups", "🏢 Orgs", "📊 Analytics"]
page = st.sidebar.radio("Navigate:", PAGES, index=0)

# ===================== HELPERS =====================
def dist_km(a, b, c, d):
    return geodesic((a, b), (c, d)).kilometers


def urgency_emoji(exp):
    diff = (exp - dt.now()).total_seconds()
    return "🔴" if diff < 3600 else ("🟡" if diff < 7200 else "🟢")


def time_left(exp):
    diff = (exp - dt.now()).total_seconds()
    if diff < 0:
        return "⚠️ EXPIRED"
    h = int(diff // 3600)
    m = int((diff % 3600) // 60)
    return f"{h}h {m}m remaining"


# ===================== PAGES =====================
if page == "🏠 Dashboard":
    st.title("🍲 FoodBridge")
    st.caption("Reducing waste • Feeding people • Powered by AI")

    # KPIs
    available = len([d for d in ss.food_donations if d["status"] == "Available"])
    delivered_count = sum(
        [int(d["quantity"].split()[0]) if d["quantity"].split()[0].isdigit() else 0 for d in ss.food_donations if d["status"] == "Delivered"]
    )
    active_vols = len([v for v in ss.volunteers if v["availability"] == "Available"])
    cards = [
        ("🍽️", "Food Available", available),
        ("👥", "People Fed Today", delivered_count + 234),
        ("🚗", "Active Volunteers", active_vols),
        ("🌱", "Food Rescued", f"{len(ss.food_donations) * 2.5:.1f} kg"),
    ]
    c1, c2, c3, c4 = st.columns(4)
    for c, (icon, label, val) in zip([c1, c2, c3, c4], cards):
        c.markdown(
            f"<div class='metric-card'><h3>{icon}</h3><h2>{val}</h2><p>{label}</p></div>",
            unsafe_allow_html=True,
        )

    # Map
    st.subheader("🗺️ Live Map")
    m = folium.Map(location=[11.0168, 76.9558], zoom_start=12)
    for dn in ss.food_donations:
        if dn["status"] == "Available":
            color = {"🔴": "red", "🟡": "orange", "🟢": "green"}[urgency_emoji(dn["expiry_time"])]
            popup = f"""
            <b>{dn['donor_name']}</b><br>
            📞 {dn['donor_phone']}<br>
            🍽️ {dn['food_type']}<br>
            📦 {dn['quantity']}<br>
            ⏰ {time_left(dn['expiry_time'])}<br>
            📝 {dn['description']}
            """
            folium.Marker(
                [dn["donor_lat"], dn["donor_lng"]],
                popup=folium.Popup(popup, max_width=300),
                icon=folium.Icon(color=color, icon="cutlery", prefix="fa"),
            ).add_to(m)
    for org in ss.organizations:
        popup = f"""
        <b>{org['name']}</b><br>
        👤 {org['contact_person']}<br>
        📞 {org['phone']}<br>
        🏢 {org['type']}<br>
        👥 {org['beneficiaries']} people
        """
        folium.Marker(
            [org["location_lat"], org["location_lng"]],
            popup=folium.Popup(popup, max_width=300),
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(m)
    st_folium(m, width=820, height=520)

    # Urgent
    urgent = [
        d
        for d in ss.food_donations
        if d["status"] == "Available" and (d["expiry_time"] - dt.now()).total_seconds() < 3600
    ]
    if urgent:
        st.markdown("### 🚨 Urgent Pickups")
        for d in urgent:
            st.markdown(
                f"<div class='alert-urgent'><strong>{d['donor_name']}</strong> — {d['food_type']} ({d['quantity']})<br>"
                f"📍 {d['donor_address']}<br>"
                f"⏰ <strong>{time_left(d['expiry_time'])}</strong></div>",
                unsafe_allow_html=True,
            )

elif page == "🍽 Donate":
    if ss.current_user_type not in ["Restaurant/Donor", "Admin", "Visitor"]:
        st.warning("Please sign in as Donor to proceed.")
    st.header("🍽️ Donate Food")
    with st.form("food_form"):
        col1, col2 = st.columns(2)
        with col1:
            donor_name = st.text_input("Restaurant/Organization Name*", value=ss.current_user or "")
            donor_phone = st.text_input("Contact Phone*", value="+91-9876543210")
            donor_address = st.text_area("Pickup Address*", value="Race Course Road, Coimbatore")
        with col2:
            food_type = st.selectbox(
                "Food Type*",
                ["Cooked Meals", "Raw Ingredients", "Bakery Items", "Fruits & Vegetables", "Packaged Food", "Dairy Products"],
            )
            quantity = st.text_input("Quantity*", placeholder="e.g., 50 servings, 10kg, 20 pieces")
            food_category = st.selectbox("Food Category*", ["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"])
        description = st.text_area("Food Description*", placeholder="Describe the food items…")
        c3, c4 = st.columns(2)
        with c3:
            expiry_date = st.date_input("Expiry Date*", value=dt.now().date())
        with c4:
            expiry_time = st.time_input("Expiry Time*", value=time(18, 0))
        submitted = st.form_submit_button("🚀 Submit Donation", type="primary")
        if submitted:
            if all([donor_name, donor_phone, donor_address, food_type, quantity, description]):
                did = f"FD{len(ss.food_donations) + 1:03d}"
                exp = dt.combine(expiry_date, expiry_time)
                ss.food_donations.append(
                    {
                        "id": did,
                        "donor_name": donor_name,
                        "donor_phone": donor_phone,
                        "donor_address": donor_address,
                        "donor_lat": 11.0168,
                        "donor_lng": 76.9558,
                        "food_type": food_type,
                        "quantity": quantity,
                        "description": description,
                        "expiry_time": exp,
                        "status": "Available",
                        "created_at": dt.now(),
                        "claimed_by": None,
                        "delivered_to": None,
                        "special_instructions": "",
                        "food_category": food_category,
                    }
                )
                st.success(f"✅ Donation submitted! ID: {did}")
                st.balloons()
            else:
                st.error("Please fill all required fields marked with *")

elif page == "🚗 Pickups":
    if ss.current_user_type not in ["Volunteer", "Admin", "Visitor"]:
        st.info("Sign in as Volunteer to claim pickups.")
    st.header("🚗 Available Pickups")
    avail = [d for d in ss.food_donations if d["status"] == "Available"]
    if not avail:
        st.info("🎉 No pickups right now.")
    else:
        st.success(f"📋 {len(avail)} donations available")
        vlat, vlng = 11.0150, 76.9600
        for d in avail:
            dist = dist_km(vlat, vlng, d["donor_lat"], d["donor_lng"])
            st.markdown(
                f"<div class='food-card'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<h4>{urgency_emoji(d['expiry_time'])} {d['donor_name']} — {d['food_type']}</h4>"
                f"<span class='pill pill-available'>AVAILABLE</span>"
                f"</div>"
                f"<div style='margin:10px 0;'>"
                f"<strong>📦</strong> {d['quantity']} &nbsp; "
                f"<strong>📍</strong> {d['donor_address']} ({dist:.1f} km) &nbsp; "
                f"<strong>⏰</strong> {time_left(d['expiry_time'])}<br>"
                f"<strong>📝</strong> {d['description']}"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns([2, 2, 1])
            if c1.button("🚗 Claim Pickup", key=f"claim_{d['id']}"):
                for i, x in enumerate(ss.food_donations):
                    if x["id"] == d["id"]:
                        ss.food_donations[i]["status"] = "Claimed"
                        ss.food_donations[i]["claimed_by"] = "VOL001"
                        break
                st.success(f"✅ Claimed! Contact {d['donor_name']} at {d['donor_phone']}")
                st.rerun()
            if c2.button("📱 Contact Donor", key=f"contact_{d['id']}"):
                st.info(f"📞 Call: {d['donor_phone']}")
            if c3.button("🗺️ Directions", key=f"dir_{d['id']}"):
                st.info(f"📍 Navigate to: {d['donor_address']}")

elif page == "🏢 Orgs":
    st.header("🏢 Partner Organizations")
    for org in ss.organizations:
        st.markdown(
            f"<div class='food-card'>"
            f"<h4 style='color:{PRIMARY}'>{org['name']}</h4>"
            f"<div style='margin:10px 0;'>"
            f"<strong>👤</strong> {org['contact_person']} &nbsp; "
            f"<strong>📞</strong> {org['phone']}<br>"
            f"<strong>📍</strong> {org['address']}<br>"
            f"<strong>🏷️</strong> {org['type']} &nbsp; "
            f"<strong>👥</strong> {org['beneficiaries']} people<br>"
            f"<strong>📝</strong> {org['requirements']}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

elif page == "📊 Analytics":
    st.header("📊 Analytics & AI")
    tabs = st.tabs(["Overview", "Predict", "Segmentation", "Geo", "Anomalies", "Network", "Optimization"])

    # ------- Overview -------
    with tabs[0]:
        total = len(ai.donations)
        succ = ai.donations["pickup_success"].mean()
        kg = ai.donations["quantity_kg"].sum()
        ppl = ai.donations["people_fed_estimate"].sum()
        cA, cB, cC, cD = st.columns(4)
        for c, (t, v) in zip(
            [cA, cB, cC, cD],
            [
                ("🎯 Total Donations", f"{total:,}"),
                ("✅ Success Rate", f"{succ:.1%}"),
                ("🍽️ Food Rescued", f"{kg:,.0f} kg"),
                ("👥 People Fed", f"{ppl:,}"),
            ],
        ):
            c.markdown(f"<div class='metric-card'><h3>{t}</h3><h2>{v}</h2></div>", unsafe_allow_html=True)

        daily = ai.donations.groupby(ai.donations["date"].dt.date)["quantity_kg"].sum().reset_index()
        daily.columns = ["Date", "Quantity (kg)"]
        st.plotly_chart(px.line(daily, x="Date", y="Quantity (kg)", title="Daily Food Volume"), use_container_width=True)

    # ------- Predict -------
    with tabs[1]:
        st.subheader("Demand & Success Prediction")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Demand (kg) for a given day context")
            temp = st.slider("Temperature (°C)", 10, 40, 25)
            rain = st.slider("Rainfall (mm)", 0.0, 20.0, 2.0)
            hum = st.slider("Humidity (%)", 40, 90, 70)
            fest = st.checkbox("Festival Day")
            hol = st.checkbox("Public Holiday")
            day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            month = st.selectbox("Month", list(range(1, 13)))
            dmap = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            pred = ai.demand.predict(np.array([[temp, rain, hum, fest, hol, dmap[day], month]]))[0]
            st.markdown(
                f"<div class='prediction-card'><h3>🎯 Predicted Demand</h3><h2>{pred:.1f} kg</h2></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.caption("Pickup success probability for a post")
            qty = st.number_input("Quantity (kg)", 1.0, 100.0, 10.0)
            hrs = st.number_input("Hours to Expire", 1.0, 168.0, 24.0)
            size = st.slider("Donor Size", 1.0, 10.0, 5.0)
            sus = st.slider("Sustainability", 1.0, 10.0, 5.0)
            hposted = st.slider("Hour Posted", 0, 23, 12)
            ftypes = {"Prepared Food": 0, "Bakery Items": 1, "Fruits": 2, "Vegetables": 3, "Dairy": 4, "Packaged Food": 5, "Beverages": 6}
            dtypes = {"Restaurant": 0, "Bakery": 1, "Grocery Store": 2, "Catering": 3, "Hotel": 4, "Cafe": 5}
            ft = st.selectbox("Food Type", list(ftypes.keys()))
            dtp = st.selectbox("Donor Type", list(dtypes.keys()))
            prob = ai.success.predict_proba(
                np.array([[qty, hrs, size, sus, temp, rain, hum, dmap[day], hposted, ftypes[ft], dtypes[dtp]]])
            )[0][1]
            st.markdown(
                f"<div class='prediction-card'><h3>📈 Success Probability</h3><h2>{prob:.1%}</h2></div>",
                unsafe_allow_html=True,
            )

        try:
            imps = ai.success.feature_importances_
            names = [
                "Quantity",
                "Hours to Expire",
                "Donor Size",
                "Sustainability",
                "Temperature",
                "Rainfall",
                "Humidity",
                "Day",
                "Hour",
                "Food Type",
                "Donor Type",
            ]
            df_imp = pd.DataFrame({"Feature": names, "Importance": imps}).sort_values("Importance")
            st.plotly_chart(px.bar(df_imp, x="Importance", y="Feature", orientation="h", title="Feature Importance"), use_container_width=True)
        except Exception:
            st.info("Feature importance unavailable.")

    # ------- Segmentation -------
    with tabs[2]:
        cl = ai.donors.copy()
        cl["Cluster"] = ai.cluster["labels"]
        st.plotly_chart(
            px.scatter(
                cl,
                x="size_score",
                y="sustainability_score",
                color="Cluster",
                hover_data=["type", "avg_daily_footfall"],
                title="Donor Segmentation",
            ),
            use_container_width=True,
        )
        st.dataframe(
            cl.groupby("Cluster")
            .agg(
                size=("size_score", "mean"),
                sustainability=("sustainability_score", "mean"),
                footfall=("avg_daily_footfall", "mean"),
                years=("years_operating", "mean"),
            )
            .round(2)
        )


    # ------- Anomalies -------
    with tabs[4]:
        st.caption("Outlier detection on donation records")
        A = ai.ml[["quantity_kg", "hours_to_expire", "people_fed_estimate", "day_of_week", "hour_posted"]].fillna(0)
        scores = ai.anom.decision_function(A)
        preds = ai.anom.predict(A)
        tmp = ai.donations.copy()
        tmp["anomaly_score"] = scores
        tmp["is_anomaly"] = preds == -1
        st.metric("Anomalies", int(tmp["is_anomaly"].sum()))
        st.plotly_chart(
            px.histogram(tmp, x="anomaly_score", color="is_anomaly", nbins=50, title="Anomaly Score Distribution"),
            use_container_width=True,
        )

    # ------- Network -------
    with tabs[5]:
        st.caption("Donor–Volunteer connection network (synthetic)")
        G = nx.Graph()
        [G.add_node(d, type="donor") for d in ai.donors["donor_id"]]
        [G.add_node(v, type="volunteer") for v in ai.volunteers["volunteer_id"]]
        for _, e in ai.network.iterrows():
            G.add_edge(e["donor_id"], e["volunteer_id"], weight=e["connection_strength"])
        st.metric("Nodes", G.number_of_nodes())
        st.metric("Edges", G.number_of_edges())
        st.metric("Density", f"{nx.density(G):.4f}")
        degrees = [G.degree(n) for n in G.nodes()]
        st.plotly_chart(px.histogram(x=degrees, nbins=30, title="Degree Distribution"), use_container_width=True)

    # ------- Optimization -------
    with tabs[6]:
        st.caption("Simple route priority (demo)")
        active = ai.donations.head(20)
        eff = np.random.uniform(0.6, 0.9, len(active))
        dfp = pd.DataFrame(
            {
                "Donation_ID": active["donation_id"],
                "Location": active[["latitude", "longitude"]].apply(lambda x: f"({x['latitude']:.3f}, {x['longitude']:.3f})", axis=1),
                "Urgency_Hours": active["hours_to_expire"],
                "Quantity_kg": active["quantity_kg"],
                "Route_Efficiency": eff,
            }
        )
        dfp["Priority_Score"] = active["hours_to_expire"].max() - active["hours_to_expire"] + eff
        st.dataframe(dfp.sort_values("Priority_Score", ascending=False))


# ===================== SIDEBAR EXTRAS =====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Emergency Contacts")
st.sidebar.markdown("**Food Safety:** 1800-123-4567")
st.sidebar.markdown("**Volunteer Support:** +91-9876543299")
st.sidebar.markdown("**24/7 Helpline:** 1800-FOODBRIDGE")

if ss.current_user_type:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Today")
    st.sidebar.markdown(
        f"🍽️ **Available:** {len([d for d in ss.food_donations if d['status']=='Available'])}"
    )
    st.sidebar.markdown(
        f"🚗 **Active Volunteers:** {len([v for v in ss.volunteers if v['availability']=='Available'])}"
    )
    st.sidebar.markdown(f"🏢 **Orgs:** {len(ss.organizations)}")
    st.sidebar.markdown(
        f"⚡ **Urgent:** {len([d for d in ss.food_donations if d['status']=='Available' and (d['expiry_time']-dt.now()).total_seconds()<3600])}"
    )

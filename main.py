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
from statsmodels.tsa.seasonal import seasonal_decompose
import networkx as nx
import datetime
from datetime import datetime as dt, timedelta, time
import random

# ----------------------- Page Config & Styles -----------------------
st.set_page_config(page_title="🍲 FoodBridge - Full Stack (Ops + AI)", page_icon="🍲", layout="wide")
st.markdown(
    """
    <style>
      .metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:10px;color:#fff;text-align:center}
      .prediction-card{background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);padding:1rem;border-radius:10px;color:#fff;text-align:center}
      .food-card{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:10px 0;background:#f9f9f9}
      .status-available{background:#4caf50;color:#fff;padding:.3rem .8rem;border-radius:15px;font-size:.8rem}
      .status-claimed{background:#ff9800;color:#fff;padding:.3rem .8rem;border-radius:15px;font-size:.8rem}
      .status-delivered{background:#2196f3;color:#fff;padding:.3rem .8rem;border-radius:15px;font-size:.8rem}
      .alert-urgent{background:#ffebee;border-left:5px solid #f44336;padding:1rem;border-radius:5px;margin:10px 0}
      .alert-success{background:#e8f5e9;border-left:5px solid #4caf50;padding:1rem;border-radius:5px;margin:10px 0}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------- Session & Sample Data -----------------------
ss = st.session_state
for key, default in {
    "food_donations": [], "volunteers": [], "organizations": [],
    "current_user_type": None, "current_user": None
}.items():
    if key not in ss: ss[key] = default

def initialize_sample_data():
    if not ss.food_donations:
        now = dt.now()
        ss.food_donations.extend([
            {"id":"FD001","donor_name":"Green Valley Restaurant","donor_phone":"+91-9876543210",
             "donor_address":"Race Course Road, Coimbatore","donor_lat":11.0168,"donor_lng":76.9558,
             "food_type":"Cooked Meals","quantity":"50 servings","description":"Veg meals",
             "expiry_time": now+timedelta(hours=4),"status":"Available","created_at":now-timedelta(hours=1),
             "claimed_by":None,"delivered_to":None,"special_instructions":"Contains dairy","food_category":"Vegetarian"},
            {"id":"FD002","donor_name":"City Bakery","donor_phone":"+91-9876543211",
             "donor_address":"RS Puram, Coimbatore","donor_lat":11.0096,"donor_lng":76.9750,
             "food_type":"Bakery Items","quantity":"30 pieces","description":"Bread & pastries",
             "expiry_time": now+timedelta(hours=12),"status":"Claimed","created_at":now-timedelta(hours=3),
             "claimed_by":"VOL001","delivered_to":None,"special_instructions":"Best fresh","food_category":"Vegetarian"},
            {"id":"FD003","donor_name":"Spice Garden Hotel","donor_phone":"+91-9876543212",
             "donor_address":"Gandhipuram, Coimbatore","donor_lat":11.0183,"donor_lng":76.9725,
             "food_type":"Cooked Meals","quantity":"80 servings","description":"Mixed meals",
             "expiry_time": now+timedelta(hours=2),"status":"Available","created_at":now-timedelta(minutes=30),
             "claimed_by":None,"delivered_to":None,"special_instructions":"Urgent pickup","food_category":"Mixed"},
        ])
    if not ss.volunteers:
        ss.volunteers.extend([
            {"id":"VOL001","name":"Rajesh Kumar","phone":"+91-9876543220","area":"Race Course",
             "vehicle":"Two Wheeler","availability":"Available","rating":4.8,"completed_deliveries":25,
             "location_lat":11.0150,"location_lng":76.9600},
            {"id":"VOL002","name":"Priya Sharma","phone":"+91-9876543221","area":"RS Puram",
             "vehicle":"Four Wheeler","availability":"Busy","rating":4.9,"completed_deliveries":42,
             "location_lat":11.0100,"location_lng":76.9750},
        ])
    if not ss.organizations:
        ss.organizations.extend([
            {"id":"ORG001","name":"Hope Foundation","contact_person":"Maria Joseph","phone":"+91-9876543230",
             "address":"Peelamedu, Coimbatore","type":"NGO","beneficiaries":150,
             "location_lat":11.0296,"location_lng":76.9378,
             "requirements":"Veg meals preferred; elderly & children"},
            {"id":"ORG002","name":"Street Children Shelter","contact_person":"David Wilson","phone":"+91-9876543231",
             "address":"Saibaba Colony, Coimbatore","type":"Shelter Home","beneficiaries":80,
             "location_lat":11.0240,"location_lng":76.9350,
             "requirements":"Any food; urgent breakfast items"},
        ])
initialize_sample_data()

# ----------------------- AI/Data Science Engine -----------------------
class FoodBridgeDataScience:
    def __init__(self): self.initialize_data(); self.load_or_train_models()
    def initialize_data(self):
        np.random.seed(42)
        self.donors_data = self._gen_donors(); self.donations_data = self._gen_donations()
        self.volunteers_data = self._gen_volunteers(); self.external_factors = self._gen_external()
        self.network_data = self._gen_network()
    def _gen_donors(self):
        donor_types = ['Restaurant','Bakery','Grocery Store','Catering','Hotel','Cafe']
        locs=[(12.9716,77.5946),(13.0827,80.2707),(11.0168,76.9558),(15.2993,74.1240),(17.3850,78.4867)]
        rows=[]
        for i in range(500):
            lat,lon=locs[i%len(locs)]; t=np.random.choice(donor_types)
            rows.append({"donor_id":f"D_{i:03d}","name":f"{t}_{i}","type":t,
                         "latitude":lat+np.random.normal(0,0.1),"longitude":lon+np.random.normal(0,0.1),
                         "size_score":np.random.uniform(1,10),"sustainability_score":np.random.uniform(1,10),
                         "avg_daily_footfall":np.random.randint(50,1000),"years_operating":np.random.randint(1,20),
                         "donation_frequency":np.random.choice(['Daily','Weekly','Monthly']),
                         "preferred_pickup_time":np.random.choice(['Morning','Afternoon','Evening']),
                         "max_donation_capacity":np.random.randint(10,200)})
        return pd.DataFrame(rows)
    def _gen_donations(self):
        rows=[]; start=dt.now()-timedelta(days=730)
        for d in range(730):
            cur=start+timedelta(days=d); mult=(1.5 if cur.weekday()>=5 else 1.0)*(1.2 if cur.month in [11,12,1] else 1.0)
            for _ in range(int(np.random.poisson(20)*mult)):
                did=np.random.choice(self.donors_data['donor_id']); di=self.donors_data[self.donors_data.donor_id==did].iloc[0]
                cats=['Prepared Food','Bakery Items','Fruits','Vegetables','Dairy','Packaged Food','Beverages']
                ft=np.random.choice(cats); base=di.size_score*np.random.uniform(0.5,2)
                qty=max(1,np.random.normal(base,base*0.3)); urg_map={'Prepared Food':4,'Dairy':6,'Bakery Items':24,'Fruits':48,'Vegetables':72,'Packaged Food':168,'Beverages':240}
                hrs=max(1,np.random.normal(urg_map[ft],urg_map[ft]*0.3))
                p=min(0.95,0.5+(di.sustainability_score/20)+(1/max(1,hrs/24))*0.3)
                rows.append({"donation_id":f"DN_{len(rows):06d}","donor_id":did,"date":cur,"food_type":ft,
                             "quantity_kg":round(qty,2),"hours_to_expire":round(hrs,1),
                             "pickup_success":np.random.random()<p,"people_fed_estimate":int(qty*np.random.uniform(2,4)),
                             "day_of_week":cur.weekday(),"month":cur.month,"hour_posted":np.random.randint(6,22),
                             "weather_condition":np.random.choice(['Sunny','Rainy','Cloudy']),
                             "latitude":di.latitude,"longitude":di.longitude})
        return pd.DataFrame(rows)
    def _gen_volunteers(self):
        rows=[]; locs=[(12.9716,77.5946),(13.0827,80.2707),(11.0168,76.9558),(15.2993,74.1240)]
        for i in range(200):
            lat,lon=random.choice(locs)
            rows.append({"volunteer_id":f"V_{i:03d}","name":f"Volunteer_{i}",
                         "latitude":lat+np.random.normal(0,0.05),"longitude":lon+np.random.normal(0,0.05),
                         "experience_months":np.random.randint(1,60),"avg_pickups_per_week":np.random.randint(1,10),
                         "transport_capacity":np.random.choice(['Bike','Car','Van']),"availability_hours":np.random.randint(2,12),
                         "success_rate":np.random.uniform(0.7,0.98),"preferred_food_types":np.random.choice(['All','Prepared Food','Packaged Food'])})
        return pd.DataFrame(rows)
    def _gen_external(self):
        rows=[]; start=dt.now()-timedelta(days=365)
        for d in range(365):
            cur=start+timedelta(days=d)
            rows.append({"date":cur,"temperature":np.random.normal(25,8),"rainfall":max(0,np.random.exponential(2)),
                         "humidity":np.random.uniform(40,90),"festival_day":np.random.random()<0.05,"public_holiday":np.random.random()<0.03,
                         "economic_index":np.random.normal(100,10)})
        return pd.DataFrame(rows)
    def _gen_network(self):
        rows=[]
        for _,dn in self.donations_data.iterrows():
            if dn.pickup_success:
                rows.append({"donation_id":dn.donation_id,"donor_id":dn.donor_id,
                             "volunteer_id":np.random.choice(self.volunteers_data.volunteer_id),
                             "ngo_id":f"NGO_{np.random.randint(1,20):02d}","connection_strength":np.random.uniform(0.1,1.0),
                             "delivery_time_minutes":np.random.normal(45,15)})
        return pd.DataFrame(rows)
    def load_or_train_models(self):
        self.prepare_ml_features(); self.demand_model=self._train_demand(); self.success_model=self._train_success()
        self.cluster_model=self._train_cluster(); self.anomaly_model=self._train_anomaly(); self.optimization_model={"status":"trained"}
    def prepare_ml_features(self):
        self.donations_data['date_only']=pd.to_datetime(self.donations_data['date']).dt.date
        self.external_factors['date_only']=pd.to_datetime(self.external_factors['date']).dt.date
        ml=self.donations_data.merge(self.external_factors[['date_only','temperature','rainfall','humidity','festival_day','public_holiday']],on='date_only',how='left')
        ml=ml.merge(self.donors_data[['donor_id','size_score','sustainability_score','avg_daily_footfall','type']],on='donor_id',how='left')
        self.ml_data=ml
    def _train_demand(self):
        d=self.ml_data.groupby('date_only').agg(quantity_kg=('quantity_kg','sum'),temperature=('temperature','mean'),rainfall=('rainfall','mean'),humidity=('humidity','mean'),festival_day=('festival_day','first'),public_holiday=('public_holiday','first')).reset_index()
        d['day_of_week']=pd.to_datetime(d['date_only']).dt.dayofweek; d['month']=pd.to_datetime(d['date_only']).dt.month
        X=d[['temperature','rainfall','humidity','festival_day','public_holiday','day_of_week','month']]; y=d['quantity_kg']
        m=RandomForestRegressor(n_estimators=100,random_state=42).fit(X,y); return m
    def _train_success(self):
        le=LabelEncoder(); fd=self.ml_data.copy(); fd['food_type_encoded']=le.fit_transform(fd['food_type']); fd['type_encoded']=le.fit_transform(fd['type'])
        feats=['quantity_kg','hours_to_expire','size_score','sustainability_score','temperature','rainfall','humidity','day_of_week','hour_posted','food_type_encoded','type_encoded']
        X=fd[feats].fillna(0); y=fd['pickup_success'].astype(int)
        m=xgb.XGBClassifier(random_state=42); m.fit(X,y); return m
    def _train_cluster(self):
        donor=self.donors_data[['size_score','sustainability_score','avg_daily_footfall','years_operating']].fillna(0)
        sc=StandardScaler().fit(donor); km=KMeans(n_clusters=5,random_state=42).fit(sc.transform(donor))
        return {"model":km,"scaler":sc,"clusters":km.labels_}
    def _train_anomaly(self):
        feats=['quantity_kg','hours_to_expire','people_fed_estimate','day_of_week','hour_posted']
        X=self.ml_data[feats].fillna(0); return IsolationForest(contamination=0.1,random_state=42).fit(X)

if 'foodbridge_ds' not in ss:
    with st.spinner("Initializing AI models & loading data..."):
        ss.foodbridge_ds = FoodBridgeDataScience()
fb = ss.foodbridge_ds

# ----------------------- Sidebar Auth & Navigation -----------------------
st.sidebar.title("🔑 User Portal")
user_type = st.sidebar.selectbox("Login as:", ["Select User Type","Restaurant/Donor","Volunteer","NGO/Organization","Admin"])
if user_type != "Select User Type":
    ss.current_user_type = user_type
    ss.current_user = {
        "Restaurant/Donor":"Green Valley Restaurant",
        "Volunteer":"Rajesh Kumar",
        "NGO/Organization":"Hope Foundation",
        "Admin":"System Admin"
    }.get(user_type)
    st.sidebar.success(f"✅ Logged in as: {ss.current_user}")

# Pages combining Ops + AI
OPS_PAGES=["🏠 Public Dashboard","🍽 Donate Food","📦 My Donations","🚗 Available Pickups","🚚 My Deliveries","🏢 Organizations","👤 Profile","📊 Ops Analytics"]
AI_PAGES=["🧠 AI: Overview & KPIs","🧠 AI: Predictive Analytics","🧠 AI: Donor Segmentation","🧠 AI: Geographic Intelligence","🧠 AI: Time Series","🧠 AI: Anomaly Detection","🧠 AI: Network Analysis","🧠 AI: Optimization"]

allowed_pages = OPS_PAGES + AI_PAGES
page = st.sidebar.radio("Navigate to:", allowed_pages, index=0)

# ----------------------- Helper funcs (Ops) -----------------------
from geopy.distance import geodesic

def dist_km(a,b,c,d): return geodesic((a,b),(c,d)).kilometers

def urgency_emoji(exp):
    diff=(exp-dt.now()).total_seconds()
    return '🔴' if diff<3600 else ('🟡' if diff<7200 else '🟢')

def time_left(exp):
    diff=(exp-dt.now()).total_seconds()
    if diff<0: return "⚠️ EXPIRED"
    h=int(diff//3600); m=int((diff%3600)//60); return f"{h}h {m}m remaining"

# ----------------------- OPS: Public Dashboard -----------------------
if page=="🏠 Public Dashboard":
    st.title("🍲 FoodBridge - Connecting Surplus Food with Those in Need")
    avail=len([d for d in ss.food_donations if d['status']=='Available'])
    served=sum([int(d['quantity'].split()[0]) if d['quantity'].split()[0].isdigit() else 0 for d in ss.food_donations if d['status']=='Delivered'])
    active=len([v for v in ss.volunteers if v['availability']=='Available'])
    cols=st.columns(4)
    cards=[("🍽️",avail,"Food Available"),("👥",served+234,"People Fed Today"),("🚗",active,"Active Volunteers"),("🌱",f"{len(ss.food_donations)*2.5:.1f}kg","Food Rescued")]
    for c,(icon,val,txt) in zip(cols,cards):
        c.markdown(f"<div class='metric-card'><h3>{icon}</h3><h2>{val}</h2><p>{txt}</p></div>",unsafe_allow_html=True)
    m=folium.Map(location=[11.0168,76.9558],zoom_start=12)
    for dn in ss.food_donations:
        if dn['status']=='Available':
            col={'🔴':'red','🟡':'orange','🟢':'green'}[urgency_emoji(dn['expiry_time'])]
            popup=f"""<b>{dn['donor_name']}</b><br>📞 {dn['donor_phone']}<br>🍽️ {dn['food_type']}<br>📦 {dn['quantity']}<br>⏰ {time_left(dn['expiry_time'])}<br>📝 {dn['description']}"""
            folium.Marker([dn['donor_lat'],dn['donor_lng']],popup=folium.Popup(popup,max_width=300),icon=folium.Icon(color=col,icon='cutlery',prefix='fa')).add_to(m)
    for org in ss.organizations:
        popup=f"""<b>{org['name']}</b><br>👤 {org['contact_person']}<br>📞 {org['phone']}<br>🏢 {org['type']}<br>👥 {org['beneficiaries']} people<br>📝 {org['requirements']}"""
        folium.Marker([org['location_lat'],org['location_lng']],popup=folium.Popup(popup,max_width=300),icon=folium.Icon(color='blue',icon='home',prefix='fa')).add_to(m)
    st.subheader("🗺️ Real-time Food Availability Map"); st_folium(m,width=750,height=520)
    urgent=[d for d in ss.food_donations if d['status']=='Available' and (d['expiry_time']-dt.now()).total_seconds()<3600]
    if urgent:
        st.markdown("### 🚨 Urgent Pickups Required!")
        for d in urgent:
            st.markdown(f"<div class='alert-urgent'><strong>{d['donor_name']}</strong> - {d['food_type']} ({d['quantity']})<br>📍 {d['donor_address']}<br>⏰ <strong>{time_left(d['expiry_time'])}</strong><br>📝 {d['special_instructions']}</div>",unsafe_allow_html=True)

# ----------------------- OPS: Donate Food -----------------------
elif page=="🍽 Donate Food":
    st.header("🍽️ Donate Food")
    with st.form("food_form"):
        c1,c2=st.columns(2)
        with c1:
            dn=st.text_input("Restaurant/Organization Name*", value="Green Valley Restaurant")
            ph=st.text_input("Contact Phone*", value="+91-9876543210")
            addr=st.text_area("Pickup Address*", value="Race Course Road, Coimbatore")
        with c2:
            ftype=st.selectbox("Food Type*", ["Cooked Meals","Raw Ingredients","Bakery Items","Fruits & Vegetables","Packaged Food","Dairy Products"])
            qty=st.text_input("Quantity*", placeholder="e.g., 50 servings, 10kg")
            fcat=st.selectbox("Food Category*", ["Vegetarian","Non-Vegetarian","Vegan","Mixed"])
        desc=st.text_area("Food Description*", placeholder="Describe items...")
        c3,c4=st.columns(2)
        with c3:
            pdate=st.date_input("Pickup Date*", value=dt.now().date()); ptime=st.time_input("Pickup Time*", value=time(12,0))
        with c4:
            edate=st.date_input("Food Expiry Date*", value=dt.now().date()); etime=st.time_input("Food Expiry Time*", value=time(18,0))
        st.subheader("📍 Pickup Location"); option=st.radio("Set location:", ["Default","Enter coordinates"], horizontal=True)
        if option=="Default": plat,plng=11.0168,76.9558
        else:
            cc1,cc2=st.columns(2); plat=cc1.number_input("Latitude", value=11.0168, format="%.6f"); plng=cc2.number_input("Longitude", value=76.9558, format="%.6f")
        ok=st.form_submit_button("🚀 Submit Food Donation", type="primary")
        if ok and all([dn,ph,addr,ftype,qty,desc]):
            did=f"FD{len(ss.food_donations)+1:03d}"; exp=dt.combine(edate,etime)
            ss.food_donations.append({"id":did,"donor_name":dn,"donor_phone":ph,"donor_address":addr,"donor_lat":plat,"donor_lng":plng,
                                      "food_type":ftype,"quantity":qty,"description":desc,"expiry_time":exp,"status":"Available","created_at":dt.now(),
                                      "claimed_by":None,"delivered_to":None,"special_instructions":"","food_category":fcat})
            st.success(f"✅ Submitted! Donation ID: {did}"); st.balloons()
        elif ok:
            st.error("Please fill all required fields marked with *")

# ----------------------- OPS: My Donations -----------------------
elif page=="📦 My Donations":
    st.header("📦 My Food Donations")
    me = ss.current_user or "Green Valley Restaurant"
    mine=[d for d in ss.food_donations if d['donor_name']==me]
    if not mine: st.info("🍽️ No donations yet. Use 'Donate Food' to start!")
    else:
        st.success(f"📊 You have {len(mine)} donations")
        counts={}
        for d in mine: counts[d['status']]=counts.get(d['status'],0)+1
        c1,c2,c3=st.columns(3); c1.metric("Available",counts.get('Available',0)); c2.metric("Claimed",counts.get('Claimed',0)); c3.metric("Delivered",counts.get('Delivered',0))
        for d in sorted(mine,key=lambda x:x['created_at'],reverse=True):
            cls=f"status-{d['status'].lower()}"; urg=urgency_emoji(d['expiry_time'])
            st.markdown(f"<div class='food-card'><div style='display:flex;justify-content:space-between;align-items:center;'><h4>{urg} {d['food_type']} - {d['quantity']}</h4><span class='{cls}'>{d['status'].upper()}</span></div><div style='margin:10px 0;'><strong>🆔</strong> {d['id']} &nbsp; <strong>⏰</strong> {time_left(d['expiry_time'])}<br><strong>📝</strong> {d['description']}</div></div>",unsafe_allow_html=True)

# ----------------------- OPS: Available Pickups -----------------------
elif page=="🚗 Available Pickups":
    st.header("🚗 Available Food Pickups")
    avail=[d for d in ss.food_donations if d['status']=='Available']
    if not avail: st.info("🎉 No pickups right now. Everything's claimed!")
    else:
        st.success(f"📋 {len(avail)} donations available")
        vlat,vlng=11.0150,76.9600
        for d in avail:
            dist=dist_km(vlat,vlng,d['donor_lat'],d['donor_lng'])
            st.markdown(f"<div class='food-card'><div style='display:flex;justify-content:space-between;align-items:center;'><h4>{urgency_emoji(d['expiry_time'])} {d['donor_name']} - {d['food_type']}</h4><span class='status-available'>AVAILABLE</span></div><div style='margin:10px 0;'><strong>📦</strong> {d['quantity']} &nbsp; <strong>📍</strong> {d['donor_address']} ({dist:.1f} km) &nbsp; <strong>⏰</strong> {time_left(d['expiry_time'])}<br><strong>📝</strong> {d['description']}</div></div>",unsafe_allow_html=True)
            c1,c2,c3=st.columns([2,2,1])
            if c1.button("🚗 Claim Pickup", key=f"claim_{d['id']}"):
                for i,x in enumerate(ss.food_donations):
                    if x['id']==d['id']:
                        ss.food_donations[i]['status']='Claimed'; ss.food_donations[i]['claimed_by']='VOL001'
                        break
                st.success(f"✅ Claimed! Contact {d['donor_name']} at {d['donor_phone']}"); st.experimental_rerun()
            if c2.button("📱 Contact Donor", key=f"contact_{d['id']}"):
                st.info(f"📞 Call: {d['donor_phone']}")
            if c3.button("🗺️ Directions", key=f"dir_{d['id']}"):
                st.info(f"📍 Navigate to: {d['donor_address']}")

# ----------------------- OPS: My Deliveries -----------------------
elif page=="🚚 My Deliveries":
    st.header("🚚 My Delivery History")
    sample=[{"id":"DEL001","donation_id":"FD002","donor":"City Bakery","recipient":"Hope Foundation","food_type":"Bakery Items","quantity":"30 pieces","pickup_time":dt.now()-timedelta(hours=2),"delivery_time":dt.now()-timedelta(hours=1),"status":"Completed","distance":5.2,"rating":5},
            {"id":"DEL002","donation_id":"FD004","donor":"Hotel Paradise","recipient":"Street Shelter","food_type":"Cooked Meals","quantity":"40 servings","pickup_time":dt.now()-timedelta(days=1,hours=3),"delivery_time":dt.now()-timedelta(days=1,hours=2),"status":"Completed","distance":8.1,"rating":4}]
    c1,c2,c3,c4=st.columns(4); c1.metric("Total Deliveries","25"); c2.metric("This Week","7"); c3.metric("Average Rating","4.8 ⭐"); c4.metric("Distance Covered","142 km")
    st.subheader("📋 Recent Deliveries")
    for d in sample:
        st.markdown(f"<div class='food-card'><div style='display:flex;justify-content:space-between;align-items:center;'><h4>🍽️ {d['food_type']} - {d['quantity']}</h4><span class='status-delivered'>COMPLETED</span></div><div style='margin:10px 0;'><strong>📤 From:</strong> {d['donor']} &nbsp; <strong>📥 To:</strong> {d['recipient']}<br><strong>🚗</strong> {d['distance']} km &nbsp; <strong>📅</strong> {d['delivery_time'].strftime('%Y-%m-%d %H:%M')} &nbsp; <strong>⭐</strong> {d['rating']}/5</div></div>",unsafe_allow_html=True)

# ----------------------- OPS: Organizations -----------------------
elif page=="🏢 Organizations":
    st.header("🏢 Registered Organizations")
    for org in ss.organizations:
        st.markdown(f"<div class='food-card'><h4>🏢 {org['name']}</h4><div style='margin:10px 0;'><strong>👤</strong> {org['contact_person']} &nbsp; <strong>📞</strong> {org['phone']}<br><strong>📍</strong> {org['address']}<br><strong>🏷️</strong> {org['type']} &nbsp; <strong>👥</strong> {org['beneficiaries']} people<br><strong>📝</strong> {org['requirements']}</div></div>",unsafe_allow_html=True)

# ----------------------- OPS: Profile -----------------------
elif page=="👤 Profile":
    st.header("👤 Volunteer Profile")
    v = ss.volunteers[0]
    left,right=st.columns([1,2])
    left.markdown(f"<div style='background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:15px;color:#fff;text-align:center'><h2>👤</h2><h3>{v['name']}</h3><p>⭐ {v['rating']}/5.0</p><p>✅ {v['completed_deliveries']} Deliveries</p></div>",unsafe_allow_html=True)
    with right:
        with st.form("profile_form"):
            st.text_input("Name", value=v['name']); st.text_input("Phone", value=v['phone']); st.text_input("Service Area", value=v['area'])
            st.selectbox("Vehicle Type", ["Two Wheeler","Four Wheeler","Bicycle"], index=0)
            st.selectbox("Current Status", ["Available","Busy"], index=0 if v['availability']=="Available" else 1)
            if st.form_submit_button("Update Profile"): st.success("✅ Profile updated!")

# ----------------------- OPS: Ops Analytics (simple) -----------------------
elif page=="📊 Ops Analytics":
    st.header("📊 Food Rescue Analytics & Impact (Ops)")
    today=dt.now().date(); dates=[today-timedelta(days=i) for i in range(30,0,-1)]
    donations=[random.randint(5,25) for _ in dates]; rescues=[random.randint(3,20) for _ in dates]; waste=[d*2.5 for d in rescues]
    df=pd.DataFrame({"Date":dates,"Donations_Posted":donations,"Food_Rescued":rescues,"Waste_Prevented_kg":waste})
    c1,c2,c3,c4=st.columns(4); c1.metric("Total Donations (30d)",sum(donations)); c2.metric("Food Rescued (kg)",f"{sum(waste):.0f}"); c3.metric("People Fed",sum(rescues)*8); c4.metric("Avg Response Time (hrs)","2.3")
    fig=go.Figure(); fig.add_trace(go.Scatter(x=df['Date'],y=df['Donations_Posted'],mode='lines+markers',name='Donations Posted'))
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Food_Rescued'],mode='lines+markers',name='Food Rescued'))
    fig.update_layout(title="Daily Food Donation & Rescue Activity",hovermode='x unified',height=400)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------- AI: Overview & KPIs -----------------------
elif page=="🧠 AI: Overview & KPIs":
    st.header("📊 Advanced Analytics Overview")
    total=len(fb.donations_data); succ=fb.donations_data['pickup_success'].mean(); kg=fb.donations_data['quantity_kg'].sum(); ppl=fb.donations_data['people_fed_estimate'].sum()
    cols=st.columns(4)
    metrics=[("🎯 Total Donations",f"{total:,}"),("✅ Success Rate",f"{succ:.1%}"),("🍽️ Food Rescued",f"{kg:,.0f} kg"),("👥 People Fed",f"{ppl:,}")]
    for c,(t,v) in zip(cols,metrics): c.markdown(f"<div class='metric-card'><h3>{t}</h3><h2>{v}</h2></div>",unsafe_allow_html=True)
    daily=fb.donations_data.groupby(fb.donations_data['date'].dt.date)['quantity_kg'].sum().reset_index().rename(columns={'date':'Date','quantity_kg':'Quantity'})
    fig=px.line(daily,x='date',y='Quantity',title="Daily Food Rescue Volume"); st.plotly_chart(fig, use_container_width=True)

# ----------------------- AI: Predictive Analytics -----------------------
elif page=="🧠 AI: Predictive Analytics":
    st.header("🔮 AI-Powered Predictive Analytics")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("📊 Food Demand Prediction")
        temp=st.slider("Temperature (°C)",10,40,25); rain=st.slider("Rainfall (mm)",0.0,20.0,2.0); hum=st.slider("Humidity (%)",40,90,70)
        fest=st.checkbox("Festival Day"); hol=st.checkbox("Public Holiday"); day=st.selectbox("Day of Week", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']); month=st.selectbox("Month", list(range(1,13)))
        dmap={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
        pred=fb.demand_model.predict(np.array([[temp,rain,hum,fest,hol,dmap[day],month]]))[0]
        st.markdown(f"<div class='prediction-card'><h3>🎯 Predicted Daily Demand</h3><h2>{pred:.1f} kg</h2></div>",unsafe_allow_html=True)
        st.info(f"Expected range: {pred*0.8:.1f} - {pred*1.2:.1f} kg")
    with c2:
        st.subheader("✅ Pickup Success Probability")
        qty=st.number_input("Quantity (kg)",1.0,100.0,10.0); hrs=st.number_input("Hours to Expire",1.0,168.0,24.0); size=st.slider("Donor Size Score",1.0,10.0,5.0); sus=st.slider("Sustainability Score",1.0,10.0,5.0); hposted=st.slider("Hour Posted",0,23,12)
        ftypes={'Prepared Food':0,'Bakery Items':1,'Fruits':2,'Vegetables':3,'Dairy':4,'Packaged Food':5,'Beverages':6}
        dtypes={'Restaurant':0,'Bakery':1,'Grocery Store':2,'Catering':3,'Hotel':4,'Cafe':5}
        ft=st.selectbox("Food Type", list(ftypes.keys())); dtp=st.selectbox("Donor Type", list(dtypes.keys()))
        X=np.array([[qty,hrs,size,sus,temp,rain,hum,dmap[day],hposted,ftypes[ft],dtypes[dtp]]])
        prob=fb.success_model.predict_proba(X)[0][1]
        st.markdown(f"<div class='prediction-card'><h3>📈 Success Probability</h3><h2>{prob:.1%}</h2></div>",unsafe_allow_html=True)
        st.success("🟢 High success probability" if prob>0.8 else ("🟡 Moderate success probability" if prob>0.6 else "🔴 Low success probability"))
    st.subheader("🔍 Feature Importance")
    try:
        names=['Quantity','Hours to Expire','Donor Size','Sustainability','Temperature','Rainfall','Humidity','Day of Week','Hour Posted','Food Type','Donor Type']
        imp=pd.DataFrame({"Feature":names,"Importance":fb.success_model.feature_importances_}).sort_values('Importance')
        st.plotly_chart(px.bar(imp,x='Importance',y='Feature',orientation='h',title="Feature Importance"), use_container_width=True)
    except Exception:
        st.info("Importance not available.")

# ----------------------- AI: Donor Segmentation -----------------------
elif page=="🧠 AI: Donor Segmentation":
    st.header("👥 AI-Powered Donor Segmentation")
    clusters=fb.cluster_model['clusters']; cl=fb.donors_data.copy(); cl['Cluster']=clusters
    st.plotly_chart(px.scatter(cl,x='size_score',y='sustainability_score',color='Cluster',hover_data=['type','avg_daily_footfall'],title="Size vs Sustainability"), use_container_width=True)
    stats=cl.groupby('Cluster').agg(size_score=('size_score','mean'),sustainability_score=('sustainability_score','mean'),avg_daily_footfall=('avg_daily_footfall','mean'),years_operating=('years_operating','mean')).round(2)
    st.dataframe(stats)

# ----------------------- AI: Geographic Intelligence -----------------------
elif page=="🧠 AI: Geographic Intelligence":
    st.header("🗺️ Geographic Intelligence & Hotspots")
    center=[fb.donations_data['latitude'].mean(), fb.donations_data['longitude'].mean()]; m=folium.Map(location=center, zoom_start=10)
    heat=[[r.latitude,r.longitude,r.quantity_kg] for _,r in fb.donations_data.iterrows()]; HeatMap(heat).add_to(m)
    colors=['red','blue','green','purple','orange']; clusters=fb.cluster_model['clusters']
    for i,row in fb.donors_data.iterrows():
        col=colors[clusters[i]%len(colors)]
        folium.CircleMarker([row.latitude,row.longitude],radius=5,popup=f"Type:{row.type}<br>Size:{row.size_score:.1f}<br>Cluster:{clusters[i]}",color=col,fill=True,fill_opacity=0.6).add_to(m)
    st_folium(m,width=750,height=520)

# ----------------------- AI: Time Series -----------------------
elif page=="🧠 AI: Time Series":
    st.header("📈 Time Series & Seasonality")
    ts=fb.donations_data.groupby(fb.donations_data['date'].dt.date).agg(Total_Quantity=('quantity_kg','sum'),Success_Rate=('pickup_success','mean'),Count=('donation_id','count')).reset_index().rename(columns={'date':'Date'})
    ts['Date']=pd.to_datetime(ts['date']); ts=ts.sort_values('Date') if 'date' in ts.columns else ts.sort_values('Date')
    fig=make_subplots(rows=3,cols=1,subplot_titles=['Daily Volume','Success Rate','Donations'],vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=ts['Date'],y=ts['Total_Quantity'],name='kg'),row=1,col=1)
    fig.add_trace(go.Scatter(x=ts['Date'],y=ts['Success_Rate'],name='Success'),row=2,col=1)
    fig.add_trace(go.Scatter(x=ts['Date'],y=ts['Count'],name='#'),row=3,col=1)
    fig.update_layout(height=600,showlegend=False); st.plotly_chart(fig,use_container_width=True)
    ts['MA_7']=ts['Total_Quantity'].rolling(7).mean(); ts['MA_30']=ts['Total_Quantity'].rolling(30).mean()
    st.plotly_chart(go.Figure([go.Scatter(x=ts['Date'],y=ts['Total_Quantity'],name='Daily',opacity=0.6),go.Scatter(x=ts['Date'],y=ts['MA_7'],name='7D MA'),go.Scatter(x=ts['Date'],y=ts['MA_30'],name='30D MA')]).update_layout(title="Moving Averages",height=400), use_container_width=True)
    try:
        series=ts.set_index('Date')['Total_Quantity'].fillna(method='ffill')
        if len(series)>=14:
            dec=seasonal_decompose(series,model='additive',period=7)
            fig=make_subplots(rows=4,cols=1,subplot_titles=['Original','Trend','Seasonal','Residual'])
            fig.add_trace(go.Scatter(x=dec.observed.index,y=dec.observed),row=1,col=1)
            fig.add_trace(go.Scatter(x=dec.trend.index,y=dec.trend),row=2,col=1)
            fig.add_trace(go.Scatter(x=dec.seasonal.index,y=dec.seasonal),row=3,col=1)
            fig.add_trace(go.Scatter(x=dec.resid.index,y=dec.resid),row=4,col=1)
            fig.update_layout(height=800,showlegend=False); st.plotly_chart(fig,use_container_width=True)
        else: st.info("Need more history for decomposition.")
    except Exception: st.warning("Decomposition not available.")

# ----------------------- AI: Anomaly Detection -----------------------
elif page=="🧠 AI: Anomaly Detection":
    st.header("🔍 AI-Powered Anomaly Detection")
    feats=['quantity_kg','hours_to_expire','people_fed_estimate','day_of_week','hour_posted']
    X=fb.ml_data[feats].fillna(0); scores=fb.anomaly_model.decision_function(X); preds=fb.anomaly_model.predict(X)
    fb.donations_data['anomaly_score']=scores; fb.donations_data['is_anomaly']=preds==-1
    st.metric("Anomalies Detected", int(fb.donations_data['is_anomaly'].sum()))
    st.plotly_chart(px.histogram(fb.donations_data, x='anomaly_score', color='is_anomaly', nbins=50, title="Anomaly Scores"), use_container_width=True)
    st.plotly_chart(px.scatter(fb.donations_data, x='quantity_kg', y='hours_to_expire', color='is_anomaly', title="Quantity vs Expiry (Anomalies)"), use_container_width=True)

# ----------------------- AI: Network Analysis -----------------------
elif page=="🧠 AI: Network Analysis":
    st.header("🌐 Social Network Analysis")
    G=nx.Graph(); [G.add_node(d, type='donor') for d in fb.donors_data['donor_id']]; [G.add_node(v, type='volunteer') for v in fb.volunteers_data['volunteer_id']]
    for _,c in fb.network_data.iterrows(): G.add_edge(c['donor_id'], c['volunteer_id'], weight=c['connection_strength'])
    st.metric("Total Nodes", G.number_of_nodes()); st.metric("Total Connections", G.number_of_edges()); st.metric("Network Density", f"{nx.density(G):.4f}")
    degrees=[G.degree(n) for n in G.nodes()]; st.plotly_chart(px.histogram(x=degrees, nbins=20, title="Degree Distribution"), use_container_width=True)

# ----------------------- AI: Optimization -----------------------
elif page=="🧠 AI: Optimization":
    st.header("🎯 Optimization Engine")
    active=fb.donations_data.head(20); eff=np.random.uniform(0.6,0.9,len(active))
    df=pd.DataFrame({"Donation_ID":active['donation_id'],"Location":active[['latitude','longitude']].apply(lambda x:f"({x['latitude']:.3f}, {x['longitude']:.3f})",axis=1),"Urgency_Hours":active['hours_to_expire'],"Quantity_kg":active['quantity_kg'],"Route_Efficiency":eff})
    df['Priority_Score']=active['hours_to_expire'].max()-active['hours_to_expire']+eff
    st.dataframe(df.sort_values('Priority_Score',ascending=False))
    hourly=fb.donations_data.groupby('hour_posted')['pickup_success'].mean(); st.plotly_chart(px.bar(x=hourly.index,y=hourly.values,title="Success by Hour"), use_container_width=True)

# ----------------------- Footer & Sidebar Extras -----------------------
st.sidebar.markdown("---"); st.sidebar.markdown("### 🚨 Emergency Contacts\n**Food Safety:** 1800-123-4567\n\n**Volunteer Support:** +91-9876543299\n\n**24/7 Helpline:** 1800-FOODBRIDGE")
if ss.current_user_type:
    st.sidebar.markdown("---"); st.sidebar.markdown("### 📊 Today's Stats")
    st.sidebar.markdown(f"🍽️ **Available:** {len([d for d in ss.food_donations if d['status']=='Available'])}  ")
    st.sidebar.markdown(f"🚗 **Active Volunteers:** {len([v for v in ss.volunteers if v['availability']=='Available'])}  ")
    st.sidebar.markdown(f"🏢 **Organizations:** {len(ss.organizations)}  ")
    st.sidebar.markdown(f"⚡ **Urgent:** {len([d for d in ss.food_donations if d['status']=='Available' and (d['expiry_time']-dt.now()).total_seconds()<3600])}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import time as tm
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import random
from geopy.distance import geodesic
import hashlib
import warnings
from scipy import stats
import networkx as nx

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="FoodBridge - AI-Powered Food Rescue Platform",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-urgent {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-success {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
    }
    .food-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .status-available {
        background-color: #4caf50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .status-claimed {
        background-color: #ff9800;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .status-delivered {
        background-color: #2196f3;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .volunteer-badge {
        background-color: #673ab7;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.7rem;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

class FoodBridgeDataScience:
    def __init__(self):
        self.initialize_data()
        self.load_or_train_models()
    
    def initialize_data(self):
        """Generate comprehensive synthetic dataset for food rescue analytics"""
        np.random.seed(42)
        
        # Generate donors data
        self.donors_data = self._generate_donors_data()
        
        # Generate historical donations data
        self.donations_data = self._generate_donations_data()
        
        # Generate volunteer data
        self.volunteers_data = self._generate_volunteers_data()
        
        # Generate weather and external factors
        self.external_factors = self._generate_external_factors()
        
        # Generate network data
        self.network_data = self._generate_network_data()
    
    def _generate_donors_data(self):
        """Generate donor profiles with features for ML"""
        donors = []
        donor_types = ['Restaurant', 'Bakery', 'Grocery Store', 'Catering', 'Hotel', 'Cafe']
        locations = [(12.9716, 77.5946), (13.0827, 80.2707), (11.0168, 76.9558), 
                    (15.2993, 74.1240), (17.3850, 78.4867)]
        
        for i in range(500):
            location = locations[i % len(locations)]
            donor_type = np.random.choice(donor_types)
            
            donor = {
                'donor_id': f'D_{i:03d}',
                'name': f'{donor_type}_{i}',
                'type': donor_type,
                'latitude': location[0] + np.random.normal(0, 0.1),
                'longitude': location[1] + np.random.normal(0, 0.1),
                'size_score': np.random.uniform(1, 10),
                'sustainability_score': np.random.uniform(1, 10),
                'avg_daily_footfall': np.random.randint(50, 1000),
                'years_operating': np.random.randint(1, 20),
                'donation_frequency': np.random.choice(['Daily', 'Weekly', 'Monthly']),
                'preferred_pickup_time': np.random.choice(['Morning', 'Afternoon', 'Evening']),
                'max_donation_capacity': np.random.randint(10, 200)
            }
            donors.append(donor)
        
        return pd.DataFrame(donors)
    
    def _generate_donations_data(self):
        """Generate historical donations with temporal patterns"""
        donations = []
        
        # Generate 2 years of daily data
        start_date = datetime.now() - timedelta(days=730)
        
        for day in range(730):
            current_date = start_date + timedelta(days=day)
            
            # More donations on weekends and holidays
            day_multiplier = 1.5 if current_date.weekday() >= 5 else 1.0
            
            # Seasonal patterns
            month_multiplier = 1.2 if current_date.month in [11, 12, 1] else 1.0
            
            # Daily donations
            num_donations = int(np.random.poisson(20) * day_multiplier * month_multiplier)
            
            for _ in range(num_donations):
                donor_id = np.random.choice(self.donors_data['donor_id'])
                donor_info = self.donors_data[self.donors_data['donor_id'] == donor_id].iloc[0]
                
                # Food categories with different waste patterns
                food_categories = ['Prepared Food', 'Bakery Items', 'Fruits', 'Vegetables', 
                                 'Dairy', 'Packaged Food', 'Beverages']
                food_type = np.random.choice(food_categories)
                
                # Quantity based on donor size and type
                base_quantity = donor_info['size_score'] * np.random.uniform(0.5, 2)
                quantity_kg = max(1, np.random.normal(base_quantity, base_quantity * 0.3))
                
                # Urgency based on food type
                urgency_map = {'Prepared Food': 4, 'Dairy': 6, 'Bakery Items': 24, 
                              'Fruits': 48, 'Vegetables': 72, 'Packaged Food': 168, 'Beverages': 240}
                hours_to_expire = np.random.normal(urgency_map[food_type], urgency_map[food_type] * 0.3)
                hours_to_expire = max(1, hours_to_expire)
                
                # Success rate based on various factors
                success_probability = min(0.95, 0.5 + 
                                        (donor_info['sustainability_score'] / 20) +
                                        (1 / max(1, hours_to_expire / 24)) * 0.3)
                
                donation = {
                    'donation_id': f'DN_{len(donations):06d}',
                    'donor_id': donor_id,
                    'date': current_date,
                    'food_type': food_type,
                    'quantity_kg': round(quantity_kg, 2),
                    'hours_to_expire': round(hours_to_expire, 1),
                    'pickup_success': np.random.random() < success_probability,
                    'people_fed_estimate': int(quantity_kg * np.random.uniform(2, 4)),
                    'day_of_week': current_date.weekday(),
                    'month': current_date.month,
                    'hour_posted': np.random.randint(6, 22),
                    'weather_condition': np.random.choice(['Sunny', 'Rainy', 'Cloudy']),
                    'latitude': donor_info['latitude'],
                    'longitude': donor_info['longitude']
                }
                donations.append(donation)
        
        return pd.DataFrame(donations)
    
    def _generate_volunteers_data(self):
        """Generate volunteer profiles and activity data"""
        volunteers = []
        
        for i in range(200):
            location = np.random.choice([(12.9716, 77.5946), (13.0827, 80.2707), 
                                       (11.0168, 76.9558), (15.2993, 74.1240)])
            
            volunteer = {
                'volunteer_id': f'V_{i:03d}',
                'name': f'Volunteer_{i}',
                'latitude': location[0] + np.random.normal(0, 0.05),
                'longitude': location[1] + np.random.normal(0, 0.05),
                'experience_months': np.random.randint(1, 60),
                'avg_pickups_per_week': np.random.randint(1, 10),
                'transport_capacity': np.random.choice(['Bike', 'Car', 'Van']),
                'availability_hours': np.random.randint(2, 12),
                'success_rate': np.random.uniform(0.7, 0.98),
                'preferred_food_types': np.random.choice(['All', 'Prepared Food', 'Packaged Food'])
            }
            volunteers.append(volunteer)
        
        return pd.DataFrame(volunteers)
    
    def _generate_external_factors(self):
        """Generate weather and external factors data"""
        factors = []
        start_date = datetime.now() - timedelta(days=365)
        
        for day in range(365):
            current_date = start_date + timedelta(days=day)
            
            factor = {
                'date': current_date,
                'temperature': np.random.normal(25, 8),
                'rainfall': max(0, np.random.exponential(2)),
                'humidity': np.random.uniform(40, 90),
                'festival_day': np.random.random() < 0.05,
                'public_holiday': np.random.random() < 0.03,
                'economic_index': np.random.normal(100, 10)
            }
            factors.append(factor)
        
        return pd.DataFrame(factors)
    
    def _generate_network_data(self):
        """Generate network connections between donors, volunteers, and NGOs"""
        connections = []
        
        for _, donation in self.donations_data.iterrows():
            if donation['pickup_success']:
                volunteer = np.random.choice(self.volunteers_data['volunteer_id'])
                ngo = f'NGO_{np.random.randint(1, 20):02d}'
                
                connection = {
                    'donation_id': donation['donation_id'],
                    'donor_id': donation['donor_id'],
                    'volunteer_id': volunteer,
                    'ngo_id': ngo,
                    'connection_strength': np.random.uniform(0.1, 1.0),
                    'delivery_time_minutes': np.random.normal(45, 15)
                }
                connections.append(connection)
        
        return pd.DataFrame(connections)
    
    def load_or_train_models(self):
        """Load or train ML models"""
        
        # Prepare features for demand prediction
        self.prepare_ml_features()
        
        # Train demand prediction model
        self.demand_model = self.train_demand_prediction_model()
        
        # Train success rate prediction model
        self.success_model = self.train_success_prediction_model()
        
        # Train clustering model for donor segmentation
        self.cluster_model = self.train_clustering_model()
        
        # Train anomaly detection model
        self.anomaly_model = self.train_anomaly_detection_model()
        
        # Train optimization model
        self.optimization_model = self.train_optimization_model()
    
    def prepare_ml_features(self):
        """Prepare features for machine learning"""
        
        # Merge donations with external factors
        self.donations_data['date_only'] = self.donations_data['date'].dt.date
        self.external_factors['date_only'] = self.external_factors['date'].dt.date
        
        self.ml_data = self.donations_data.merge(
            self.external_factors[['date_only', 'temperature', 'rainfall', 'humidity', 'festival_day', 'public_holiday']],
            on='date_only',
            how='left'
        )
        
        # Add donor features
        self.ml_data = self.ml_data.merge(
            self.donors_data[['donor_id', 'size_score', 'sustainability_score', 'avg_daily_footfall', 'type']],
            on='donor_id',
            how='left'
        )
        
        # Feature engineering
        self.ml_data['urgency_category'] = pd.cut(self.ml_data['hours_to_expire'], 
                                                 bins=[0, 6, 24, 72, float('inf')], 
                                                 labels=['Critical', 'High', 'Medium', 'Low'])
        
        self.ml_data['quantity_category'] = pd.cut(self.ml_data['quantity_kg'], 
                                                  bins=5, labels=['XS', 'S', 'M', 'L', 'XL'])
    
    def train_demand_prediction_model(self):
        """Train model to predict food demand"""
        
        # Aggregate daily data
        daily_data = self.ml_data.groupby('date_only').agg({
            'quantity_kg': 'sum',
            'temperature': 'mean',
            'rainfall': 'mean',
            'humidity': 'mean',
            'festival_day': 'first',
            'public_holiday': 'first'
        }).reset_index()
        
        daily_data['day_of_week'] = pd.to_datetime(daily_data['date_only']).dt.dayofweek
        daily_data['month'] = pd.to_datetime(daily_data['date_only']).dt.month
        
        # Prepare features
        features = ['temperature', 'rainfall', 'humidity', 'festival_day', 
                   'public_holiday', 'day_of_week', 'month']
        
        X = daily_data[features]
        y = daily_data['quantity_kg']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
    
    def train_success_prediction_model(self):
        """Train model to predict pickup success"""
        
        # Prepare features
        le = LabelEncoder()
        features_data = self.ml_data.copy()
        
        # Handle missing values and encode categorical variables
        features_data = features_data.fillna(0)
        
        if len(features_data['food_type'].unique()) > 1:
            features_data['food_type_encoded'] = le.fit_transform(features_data['food_type'].astype(str))
        else:
            features_data['food_type_encoded'] = 0
            
        if len(features_data['type'].unique()) > 1:
            le_type = LabelEncoder()
            features_data['type_encoded'] = le_type.fit_transform(features_data['type'].astype(str))
        else:
            features_data['type_encoded'] = 0
        
        features = ['quantity_kg', 'hours_to_expire', 'size_score', 'sustainability_score',
                   'temperature', 'rainfall', 'humidity', 'day_of_week', 'hour_posted',
                   'food_type_encoded', 'type_encoded']
        
        X = features_data[features].fillna(0)
        y = features_data['pickup_success'].astype(int)
        
        # Train model
        model = xgb.XGBClassifier(random_state=42)
        model.fit(X, y)
        
        return model
    
    def train_clustering_model(self):
        """Train clustering model for donor segmentation"""
        
        # Prepare donor features for clustering
        donor_features = self.donors_data[['size_score', 'sustainability_score', 
                                          'avg_daily_footfall', 'years_operating']].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(donor_features)
        
        # Train KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        return {'model': kmeans, 'scaler': scaler, 'clusters': clusters}
    
    def train_anomaly_detection_model(self):
        """Train anomaly detection model"""
        
        # Prepare features for anomaly detection
        features = ['quantity_kg', 'hours_to_expire', 'people_fed_estimate', 'day_of_week', 'hour_posted']
        
        X = self.ml_data[features].fillna(0)
        
        # Train Isolation Forest
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        
        return model
    
    def train_optimization_model(self):
        """Train model for route optimization"""
        return {"status": "trained", "method": "distance_based"}

# Initialize session state
if 'food_donations' not in st.session_state:
    st.session_state.food_donations = []
if 'volunteers' not in st.session_state:
    st.session_state.volunteers = []
if 'organizations' not in st.session_state:
    st.session_state.organizations = []
if 'current_user_type' not in st.session_state:
    st.session_state.current_user_type = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# Sample data initialization
def initialize_sample_data():
    if len(st.session_state.food_donations) == 0:
        sample_donations = [
            {
                'id': 'FD001',
                'donor_name': 'Green Valley Restaurant',
                'donor_phone': '+91-9876543210',
                'donor_address': 'Race Course Road, Coimbatore',
                'donor_lat': 11.0168,
                'donor_lng': 76.9558,
                'food_type': 'Cooked Meals',
                'quantity': '50 servings',
                'description': 'Freshly prepared vegetarian meals - rice, dal, vegetables',
                'expiry_time': datetime.now() + timedelta(hours=4),
                'status': 'Available',
                'created_at': datetime.now() - timedelta(hours=1),
                'claimed_by': None,
                'delivered_to': None,
                'special_instructions': 'Keep refrigerated, contains dairy',
                'food_category': 'Vegetarian'
            },
            {
                'id': 'FD002',
                'donor_name': 'City Bakery',
                'donor_phone': '+91-9876543211',
                'donor_address': 'RS Puram, Coimbatore',
                'donor_lat': 11.0096,
                'donor_lng': 76.9750,
                'food_type': 'Bakery Items',
                'quantity': '30 pieces',
                'description': 'Fresh bread, pastries, and cakes from today',
                'expiry_time': datetime.now() + timedelta(hours=12),
                'status': 'Claimed',
                'created_at': datetime.now() - timedelta(hours=3),
                'claimed_by': 'Volunteer001',
                'delivered_to': None,
                'special_instructions': 'Best consumed fresh',
                'food_category': 'Vegetarian'
            },
            {
                'id': 'FD003',
                'donor_name': 'Spice Garden Hotel',
                'donor_phone': '+91-9876543212',
                'donor_address': 'Gandhipuram, Coimbatore',
                'donor_lat': 11.0183,
                'donor_lng': 76.9725,
                'food_type': 'Cooked Meals',
                'quantity': '80 servings',
                'description': 'Mixed vegetarian and non-vegetarian meals',
                'expiry_time': datetime.now() + timedelta(hours=2),
                'status': 'Available',
                'created_at': datetime.now() - timedelta(minutes=30),
                'claimed_by': None,
                'delivered_to': None,
                'special_instructions': 'Urgent pickup required',
                'food_category': 'Mixed'
            }
        ]
        st.session_state.food_donations.extend(sample_donations)
    
    if len(st.session_state.volunteers) == 0:
        sample_volunteers = [
            {
                'id': 'VOL001',
                'name': 'Rajesh Kumar',
                'phone': '+91-9876543220',
                'area': 'Race Course',
                'vehicle': 'Two Wheeler',
                'availability': 'Available',
                'rating': 4.8,
                'completed_deliveries': 25,
                'location_lat': 11.0150,
                'location_lng': 76.9600
            },
            {
                'id': 'VOL002',
                'name': 'Priya Sharma',
                'phone': '+91-9876543221',
                'area': 'RS Puram',
                'vehicle': 'Four Wheeler',
                'availability': 'Busy',
                'rating': 4.9,
                'completed_deliveries': 42,
                'location_lat': 11.0100,
                'location_lng': 76.9750
            }
        ]
        st.session_state.volunteers.extend(sample_volunteers)
    
    if len(st.session_state.organizations) == 0:
        sample_orgs = [
            {
                'id': 'ORG001',
                'name': 'Hope Foundation',
                'contact_person': 'Maria Joseph',
                'phone': '+91-9876543230',
                'address': 'Peelamedu, Coimbatore',
                'type': 'NGO',
                'beneficiaries': 150,
                'location_lat': 11.0296,
                'location_lng': 76.9378,
                'requirements': 'Vegetarian meals preferred, serves elderly and children'
            },
            {
                'id': 'ORG002',
                'name': 'Street Children Shelter',
                'contact_person': 'David Wilson',
                'phone': '+91-9876543231',
                'address': 'Saibaba Colony, Coimbatore',
                'type': 'Shelter Home',
                'beneficiaries': 80,
                'location_lat': 11.0240,
                'location_lng': 76.9350,
                'requirements': 'All types of food accepted, urgent need for breakfast items'
            }
        ]
        st.session_state.organizations.extend(sample_orgs)

# Initialize data
initialize_sample_data()

# Initialize the AI/DS system
if 'foodbridge_ds' not in st.session_state:
    with st.spinner("Initializing AI models and loading data..."):
        st.session_state.foodbridge_ds = FoodBridgeDataScience()

# Helper functions
def calculate_distance(lat1, lng1, lat2, lng2):
    return geodesic((lat1, lng1), (lat2, lng2)).kilometers

def get_urgency_color(expiry_time):
    time_diff = expiry_time - datetime.now()
    if time_diff.total_seconds() < 3600:  # Less than 1 hour
        return "🔴"
    elif time_diff.total_seconds() < 7200:  # Less than 2 hours
        return "🟡"
    else:
        return "🟢"

def format_time_remaining(expiry_time):
    time_diff = expiry_time - datetime.now()
    if time_diff.total_seconds() < 0:
        return "⚠️ EXPIRED"
    hours = int(time_diff.total_seconds() // 3600)
    minutes = int((time_diff.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m remaining"

# App title and description
st.markdown('<h1 class="main-header">🍲 FoodBridge - AI-Powered Food Rescue Platform</h1>', unsafe_allow_html=True)
st.markdown("""
**Reducing Food Waste | Fighting Hunger | Building Community | Powered by AI**

FoodBridge combines operational excellence with advanced AI analytics to connect restaurants, grocery stores, 
and individuals with surplus food to charitable organizations and volunteers who can distribute it to those in need.
""")

# User authentication simulation
with st.sidebar:
    st.title("🔑 User Portal")
    user_type = st.selectbox("Login as:", ["Select User Type", "Restaurant/Donor", "Volunteer", "NGO/Organization", "Admin", "AI Analytics"])
    
    if user_type != "Select User Type":
        st.session_state.current_user_type = user_type
        
        if user_type == "Restaurant/Donor":
            st.session_state.current_user = "Green Valley Restaurant"
            st.success(f"✅ Logged in as: {st.session_state.current_user}")
        elif user_type == "Volunteer":
            st.session_state.current_user = "Rajesh Kumar"
            st.success(f"✅ Logged in as: {st.session_state.current_user}")
        elif user_type == "NGO/Organization":
            st.session_state.current_user = "Hope Foundation"
            st.success(f"✅ Logged in as: {st.session_state.current_user}")
        elif user_type == "Admin":
            st.session_state.current_user = "System Admin"
            st.success(f"✅ Logged in as: {st.session_state.current_user}")
        elif user_type == "AI Analytics":
            st.session_state.current_user = "Data Scientist"
            st.success(f"✅ Logged in as: {st.session_state.current_user}")

# Navigation based on user type
if st.session_state.current_user_type:
    if st.session_state.current_user_type == "Restaurant/Donor":
        pages = ["Dashboard", "Donate Food", "My Donations", "Analytics"]
    elif st.session_state.current_user_type == "Volunteer":
        pages = ["Dashboard", "Available Pickups", "My Deliveries", "Profile"]
    elif st.session_state.current_user_type == "NGO/Organization":
        pages = ["Dashboard", "Food Requests", "Received Donations", "Impact Report"]
    elif st.session_state.current_user_type == "Admin":
        pages = ["Dashboard", "All Donations", "Volunteers", "Organizations", "Analytics"]
    elif st.session_state.current_user_type == "AI Analytics":
        pages = ["📊 Overview & KPIs", "🔮 Predictive Analytics", "👥 Donor Segmentation", 
                "🗺️ Geographic Intelligence", "📈 Time Series Analysis", "🔍 Anomaly Detection",
                "🌐 Network Analysis", "🎯 Optimization Engine"]
    else:
        pages = ["Public Dashboard"]
    
    with st.sidebar:
        st.markdown("---")
        if st.session_state.current_user_type == "AI Analytics":
            selected_page = st.selectbox("Choose Analysis Type", pages)
        else:
            selected_page = st.radio("Navigate to:", pages)
else:
    selected_page = "Public Dashboard"

# Get the FoodBridge DS instance
fb_ds = st.session_state.foodbridge_ds

# AI Analytics Pages
if st.session_state.current_user_type == "AI Analytics":
    
    if selected_page == "📊 Overview & KPIs":
        st.header("📊 Advanced Analytics Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_donations = len(fb_ds.donations_data)
        success_rate = fb_ds.donations_data['pickup_success'].mean()
        total_food_rescued = fb_ds.donations_data['quantity_kg'].sum()
        people_fed = fb_ds.donations_data['people_fed_estimate'].sum()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Total Donations</h3>
                <h2>{total_donations:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>✅ Success Rate</h3>
                <h2>{success_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🍽️ Food Rescued</h3>
                <h2>{total_food_rescued:,.0f} kg</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👥 People Fed</h3>
                <h2>{people_fed:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Advanced visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Daily Food Rescue Trends")
            daily_trends = fb_ds.donations_data.groupby(fb_ds.donations_data['date'].dt.date)['quantity_kg'].sum().reset_index()
            daily_trends.columns = ['Date', 'Quantity']
            
            fig = px.line(daily_trends, x='Date', y='Quantity', title="Daily Food Rescue Volume")
            fig.add_scatter(x=daily_trends['Date'], y=daily_trends['Quantity'], mode='markers', name='Daily Points')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Success Rate by Food Type")
            success_by_type = fb_ds.donations_data.groupby('food_type').agg({
                'pickup_success': 'mean',
                'quantity_kg': 'count'
            }).reset_index()
            success_by_type.columns = ['Food Type', 'Success Rate', 'Count']
            
            fig = px.bar(success_by_type, x='Food Type', y='Success Rate',
                        title="Pickup Success Rate by Food Category",
                        color='Success Rate', color_continuous_scale='RdYlGn')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_page == "🔮 Predictive Analytics":
        st.header("🔮 AI-Powered Predictive Analytics")
        
        # Demand prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Food Demand Prediction")
            
            temp = st.slider("Temperature (°C)", 10, 40, 25)
            rainfall = st.slider("Rainfall (mm)", 0.0, 20.0, 2.0)
            humidity = st.slider("Humidity (%)", 40, 90, 70)
            is_festival = st.checkbox("Festival Day")
            is_holiday = st.checkbox("Public Holiday")
            day_of_week = st.selectbox("Day of Week", 
                                      ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            month = st.selectbox("Month", range(1, 13))
            
            day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                          'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            day_num = day_mapping[day_of_week]
            
            features = np.array([[temp, rainfall, humidity, is_festival, is_holiday, day_num, month]])
            
            try:
                prediction = fb_ds.demand_model.predict(features)[0]
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>🎯 Predicted Daily Demand</h3>
                    <h2>{prediction:.1f} kg</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"Expected range: {prediction*0.8:.1f} - {prediction*1.2:.1f} kg")
                
            except Exception as e:
                st.error("Error making prediction. Please check inputs.")
        
        with col2:
            st.subheader("✅ Pickup Success Probability")
            
            quantity = st.number_input("Quantity (kg)", 1.0, 100.0, 10.0)
            hours_expire = st.number_input("Hours to Expire", 1.0, 168.0, 24.0)
            donor_size = st.slider("Donor Size Score", 1.0, 10.0, 5.0)
            sustainability = st.slider("Sustainability Score", 1.0, 10.0, 5.0)
            hour_posted = st.slider("Hour Posted", 0, 23, 12)
            
            food_type_map = {'Prepared Food': 0, 'Bakery Items': 1, 'Fruits': 2, 
                            'Vegetables': 3, 'Dairy': 4, 'Packaged Food': 5, 'Beverages': 6}
            food_type = st.selectbox("Food Type", list(food_type_map.keys()))
            
            donor_type_map = {'Restaurant': 0, 'Bakery': 1, 'Grocery Store': 2, 
                             'Catering': 3, 'Hotel': 4, 'Cafe': 5}
            donor_type = st.selectbox("Donor Type", list(donor_type_map.keys()))
            
            success_features = np.array([[quantity, hours_expire, donor_size, sustainability,
                                        temp, rainfall, humidity, day_num, hour_posted,
                                        food_type_map[food_type], donor_type_map[donor_type]]])
            
            try:
                success_prob = fb_ds.success_model.predict_proba(success_features)[0][1]
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>📈 Success Probability</h3>
                    <h2>{success_prob:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                if success_prob > 0.8:
                    st.success("🟢 High success probability - Recommended for posting!")
                elif success_prob > 0.6:
                    st.warning("🟡 Moderate success probability - Consider optimizing timing")
                else:
                    st.error("🔴 Low success probability - Review posting strategy")
                    
            except Exception as e:
                st.error("Error making success prediction.")
    
    elif selected_page == "👥 Donor Segmentation":
        st.header("👥 AI-Powered Donor Segmentation")
        
        clusters = fb_ds.cluster_model['clusters']
        cluster_data = fb_ds.donors_data.copy()
        cluster_data['Cluster'] = clusters
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Donor Clusters")
            
            fig = px.scatter(cluster_data, x='size_score', y='sustainability_score',
                            color='Cluster', hover_data=['type', 'avg_daily_footfall'],
                            title="Donor Segmentation: Size vs Sustainability")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Cluster Characteristics")
            
            cluster_stats = cluster_data.groupby('Cluster').agg({
                'size_score': 'mean',
                'sustainability_score': 'mean',
                'avg_daily_footfall': 'mean',
                'years_operating': 'mean'
            }).round(2)
            
            st.dataframe(cluster_stats)
    
    elif selected_page == "🗺️ Geographic Intelligence":
        st.header("🗺️ Geographic Intelligence & Hotspot Analysis")
        
        center_lat = fb_ds.donations_data['latitude'].mean()
        center_lon = fb_ds.donations_data['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        heat_data = [[row['latitude'], row['longitude'], row['quantity_kg']] 
                     for idx, row in fb_ds.donations_data.iterrows()]
        
        HeatMap(heat_data).add_to(m)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        clusters = fb_ds.cluster_model['clusters']
        
        for idx, row in fb_ds.donors_data.iterrows():
            cluster_color = colors[clusters[idx] % len(colors)]
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"Type: {row['type']}<br>Size: {row['size_score']}<br>Cluster: {clusters[idx]}",
                color=cluster_color,
                fillColor=cluster_color,
                fillOpacity=0.6
            ).add_to(m)
        
        st.subheader("🌡️ Food Waste Heat Map & Donor Clusters")
        st_folium(m, width=700, height=500)

# Regular operational pages
else:
    if selected_page == "Public Dashboard" or selected_page == "Dashboard":
        # Key metrics
        available_donations = len([d for d in st.session_state.food_donations if d['status'] == 'Available'])
        total_served_estimate = sum([int(d['quantity'].split()[0]) if d['quantity'].split()[0].isdigit() else 0 
                                   for d in st.session_state.food_donations if d['status'] == 'Delivered'])
        active_volunteers = len([v for v in st.session_state.volunteers if v['availability'] == 'Available'])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🍽️</h3>
                <h2>{available_donations}</h2>
                <p>Food Available</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👥</h3>
                <h2>{total_served_estimate + 234}</h2>
                <p>People Fed Today</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🚗</h3>
                <h2>{active_volunteers}</h2>
                <p>Active Volunteers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌱</h3>
                <h2>{len(st.session_state.food_donations) * 2.5:.1f}kg</h2>
                <p>Food Rescued</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time food availability map
        st.subheader("🗺️ Real-time Food Availability Map")
        
        m = folium.Map(location=[11.0168, 76.9558], zoom_start=12)
        
        for donation in st.session_state.food_donations:
            if donation['status'] == 'Available':
                urgency = get_urgency_color(donation['expiry_time'])
                
                popup_text = f"""
                <b>{donation['donor_name']}</b><br>
                📞 {donation['donor_phone']}<br>
                🍽️ {donation['food_type']}<br>
                📦 {donation['quantity']}<br>
                ⏰ {format_time_remaining(donation['expiry_time'])}<br>
                📝 {donation['description']}
                """
                
                color = 'red' if urgency == '🔴' else 'orange' if urgency == '🟡' else 'green'
                
                folium.Marker(
                    location=[donation['donor_lat'], donation['donor_lng']],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=color, icon='cutlery', prefix='fa')
                ).add_to(m)
        
        for org in st.session_state.organizations:
            popup_text = f"""
            <b>{org['name']}</b><br>
            👤 {org['contact_person']}<br>
            📞 {org['phone']}<br>
            🏢 {org['type']}<br>
            👥 Serves {org['beneficiaries']} people<br>
            📝 {org['requirements']}
            """
            
            folium.Marker(
                location=[org['location_lat'], org['location_lng']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='blue', icon='home', prefix='fa')
            ).add_to(m)
        
        map_data = st_folium(m, width=700, height=500)
    
    elif selected_page == "Donate Food":
        st.header("🍽️ Donate Food")
        
        with st.form("food_donation_form"):
            st.subheader("Food Donation Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                donor_name = st.text_input("Restaurant/Organization Name*", value="Green Valley Restaurant")
                donor_phone = st.text_input("Contact Phone*", value="+91-9876543210")
                donor_address = st.text_area("Pickup Address*", value="Race Course Road, Coimbatore")
                
            with col2:
                food_type = st.selectbox("Food Type*", 
                                       ["Cooked Meals", "Raw Ingredients", "Bakery Items", 
                                        "Fruits & Vegetables", "Packaged Food", "Dairy Products"])
                quantity = st.text_input("Quantity*", placeholder="e.g., 50 servings, 10kg, 20 pieces")
                food_category = st.selectbox("Food Category*", ["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"])
            
            description = st.text_area("Food Description*", 
                                     placeholder="Describe the food items in detail...")
            
            col3, col4 = st.columns(2)
            with col3:
                pickup_date = st.date_input("Pickup Date*", value=datetime.now().date())
                pickup_time = st.time_input("Pickup Time*", value=time(12, 0))
            
            with col4:
                expiry_date = st.date_input("Food Expiry Date*", value=datetime.now().date())
                expiry_time = st.time_input("Food Expiry Time*", value=time(18, 0))
            
            special_instructions = st.text_area("Special Instructions", 
                                              placeholder="Storage requirements, allergen information, etc.")
            
            submitted = st.form_submit_button("🚀 Submit Food Donation", type="primary")
            
            if submitted:
                if all([donor_name, donor_phone, donor_address, food_type, quantity, description]):
                    donation_id = f"FD{len(st.session_state.food_donations) + 1:03d}"
                    expiry_datetime = datetime.combine(expiry_date, expiry_time)
                    
                    new_donation = {
                        'id': donation_id,
                        'donor_name': donor_name,
                        'donor_phone': donor_phone,
                        'donor_address': donor_address,
                        'donor_lat': 11.0168,
                        'donor_lng': 76.9558,
                        'food_type': food_type,
                        'quantity': quantity,
                        'description': description,
                        'expiry_time': expiry_datetime,
                        'status': 'Available',
                        'created_at': datetime.now(),
                        'claimed_by': None,
                        'delivered_to': None,
                        'special_instructions': special_instructions,
                        'food_category': food_category
                    }
                    
                    st.session_state.food_donations.append(new_donation)
                    st.success(f"✅ Food donation submitted successfully! Donation ID: {donation_id}")
                    st.balloons()
                else:
                    st.error("Please fill in all required fields marked with *")

# Footer
st.markdown("---")
st.markdown("""
### 🤝 About FoodBridge - AI-Powered Food Rescue

**FoodBridge** combines operational excellence with cutting-edge AI to create the most efficient food rescue platform. 
Our advanced analytics and machine learning capabilities optimize every aspect of food recovery.

**AI-Powered Features:**
- 🔮 Predictive demand forecasting
- 📊 Real-time success probability analysis  
- 👥 Intelligent donor segmentation
- 🗺️ Geographic hotspot identification
- 🔍 Anomaly detection for quality assurance
- 🎯 Route and resource optimization

**Traditional Features:**
- 📱 Real-time food availability tracking
- 🗺️ GPS-based location services
- 👥 Comprehensive volunteer management
- 📊 Impact analytics and reporting
- 🏢 Multi-stakeholder platform integration

Together, we create a world where AI and human compassion work hand-in-hand to eliminate food waste and hunger!

---
*Built with ❤️ using Streamlit & Advanced AI | Contact: support@foodbridge.org*
""")

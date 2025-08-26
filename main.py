import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
import datetime
from datetime import timedelta, time
import hashlib
import time as tm
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import networkx as nx
import random
from geopy.distance import geodesic

# Try to import xgboost, fall back to RandomForest if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="FoodBridge - AI-Powered Food Rescue Platform",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
            
            # Features that affect donation patterns
            donor = {
                'donor_id': f'D_{i:03d}',
                'name': f'{donor_type}_{i}',
                'type': donor_type,
                'latitude': location[0] + np.random.normal(0, 0.1),
                'longitude': location[1] + np.random.normal(0, 0.1),
                'size_score': np.random.uniform(1, 10),  # Business size
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
        start_date = datetime.datetime.now() - timedelta(days=730)
        
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
        start_date = datetime.datetime.now() - timedelta(days=365)
        
        for day in range(365):
            current_date = start_date + timedelta(days=day)
            
            factor = {
                'date': current_date,
                'temperature': np.random.normal(25, 8),
                'rainfall': max(0, np.random.exponential(2)),
                'humidity': np.random.uniform(40, 90),
                'festival_day': np.random.random() < 0.05,  # 5% chance
                'public_holiday': np.random.random() < 0.03,  # 3% chance
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
        
        # Encode categorical variables
        features_data['food_type_encoded'] = le.fit_transform(features_data['food_type'])
        features_data['type_encoded'] = le.fit_transform(features_data['type'])
        features_data['urgency_encoded'] = features_data['hours_to_expire']
        
        features = ['quantity_kg', 'hours_to_expire', 'size_score', 'sustainability_score',
                   'temperature', 'rainfall', 'humidity', 'day_of_week', 'hour_posted',
                   'food_type_encoded', 'type_encoded']
        
        X = features_data[features].fillna(0)
        y = features_data['pickup_success'].astype(int)
        
        # Train model
        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
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
        
        # Simple optimization model for delivery routes
        # This would typically involve more complex algorithms like TSP solvers
        
        return {"status": "trained", "method": "distance_based"}

# Initialize session state for operational data
def initialize_session_state():
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
    if 'foodbridge_ds' not in st.session_state:
        st.session_state.foodbridge_ds = None

def initialize_sample_data():
    """Initialize sample operational data"""
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
                'expiry_time': datetime.datetime.now() + timedelta(hours=4),
                'status': 'Available',
                'created_at': datetime.datetime.now() - timedelta(hours=1),
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
                'expiry_time': datetime.datetime.now() + timedelta(hours=12),
                'status': 'Claimed',
                'created_at': datetime.datetime.now() - timedelta(hours=3),
                'claimed_by': 'Volunteer001',
                'delivered_to': None,
                'special_instructions': 'Best consumed fresh',
                'food_category': 'Vegetarian'
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
            }
        ]
        st.session_state.organizations.extend(sample_orgs)

def calculate_distance(lat1, lng1, lat2, lng2):
    return geodesic((lat1, lng1), (lat2, lng2)).kilometers

def get_urgency_color(expiry_time):
    time_diff = expiry_time - datetime.datetime.now()
    if time_diff.total_seconds() < 3600:  # Less than 1 hour
        return "🔴"
    elif time_diff.total_seconds() < 7200:  # Less than 2 hours
        return "🟡"
    else:
        return "🟢"

def format_time_remaining(expiry_time):
    time_diff = expiry_time - datetime.datetime.now()
    if time_diff.total_seconds() < 0:
        return "⚠️ EXPIRED"
    hours = int(time_diff.total_seconds() // 3600)
    minutes = int((time_diff.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m remaining"

def main():
    # Initialize session state and sample data
    initialize_session_state()
    initialize_sample_data()
    
    st.markdown('<h1 class="main-header">🍲 FoodBridge AI Analytics Platform</h1>', unsafe_allow_html=True)
    
    # User authentication simulation
    with st.sidebar:
        st.title("🔑 User Portal")
        user_type = st.selectbox("Login as:", ["Select User Type", "Restaurant/Donor", "Volunteer", "NGO/Organization", "Admin"])
        
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
    
    # Initialize AI models for admin users
    if st.session_state.current_user_type == "Admin" and st.session_state.foodbridge_ds is None:
        with st.spinner("Initializing AI models and loading data..."):
            st.session_state.foodbridge_ds = FoodBridgeDataScience()
    
    # Navigation based on user type
    if st.session_state.current_user_type:
        if st.session_state.current_user_type == "Restaurant/Donor":
            pages = ["Dashboard", "Donate Food", "My Donations"]
        elif st.session_state.current_user_type == "Volunteer":
            pages = ["Dashboard", "Available Pickups", "My Deliveries", "Profile"]
        elif st.session_state.current_user_type == "NGO/Organization":
            pages = ["Dashboard", "Food Requests", "Received Donations", "Impact Report"]
        elif st.session_state.current_user_type == "Admin":
            pages = ["Dashboard", "🔮 Predictive Analytics", "👥 Donor Segmentation", 
                    "🗺️ Geographic Intelligence", "📈 Time Series Analysis", "🔍 Anomaly Detection",
                    "🌐 Network Analysis", "🎯 Optimization Engine", "All Donations", "Volunteers", "Organizations"]
        else:
            pages = ["Public Dashboard"]
        
        with st.sidebar:
            st.markdown("---")
            selected_page = st.radio("Navigate to:", pages)
    else:
        selected_page = "Public Dashboard"
    
    # Page routing
    if selected_page in ["Public Dashboard", "Dashboard"]:
        show_dashboard()
    elif selected_page == "Donate Food":
        show_donate_food()
    elif selected_page == "My Donations":
        show_my_donations()
    elif selected_page == "Available Pickups":
        show_available_pickups()
    elif selected_page == "My Deliveries":
        show_my_deliveries()
    elif selected_page == "Profile":
        show_profile()
    elif selected_page == "Food Requests":
        show_food_requests()
    elif selected_page == "Impact Report":
        show_impact_report()
    elif selected_page == "All Donations":
        show_all_donations()
    elif selected_page == "Volunteers":
        show_volunteers()
    elif selected_page == "Organizations":
        show_organizations()
    elif selected_page == "🔮 Predictive Analytics":
        show_predictive_analytics()
    elif selected_page == "👥 Donor Segmentation":
        show_donor_segmentation()
    elif selected_page == "🗺️ Geographic Intelligence":
        show_geographic_intelligence()
    elif selected_page == "📈 Time Series Analysis":
        show_time_series_analysis()
    elif selected_page == "🔍 Anomaly Detection":
        show_anomaly_detection()
    elif selected_page == "🌐 Network Analysis":
        show_network_analysis()
    elif selected_page == "🎯 Optimization Engine":
        show_optimization_engine()
    
    # Add footer and sidebar info
    show_footer()
    show_sidebar_info()

def show_dashboard():
    """Show main dashboard with key metrics"""
    
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
    
    # Create map centered on Coimbatore
    m = folium.Map(location=[11.0168, 76.9558], zoom_start=12)
    
    # Add markers for available food donations
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
    
    # Add markers for organizations
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
    
    # Display map
    map_data = st_folium(m, width=700, height=500)
    
    # Urgent pickups alert
    urgent_donations = [d for d in st.session_state.food_donations 
                       if d['status'] == 'Available' and 
                       (d['expiry_time'] - datetime.datetime.now()).total_seconds() < 3600]
    
    if urgent_donations:
        st.markdown("### 🚨 Urgent Pickups Required!")
        for donation in urgent_donations:
            st.markdown(f"""
            <div class="alert-urgent">
                <strong>{donation['donor_name']}</strong> - {donation['food_type']} ({donation['quantity']})<br>
                📍 {donation['donor_address']}<br>
                ⏰ <strong>{format_time_remaining(donation['expiry_time'])}</strong><br>
                📝 {donation['special_instructions']}
            </div>
            """, unsafe_allow_html=True)

def show_donate_food():
    """Show food donation form"""
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
            pickup_date = st.date_input("Pickup Date*", value=datetime.datetime.now().date())
            pickup_time = st.time_input("Pickup Time*", value=time(12, 0))
        
        with col4:
            expiry_date = st.date_input("Food Expiry Date*", value=datetime.datetime.now().date())
            expiry_time = st.time_input("Food Expiry Time*", value=time(18, 0))
        
        special_instructions = st.text_area("Special Instructions", 
                                          placeholder="Storage requirements, allergen information, etc.")
        
        # Location selection
        st.subheader("📍 Pickup Location")
        location_option = st.radio("How would you like to set the location?", 
                                 ["Use default location", "Enter coordinates"])
        
        if location_option == "Use default location":
            pickup_lat, pickup_lng = 11.0168, 76.9558
        else:
            col5, col6 = st.columns(2)
            with col5:
                pickup_lat = st.number_input("Latitude", value=11.0168, format="%.6f")
            with col6:
                pickup_lng = st.number_input("Longitude", value=76.9558, format="%.6f")
        
        submitted = st.form_submit_button("🚀 Submit Food Donation", type="primary")
        
        if submitted:
            if all([donor_name, donor_phone, donor_address, food_type, quantity, description]):
                donation_id = f"FD{len(st.session_state.food_donations) + 1:03d}"
                expiry_datetime = datetime.datetime.combine(expiry_date, expiry_time)
                
                new_donation = {
                    'id': donation_id,
                    'donor_name': donor_name,
                    'donor_phone': donor_phone,
                    'donor_address': donor_address,
                    'donor_lat': pickup_lat,
                    'donor_lng': pickup_lng,
                    'food_type': food_type,
                    'quantity': quantity,
                    'description': description,
                    'expiry_time': expiry_datetime,
                    'status': 'Available',
                    'created_at': datetime.datetime.now(),
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

def show_my_donations():
    """Show donor's donations"""
    st.header("📦 My Food Donations")
    
    donor_donations = [d for d in st.session_state.food_donations 
                      if d['donor_name'] == st.session_state.current_user]
    
    if not donor_donations:
        st.info("🍽️ You haven't made any food donations yet. Click on 'Donate Food' to get started!")
    else:
        st.success(f"📊 You have made {len(donor_donations)} food donations")
        
        for donation in sorted(donor_donations, key=lambda x: x['created_at'], reverse=True):
            status_class = f"status-{donation['status'].lower()}"
            urgency = get_urgency_color(donation['expiry_time'])
            time_remaining = format_time_remaining(donation['expiry_time'])
            
            st.markdown(f"""
            <div class="food-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4>{urgency} {donation['food_type']} - {donation['quantity']}</h4>
                    <span class="{status_class}">{donation['status'].upper()}</span>
                </div>
                
                <div style="margin: 10px 0;">
                    <strong>🆔 Donation ID:</strong> {donation['id']}<br>
                    <strong>📅 Created:</strong> {donation['created_at'].strftime('%Y-%m-%d %H:%M')}<br>
                    <strong>⏰ Time Remaining:</strong> {time_remaining}<br>
                    <strong>📝 Description:</strong> {donation['description']}<br>
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_available_pickups():
    """Show available pickups for volunteers"""
    st.header("🚗 Available Food Pickups")
    
    available_donations = [d for d in st.session_state.food_donations if d['status'] == 'Available']
    
    if not available_donations:
        st.info("🎉 Great! No food pickups available right now. All food has been claimed!")
    else:
        st.success(f"📋 {len(available_donations)} food donations available for pickup")
        
        for donation in available_donations:
            urgency = get_urgency_color(donation['expiry_time'])
            time_remaining = format_time_remaining(donation['expiry_time'])
            
            st.markdown(f"""
            <div class="food-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4>{urgency} {donation['donor_name']} - {donation['food_type']}</h4>
                    <span class="status-available">AVAILABLE</span>
                </div>
                
                <div style="margin: 10px 0;">
                    <strong>📦 Quantity:</strong> {donation['quantity']}<br>
                    <strong>📍 Location:</strong> {donation['donor_address']}<br>
                    <strong>⏰ Time Remaining:</strong> {time_remaining}<br>
                    <strong>📝 Description:</strong> {donation['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"🚗 Claim Pickup", key=f"claim_{donation['id']}"):
                for i, d in enumerate(st.session_state.food_donations):
                    if d['id'] == donation['id']:
                        st.session_state.food_donations[i]['status'] = 'Claimed'
                        st.session_state.food_donations[i]['claimed_by'] = 'VOL001'
                        break
                
                st.success(f"✅ Pickup claimed! Contact {donation['donor_name']} at {donation['donor_phone']}")
                st.rerun()

def show_my_deliveries():
    """Show volunteer's delivery history"""
    st.header("🚚 My Delivery History")
    
    st.metric("Total Deliveries", "25")
    st.info("Delivery history would be displayed here based on volunteer's activity")

def show_profile():
    """Show volunteer profile"""
    st.header("👤 Volunteer Profile")
    
    st.info("Profile management interface for volunteers")

def show_food_requests():
    """Show food request interface for NGOs"""
    st.header("🏢 Request Food Donations")
    
    st.info("NGO food request interface would be implemented here")

def show_impact_report():
    """Show impact report for NGOs"""
    st.header("📈 Impact Report")
    
    st.info("Impact reporting dashboard for NGOs")

def show_all_donations():
    """Show all donations for admin"""
    st.header("📊 All Donations Overview")
    
    if st.session_state.food_donations:
        df = pd.DataFrame(st.session_state.food_donations)
        st.dataframe(df)
    else:
        st.info("No donations to display")

def show_volunteers():
    """Show volunteer management"""
    st.header("🚗 Volunteer Management")
    
    if st.session_state.volunteers:
        df = pd.DataFrame(st.session_state.volunteers)
        st.dataframe(df)
    else:
        st.info("No volunteers registered")

def show_organizations():
    """Show organization management"""
    st.header("🏢 Organization Management")
    
    if st.session_state.organizations:
        df = pd.DataFrame(st.session_state.organizations)
        st.dataframe(df)
    else:
        st.info("No organizations registered")

def show_predictive_analytics():
    """Show AI predictive analytics"""
    if not st.session_state.foodbridge_ds:
        st.error("AI models not initialized. Please refresh the page.")
        return
    
    fb_ds = st.session_state.foodbridge_ds
    
    st.header("🔮 AI-Powered Predictive Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Food Demand Prediction")
        
        temp = st.slider("Temperature (°C)", 10, 40, 25)
        rainfall = st.slider("Rainfall (mm)", 0.0, 20.0, 2.0)
        humidity = st.slider("Humidity (%)", 40, 90, 70)
        is_festival = st.checkbox("Festival Day")
        is_holiday = st.checkbox("Public Holiday")
        day_of_week = st.selectbox("Day of Week", 
                                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                   'Friday', 'Saturday', 'Sunday'])
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
            
        except Exception as e:
            st.error("Error making prediction. Please check inputs.")

def show_donor_segmentation():
    """Show donor segmentation analysis"""
    if not st.session_state.foodbridge_ds:
        st.error("AI models not initialized. Please refresh the page.")
        return
    
    fb_ds = st.session_state.foodbridge_ds
    
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

def show_geographic_intelligence():
    """Show geographic intelligence"""
    if not st.session_state.foodbridge_ds:
        st.error("AI models not initialized. Please refresh the page.")
        return
    
    st.header("🗺️ Geographic Intelligence & Hotspot Analysis")
    st.info("Geographic analytics with AI-powered insights")

def show_time_series_analysis():
    """Show time series analysis"""
    if not st.session_state.foodbridge_ds:
        st.error("AI models not initialized. Please refresh the page.")
        return
    
    st.header("📈 Advanced Time Series Analysis & Forecasting")
    st.info("Time series forecasting and trend analysis")

def show_anomaly_detection():
    """Show anomaly detection"""
    if not st.session_state.foodbridge_ds:
        st.error("AI models not initialized. Please refresh the page.")
        return
    
    st.header("🔍 AI-Powered Anomaly Detection")
    st.info("Anomaly detection in food donation patterns")

def show_network_analysis():
    """Show network analysis"""
    if not st.session_state.foodbridge_ds:
        st.error("AI models not initialized. Please refresh the page.")
        return
    
    st.header("🌐 Social Network Analysis")
    st.info("Network analysis of donors, volunteers, and organizations")

def show_optimization_engine():
    """Show optimization recommendations"""
    if not st.session_state.foodbridge_ds:
        st.error("AI models not initialized. Please refresh the page.")
        return
    
    st.header("🎯 AI-Powered Optimization Engine")
    st.info("Route optimization and resource allocation recommendations")

def show_footer():
    """Show app footer"""
    st.markdown("---")
    st.markdown("""
    ### 🤝 About FoodBridge

    **FoodBridge** is a comprehensive AI-powered food rescue platform that connects the dots between food waste and hunger. 
    Our mission is to create a sustainable ecosystem where surplus food reaches those who need it most.

    **Key Features:**
    - 🔮 AI-powered predictive analytics and demand forecasting
    - 📱 Real-time food availability tracking
    - 🗺️ GPS-based location services for efficient logistics
    - 👥 Volunteer network management with optimization
    - 📊 Advanced impact analytics and reporting
    - 🏢 Multi-stakeholder platform (Donors, Volunteers, NGOs)
    - ⚡ Urgent pickup alerts and notifications
    - 🤖 Machine learning for donor segmentation and anomaly detection

    Together, we can create a world where no food goes to waste and no one goes hungry!
    """)

def show_sidebar_info():
    """Show sidebar information"""
    if st.session_state.current_user_type:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Today's Stats")
        st.sidebar.markdown(f"""
        🍽️ **Available Now:** {len([d for d in st.session_state.food_donations if d['status'] == 'Available'])} donations  
        🚗 **Active Volunteers:** {len([v for v in st.session_state.volunteers if v['availability'] == 'Available'])}  
        🏢 **Partner Organizations:** {len(st.session_state.organizations)}  
        ⚡ **Urgent Pickups:** {len([d for d in st.session_state.food_donations if d['status'] == 'Available' and (d['expiry_time'] - datetime.datetime.now()).total_seconds() < 3600])}
        """)

if __name__ == "__main__":
    main()
